#!/usr/bin/env python3
"""
Enhanced model analysis with layer type mapping, MoE/Mamba-2/GQA analysis,
memory breakdown, conversion compatibility, and hybrid partition recommendations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure stdout/stderr can handle Unicode on Windows
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Fallback: use ASCII-safe characters only
        pass

try:
    from transformers import AutoConfig
except ImportError:
    print("ERROR: transformers library not found. Install with: pip install transformers", file=sys.stderr)
    sys.exit(1)


def analyze_layer_types(config: Dict, num_layers: int) -> Dict[int, str]:
    """Map layer indices to layer types based on architecture."""
    layer_type_map = {}
    
    # Check for MoE configuration
    moe_config = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    num_experts = moe_config if moe_config else 0
    
    # Check for Mamba configuration
    has_mamba = hasattr(config, "mamba_config") or "mamba" in str(config).lower()
    
    # Check for GQA configuration
    num_kv_heads = getattr(config, "num_key_value_heads", None) or \
                   getattr(config, "num_kv_heads", None) or \
                   getattr(config, "num_attention_heads", None)
    num_heads = getattr(config, "num_attention_heads", None) or \
                getattr(config, "n_head", None) or 0
    is_gqa = num_kv_heads and num_heads and num_kv_heads < num_heads
    
    # Architecture-specific layer distribution
    # For Nemotron-3-Nano: 23 MoE + 23 Mamba-2 + 6 GQA = 52 layers
    # This is a heuristic - actual distribution may vary
    
    if num_experts > 0:
        # Assume MoE layers are in the middle section
        moe_start = num_layers // 4
        moe_end = moe_start + (num_layers // 2)
        for i in range(moe_start, min(moe_end, num_layers)):
            layer_type_map[i] = "MoE"
    
    if has_mamba:
        # Assume Mamba layers are in another section
        mamba_start = num_layers // 2
        mamba_end = mamba_start + (num_layers // 2)
        for i in range(mamba_start, min(mamba_end, num_layers)):
            if i not in layer_type_map:  # Don't overwrite MoE
                layer_type_map[i] = "Mamba-2"
    
    if is_gqa:
        # GQA layers typically at the end
        gqa_start = max(0, num_layers - 6)
        for i in range(gqa_start, num_layers):
            if i not in layer_type_map:
                layer_type_map[i] = "GQA"
    
    # Fill remaining with standard transformer
    for i in range(num_layers):
        if i not in layer_type_map:
            layer_type_map[i] = "Transformer"
    
    return layer_type_map


def analyze_moe(config: Dict) -> Dict:
    """Analyze MoE (Mixture of Experts) configuration."""
    moe_info = {
        "has_moe": False,
        "num_experts": 0,
        "num_routed_experts": 0,
        "num_shared_experts": 0,
        "experts_per_token": 0,
        "routing_type": "unknown"
    }
    
    num_experts = getattr(config, "num_experts", None) or \
                  getattr(config, "num_local_experts", None) or 0
    num_routed = getattr(config, "num_experts_per_tok", None) or \
                 getattr(config, "num_experts_per_tok", None) or 0
    
    if num_experts > 0:
        moe_info["has_moe"] = True
        moe_info["num_experts"] = num_experts
        moe_info["num_routed_experts"] = num_experts - 1  # Usually one shared
        moe_info["num_shared_experts"] = 1
        moe_info["experts_per_token"] = num_routed if num_routed > 0 else 6  # Default 6
        moe_info["routing_type"] = getattr(config, "router_aux_loss_coef", None) and "aux_loss" or "standard"
    
    return moe_info


def analyze_mamba(config: Dict) -> Dict:
    """Analyze Mamba-2 configuration."""
    mamba_info = {
        "has_mamba": False,
        "state_space_dim": 0,
        "conv_kernel": 0,
        "use_selective_scan": False
    }
    
    # Check for Mamba-specific attributes
    if hasattr(config, "mamba_config") or "mamba" in str(config).lower():
        mamba_info["has_mamba"] = True
        mamba_info["state_space_dim"] = getattr(config, "ssm_state_size", None) or 16
        mamba_info["conv_kernel"] = getattr(config, "conv_kernel", None) or 4
        mamba_info["use_selective_scan"] = True
    
    return mamba_info


def analyze_gqa(config: Dict) -> Dict:
    """Analyze GQA (Grouped Query Attention) configuration."""
    gqa_info = {
        "has_gqa": False,
        "num_heads": 0,
        "num_kv_heads": 0,
        "num_groups": 0,
        "head_dim": 0
    }
    
    num_heads = getattr(config, "num_attention_heads", None) or \
                getattr(config, "n_head", None) or 0
    num_kv_heads = getattr(config, "num_key_value_heads", None) or \
                   getattr(config, "num_kv_heads", None) or num_heads
    
    if num_kv_heads < num_heads:
        gqa_info["has_gqa"] = True
        gqa_info["num_heads"] = num_heads
        gqa_info["num_kv_heads"] = num_kv_heads
        gqa_info["num_groups"] = num_heads // num_kv_heads if num_kv_heads > 0 else 1
        hidden_size = getattr(config, "hidden_size", None) or \
                      getattr(config, "n_embd", None) or 0
        gqa_info["head_dim"] = hidden_size // num_heads if num_heads > 0 else 0
    
    return gqa_info


def calculate_memory_breakdown(model_path: Path, layer_type_map: Dict[int, str], 
                               total_size_gb: float, num_layers: int) -> Dict:
    """Calculate memory breakdown by layer type."""
    breakdown = {
        "by_type": {},
        "per_layer_avg": {},
        "kv_cache_estimate": {}
    }
    
    # Count layers by type
    type_counts = {}
    for layer_type in layer_type_map.values():
        type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
    
    # Estimate size per layer type (rough heuristic)
    avg_layer_size = total_size_gb / num_layers if num_layers > 0 else 0
    
    for layer_type, count in type_counts.items():
        breakdown["by_type"][layer_type] = {
            "count": count,
            "total_size_gb": avg_layer_size * count,
            "avg_size_gb": avg_layer_size
        }
    
    # KV cache estimates (per token, in bytes)
    # Rough estimate: 2 * num_layers * hidden_size * num_kv_heads * head_dim * dtype_size
    # This is simplified - actual depends on architecture
    
    return breakdown


def analyze_conversion_compatibility(config: Dict, layer_type_map: Dict[int, str]) -> Dict:
    """Analyze conversion compatibility and challenges."""
    compatibility = {
        "overall": "compatible",
        "challenges": [],
        "custom_ops_needed": [],
        "recommendations": []
    }
    
    # Check for MoE
    moe_layers = [i for i, t in layer_type_map.items() if t == "MoE"]
    if moe_layers:
        compatibility["challenges"].append("MoE routing logic requires custom ONNX ops or post-processing")
        compatibility["custom_ops_needed"].append("Expert routing/selection")
        compatibility["recommendations"].append("Consider using ONNX Runtime GenAI builder with MoE support")
    
    # Check for Mamba
    mamba_layers = [i for i, t in layer_type_map.items() if t == "Mamba-2"]
    if mamba_layers:
        compatibility["challenges"].append("Mamba-2 state space operations may need custom ONNX ops")
        compatibility["custom_ops_needed"].append("Selective scan operations")
        compatibility["recommendations"].append("Verify ONNX Runtime GenAI supports Mamba-2 state ops")
    
    # Check for GQA
    gqa_layers = [i for i, t in layer_type_map.items() if t == "GQA"]
    if gqa_layers:
        compatibility["challenges"].append("GQA attention structure is standard but requires grouped KV handling")
        compatibility["recommendations"].append("GQA should convert cleanly with standard attention ops")
    
    if not compatibility["challenges"]:
        compatibility["overall"] = "fully_compatible"
    elif len(compatibility["challenges"]) <= 1:
        compatibility["overall"] = "mostly_compatible"
    else:
        compatibility["overall"] = "requires_custom_ops"
    
    return compatibility


def suggest_hybrid_partitions(layer_type_map: Dict[int, str], total_size_gb: float,
                               num_layers: int, vram_budgets: List[float] = [16.0, 24.0, 32.0]) -> List[Dict]:
    """Suggest optimal hybrid partition points."""
    recommendations = []
    
    avg_layer_size = total_size_gb / num_layers if num_layers > 0 else 0
    
    for vram_budget in vram_budgets:
        # Calculate how many layers fit in VRAM
        max_gpu_layers = int(vram_budget / avg_layer_size) if avg_layer_size > 0 else num_layers
        
        # Prefer keeping MoE layers on GPU (they're compute-intensive)
        # But they're also memory-intensive, so balance is needed
        
        # Strategy: Keep early layers (including MoE if possible) on GPU
        # Split at a natural boundary (not in middle of MoE block)
        
        split_layer = min(max_gpu_layers, num_layers - 1)
        
        # Adjust to avoid splitting in middle of layer type groups
        layer_types = [layer_type_map.get(i, "Transformer") for i in range(num_layers)]
        
        # Find a good split point (prefer boundaries between layer types)
        for i in range(split_layer, max(0, split_layer - 5), -1):
            if i < num_layers - 1:
                current_type = layer_types[i]
                next_type = layer_types[i + 1]
                if current_type != next_type:
                    split_layer = i + 1
                    break
        
        gpu_layers = split_layer
        cpu_layers = num_layers - split_layer
        gpu_size = avg_layer_size * gpu_layers
        cpu_size = avg_layer_size * cpu_layers
        
        # Calculate score (higher is better)
        # Prefer: more GPU layers, balanced split, type boundaries
        score = (gpu_layers / num_layers) * 0.4 + \
                (1.0 - abs(gpu_size - vram_budget) / vram_budget) * 0.4 + \
                (1.0 if split_layer < num_layers and layer_types[split_layer-1] != layer_types[split_layer] else 0.5) * 0.2
        
        recommendations.append({
            "vram_budget_gb": vram_budget,
            "split_layer": split_layer,
            "gpu_layers": gpu_layers,
            "cpu_layers": cpu_layers,
            "gpu_size_gb": gpu_size,
            "cpu_size_gb": cpu_size,
            "score": score,
            "reason": f"Split at layer {split_layer} for {vram_budget}GB VRAM budget"
        })
    
    # Sort by score (best first)
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    return recommendations


def analyze_enhanced(model_path: Path):
    """Run enhanced analysis."""
    print("=" * 70)
    print("ENHANCED MODEL ANALYSIS")
    print("=" * 70)
    print()
    
    # Load config
    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Basic info
    num_layers = getattr(config, "num_hidden_layers", None) or \
                 getattr(config, "num_layers", None) or \
                 getattr(config, "n_layer", None) or 0
    
    # Calculate total size
    total_size_bytes = 0
    for file in model_path.glob("*.safetensors"):
        total_size_bytes += file.stat().st_size
    for file in model_path.glob("*.bin"):
        total_size_bytes += file.stat().st_size
    
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    # Layer type mapping
    print("LAYER TYPE MAPPING")
    print("-" * 70)
    layer_type_map = analyze_layer_types(config, num_layers)
    type_counts = {}
    for layer_type in layer_type_map.values():
        type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
    
    for layer_type, count in sorted(type_counts.items()):
        print(f"  {layer_type}: {count} layers")
    print()
    
    # Show layer distribution
    print("Layer Distribution (first 10 and last 10):")
    for i in range(min(10, num_layers)):
        print(f"  Layer {i}: {layer_type_map.get(i, 'Unknown')}")
    if num_layers > 20:
        print("  ...")
        for i in range(max(10, num_layers - 10), num_layers):
            print(f"  Layer {i}: {layer_type_map.get(i, 'Unknown')}")
    print()
    
    # MoE Analysis
    print("MoE (Mixture of Experts) ANALYSIS")
    print("-" * 70)
    moe_info = analyze_moe(config)
    if moe_info["has_moe"]:
        print(f"  [OK] MoE Enabled")
        print(f"  Total Experts: {moe_info['num_experts']}")
        print(f"  Routed Experts: {moe_info['num_routed_experts']}")
        print(f"  Shared Experts: {moe_info['num_shared_experts']}")
        print(f"  Experts per Token: {moe_info['experts_per_token']}")
        print(f"  Routing Type: {moe_info['routing_type']}")
    else:
        print("  [INFO] No MoE detected")
    print()
    
    # Mamba-2 Analysis
    print("Mamba-2 ANALYSIS")
    print("-" * 70)
    mamba_info = analyze_mamba(config)
    if mamba_info["has_mamba"]:
        print(f"  [OK] Mamba-2 Enabled")
        print(f"  State Space Dim: {mamba_info['state_space_dim']}")
        print(f"  Conv Kernel: {mamba_info['conv_kernel']}")
        print(f"  Selective Scan: {mamba_info['use_selective_scan']}")
    else:
        print("  [INFO] No Mamba-2 detected")
    print()
    
    # GQA Analysis
    print("GQA (Grouped Query Attention) ANALYSIS")
    print("-" * 70)
    gqa_info = analyze_gqa(config)
    if gqa_info["has_gqa"]:
        print(f"  [OK] GQA Enabled")
        print(f"  Attention Heads: {gqa_info['num_heads']}")
        print(f"  KV Heads: {gqa_info['num_kv_heads']}")
        print(f"  Groups: {gqa_info['num_groups']}")
        print(f"  Head Dimension: {gqa_info['head_dim']}")
    else:
        print("  [INFO] Standard MHA (Multi-Head Attention)")
    print()
    
    # Memory Breakdown
    print("MEMORY BREAKDOWN")
    print("-" * 70)
    breakdown = calculate_memory_breakdown(model_path, layer_type_map, total_size_gb, num_layers)
    for layer_type, info in breakdown["by_type"].items():
        print(f"  {layer_type}:")
        print(f"    Layers: {info['count']}")
        print(f"    Total Size: {info['total_size_gb']:.2f} GB")
        print(f"    Avg per Layer: {info['avg_size_gb']:.2f} GB")
    print()
    
    # Conversion Compatibility
    print("CONVERSION COMPATIBILITY")
    print("-" * 70)
    compatibility = analyze_conversion_compatibility(config, layer_type_map)
    print(f"  Overall Status: {compatibility['overall'].upper().replace('_', ' ')}")
    if compatibility["challenges"]:
        print("  Challenges:")
        for challenge in compatibility["challenges"]:
            print(f"    - {challenge}")
    if compatibility["custom_ops_needed"]:
        print("  Custom Ops Needed:")
        for op in compatibility["custom_ops_needed"]:
            print(f"    - {op}")
    if compatibility["recommendations"]:
        print("  Recommendations:")
        for rec in compatibility["recommendations"]:
            print(f"    - {rec}")
    print()
    
    # Hybrid Partition Recommendations
    print("HYBRID PARTITION RECOMMENDATIONS")
    print("-" * 70)
    recommendations = suggest_hybrid_partitions(layer_type_map, total_size_gb, num_layers)
    for i, rec in enumerate(recommendations, 1):
        print(f"  Option {i} (Score: {rec['score']:.2f}):")
        print(f"    VRAM Budget: {rec['vram_budget_gb']} GB")
        print(f"    Split at Layer: {rec['split_layer']}")
        print(f"    GPU Layers: 0-{rec['gpu_layers']-1} ({rec['gpu_layers']} layers, ~{rec['gpu_size_gb']:.2f} GB)")
        print(f"    CPU Layers: {rec['split_layer']}-{num_layers-1} ({rec['cpu_layers']} layers, ~{rec['cpu_size_gb']:.2f} GB)")
        print(f"    Reason: {rec['reason']}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced model analysis with architecture details")
    parser.add_argument("model_path", help="Path to model directory (containing config.json)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    if args.verbose:
        os.environ["VERBOSE"] = "1"
    
    analyze_enhanced(Path(args.model_path))
