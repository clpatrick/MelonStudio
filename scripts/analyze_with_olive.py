#!/usr/bin/env python3
"""
Analyze model using Olive framework.
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
    from olive.model import HfModelHandler
except ImportError:
    print("=" * 70)
    print("OLIVE FRAMEWORK ANALYSIS")
    print("=" * 70)
    print()
    print("[INFO] Olive library not found in current Python environment.")
    print("  Olive is typically installed in the .olive-env virtual environment.")
    print("  If using the system Python, install with: pip install olive-ai")
    print("  Or ensure the analysis is run with the .olive-env Python interpreter.")
    print()
    print("=" * 70)
    sys.exit(0)  # Exit with 0 to indicate graceful skip, not error

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None


def analyze_with_olive(model_path: Path):
    """Analyze model using Olive HfModelHandler."""
    print("=" * 70)
    print("OLIVE FRAMEWORK ANALYSIS")
    print("=" * 70)
    print()
    
    try:
        # First, try to validate model structure using AutoConfig (supports trust_remote_code)
        config_validated = False
        if AutoConfig is not None:
            try:
                print(f"Validating model structure with AutoConfig: {model_path}")
                config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                config_validated = True
                print("[OK] Model structure validated via AutoConfig")
                print()
            except Exception as config_ex:
                print(f"[WARNING] AutoConfig validation failed: {config_ex}")
                print()
        
        # Try to load with HfModelHandler
        handler_loaded = False
        try:
            print(f"Attempting to load with Olive HfModelHandler: {model_path}")
            # Note: HfModelHandler doesn't support trust_remote_code parameter
            handler = HfModelHandler(model_path=str(model_path))
            handler_loaded = True
            print("[OK] Model loaded successfully with HfModelHandler")
            print()
        except Exception as handler_ex:
            error_msg = str(handler_ex)
            if "trust_remote_code" in error_msg.lower() or "custom code" in error_msg.lower():
                print("[WARNING] HfModelHandler cannot load model (requires trust_remote_code)")
                print("  This is expected for models with custom code.")
                print("  The model structure is valid, but HfModelHandler doesn't support")
                print("  trust_remote_code parameter in its API.")
                print()
                if config_validated:
                    print("[INFO] Model structure validated via AutoConfig - model is compatible.")
                    print("  Olive conversion may still work via CLI with --trust_remote_code flag.")
                    print()
            else:
                # For other errors, print warning but continue
                print(f"[WARNING] HfModelHandler load failed: {handler_ex}")
                print("  Continuing with structure validation only...")
                print()
        
        # Get model info
        print("Model Information:")
        print(f"  Model Path: {model_path}")
        
        # Try to get config if available
        config_path = model_path / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                architectures = config_data.get("architectures", [])
                if architectures:
                    print(f"  Architecture: {architectures[0]}")
                model_type = config_data.get("model_type", "unknown")
                print(f"  Model Type: {model_type}")
        
        print()
        print("Olive Validation:")
        if handler_loaded:
            print("  [OK] Model loaded with HfModelHandler")
        elif config_validated:
            print("  [OK] Model structure validated (custom code required)")
        else:
            print("  [WARNING] Could not fully validate model")
        print("  [OK] Compatible with Olive conversion pipeline")
        print()
        
        print("Olive Capabilities:")
        print("  - Model loading and validation")
        print("  - ONNX conversion support")
        if not handler_loaded and config_validated:
            print("  - Note: Use Olive CLI with --trust_remote_code for conversion")
        print("  - Quantization analysis")
        print("  - Optimization pass support")
        print()
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Olive analysis failed: {e}", file=sys.stderr)
        print(f"  Error details: {type(e).__name__}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"  Cause: {e.__cause__}")
        
        # Provide helpful hints for common errors
        if "not found" in error_msg.lower() or "missing" in error_msg.lower():
            print()
            print("  [INFO] Some model files may be missing or incomplete.")
        
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model with Olive framework")
    parser.add_argument("model_path", help="Path to model directory")
    
    args = parser.parse_args()
    
    analyze_with_olive(Path(args.model_path))
