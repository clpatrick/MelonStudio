import onnx
import onnxruntime
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

# Try to import HQQ config, otherwise fallback to default with asymmetric
try:
    from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig
    HAS_HQQ = True
    print("HQQ (High Quality Quantization) is available.")
except ImportError:
    HAS_HQQ = False
    print("HQQ not available, using Asymmetric RTN.")

def quantize_cpu_hq(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path} to {output_model_path} (High Quality 4-bit)...")
    
    algo_config = None
    is_symmetric = False # Asymmetric often captures distribution better
    
    if HAS_HQQ:
        # HQQ Configuration
        # HQQ typically implies certain symmetry/bit settings internally
        algo_config = HQQWeightOnlyQuantConfig(
            axis=1, # Channel-wise
            bits=4,
            block_size=32
        )
    
    quantizer = MatMulNBitsQuantizer(
        model=input_model_path,
        block_size=32,
        is_symmetric=is_symmetric,
        algo_config=algo_config,
        accuracy_level=4 # Highest precision for compute
    )
    
    quantizer.process()
    quantizer.model.save_model_to_file(output_model_path)
    
    print("HQ Quantization complete.")

if __name__ == "__main__":
    import os
    
    base_dir = r"models\phi3.5-hybrid"
    input_model = os.path.join(base_dir, "cpu_part.onnx")
    output_model = os.path.join(base_dir, "cpu_part.int4_hq.onnx")
    
    if not os.path.exists(input_model):
        print(f"Error: Input model not found at {input_model}")
        exit(1)
        
    quantize_cpu_hq(input_model, output_model)
