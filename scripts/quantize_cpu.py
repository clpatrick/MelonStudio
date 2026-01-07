import onnx
import onnxruntime
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

def quantize_cpu_model(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path} to {output_model_path}...")
    
    # MatMulNBitsQuantizer defaults to 4-bits if not specified or via config
    quantizer = MatMulNBitsQuantizer(
        model=input_model_path, 
        block_size=32,
        is_symmetric=True,
        accuracy_level=None,
        algo_config=None
    )
    
    quantizer.process()
    quantizer.model.save_model_to_file(output_model_path)
    
    print("Quantization complete.")

if __name__ == "__main__":
    import os
    
    # Default paths
    base_dir = r"models\phi3.5-hybrid"
    input_model = os.path.join(base_dir, "cpu_part.onnx")
    output_model = os.path.join(base_dir, "cpu_part.int4.onnx")
    
    if not os.path.exists(input_model):
        print(f"Error: Input model not found at {input_model}")
        exit(1)
        
    quantize_cpu_model(input_model, output_model)
