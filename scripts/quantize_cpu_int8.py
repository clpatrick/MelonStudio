import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_cpu_model_int8(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path} to {output_model_path} (INT8 Dynamic)...")
    
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print("Quantization complete.")

if __name__ == "__main__":
    import os
    
    # Default paths
    base_dir = r"models\phi3.5-hybrid"
    input_model = os.path.join(base_dir, "cpu_part.onnx")
    output_model = os.path.join(base_dir, "cpu_part.int8.onnx")
    
    if not os.path.exists(input_model):
        print(f"Error: Input model not found at {input_model}")
        exit(1)
        
    quantize_cpu_model_int8(input_model, output_model)
