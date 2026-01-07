import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def fix_and_quantize():
    base_dir = r"models\phi3.5-hybrid"
    fp32_model_path = os.path.join(base_dir, "cpu_part.fp32.onnx")
    int8_model_path = os.path.join(base_dir, "cpu_part.int8.onnx")
    
    if not os.path.exists(fp32_model_path):
        print("FP32 model not found to fix.")
        exit(1)
        
    print(f"Loading {fp32_model_path}...")
    model = onnx.load(fp32_model_path)
    
    # Check/Fix Opset
    ai_onnx_opset = None
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            ai_onnx_opset = opset
            break
            
    if ai_onnx_opset is None:
        print("Adding missing ai.onnx opset...")
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 14 # Standard modern opset
    else:
        print(f"Found existing opset: version {ai_onnx_opset.version}")
        # Force update if too old, but 14 is generally fine.
        if ai_onnx_opset.version < 11:
            ai_onnx_opset.version = 14
            print("Upgraded opset to 14.")

    # Convert model to version 14 explicitly to ensure compatibility
    # (Just saving it with the tag might be enough if the nodes are compatible)
    print("Saving repaired FP32 model...")
    onnx.save(model, fp32_model_path)
    
    print(f"Quantizing to {int8_model_path} (INT8)...")
    quantize_dynamic(
        fp32_model_path,
        int8_model_path,
        weight_type=QuantType.QUInt8
    )
    print("INT8 Quantization complete.")

if __name__ == "__main__":
    try:
        fix_and_quantize()
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
