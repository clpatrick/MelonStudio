import onnx
from onnx import TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

def convert_fp16_to_fp32(model_path, output_path):
    print(f"Converting {model_path} to FP32...")
    model = onnx.load(model_path)
    
    # Iterate through all tensors in the graph and convert Float16 to Float
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            # Convert internal data
            float_data = [float(x) for x in onnx.numpy_helper.to_array(tensor).flatten()]
            tensor.ClearField("raw_data")
            tensor.ClearField("float_data")
            tensor.ClearField("int32_data")
            tensor.float_data.extend(float_data)
            tensor.data_type = TensorProto.FLOAT
            
    # Update inputs/outputs
    for input_node in model.graph.input:
        if input_node.type.tensor_type.elem_type == TensorProto.FLOAT16:
            input_node.type.tensor_type.elem_type = TensorProto.FLOAT
            
    for output_node in model.graph.output:
        if output_node.type.tensor_type.elem_type == TensorProto.FLOAT16:
            output_node.type.tensor_type.elem_type = TensorProto.FLOAT
            
    # Update value info
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    # Update nodes (cast operators might need handling, but for simple weight conversion this usually suffices
    # or we rely on onnx.convert_float16_to_float if available.
    # Actually, simplistic conversion might break if Cast nodes exist.
    # Let's try the safer onnx library converter if available.
    
    # Save with external data to support models > 2GB
    external_data_file = os.path.basename(output_path) + ".data"
    onnx.save(
        model, 
        output_path, 
        save_as_external_data=True, 
        all_tensors_to_one_file=True, 
        location=external_data_file, 
        size_threshold=1024, 
        convert_attribute=False
    )
    print(f"FP32 conversion complete (External Data: {external_data_file}).")

def quantize_int8(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path} to {output_model_path} (INT8)...")
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8
    )
    print("INT8 Quantization complete.")

if __name__ == "__main__":
    import os
    from onnx import version_converter
    
    base_dir = r"models\phi3.5-hybrid"
    input_model = os.path.join(base_dir, "cpu_part.onnx")
    fp32_model = os.path.join(base_dir, "cpu_part.fp32.onnx")
    int8_model = os.path.join(base_dir, "cpu_part.int8.onnx")
    
    if not os.path.exists(input_model):
        print("Input model not found.")
        exit(1)
        
    try:
        # Simplest way: Load, and rely on ORT tools or just manual conversion.
        # Check if we have the simplify tool or similar. 
        # Actually proper way: use onnxmltools or simple graph traversal.
        # Let's try the graph traversal defined above.
        
        convert_fp16_to_fp32(input_model, fp32_model)
        quantize_int8(fp32_model, int8_model)
        
        # Cleanup intermediate file
        if os.path.exists(fp32_model):
            os.remove(fp32_model)
            
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
