import onnx

model_path = "models/phi3.5-hybrid/gpu_part.onnx"
model = onnx.load(model_path)

print(f"Inspecting {model_path}")
print("Inputs:")
for inp in model.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    print(f"  {inp.name}: [{', '.join(dims)}]")
