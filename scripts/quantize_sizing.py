
import argparse
import os
import sys

# Robust import for MatMulNBitsQuantizer
try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
except ImportError:
    print("Error: MatMulNBitsQuantizer not found in onnxruntime.quantization.matmul_nbits_quantizer")
    sys.exit(1)

def quantize_for_sizing(input_model_path, output_model_path):
    print(f"Quantizing {input_model_path} to {output_model_path} (INT4 RTN)...")
    
    # Use MatMulNBitsQuantizer (New API)
    quantizer = MatMulNBitsQuantizer(
        model_or_path=input_model_path,
        n_bits=4,
        block_size=128,
        is_symmetric=True,
        # accuracy_level=None
    )
    
    # Process and save
    quantizer.process(
        use_gpu=False,
    )
    quantizer.model.save_model_to_file(output_model_path)
    print("Quantization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to FP32/BF16 ONNX model")
    parser.add_argument("output", help="Path to output INT4 ONNX model")
    args = parser.parse_args()
    
    quantize_for_sizing(args.input, args.output)
