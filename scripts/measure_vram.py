
import argparse
import time
import onnxruntime as ort
import sys
# Try to import pynvml, or use fallback
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def measure_vram(model_path):
    print(f"Measuring VRAM for {model_path}...")
    
    if HAS_NVML:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assume GPU 0
        info_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Initial VRAM Used: {info_start.used / 1024**2:.2f} MB")
    else:
        print("Warning: pynvml not found. Precise VRAM measurement unavailable.")
        handle = None

    # Load Session
    providers = ['CUDAExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Failed to load session on CUDA: {e}")
        return

    # Warmup (optional, triggers allocation)
    # session.get_inputs()...
    
    if HAS_NVML:
        info_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"VRAM Used after Load: {info_end.used / 1024**2:.2f} MB")
        delta = info_end.used - info_start.used
        print(f"Delta (Model Size in VRAM): {delta / 1024**2:.2f} MB")
        
        # Keep alive for a moment?
        # time.sleep(2)
        
        pynvml.nvmlShutdown()
        
        return delta
    else:
        print("Model loaded successfully (VRAM check skipped).")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to ONNX model")
    args = parser.parse_args()
    
    try:
        measure_vram(args.model)
    except Exception as e:
        print(f"Error: {e}")
