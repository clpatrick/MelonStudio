
from transformers import AutoConfig, AutoModelForCausalLM
import sys

try:
    print("Loading config...")
    config = AutoConfig.from_pretrained("models/gpt-oss-120b-20L", trust_remote_code=True)
    print(f"Config loaded. Layers: {config.num_hidden_layers}")
    
    print("Loading model (meta device)...")
    # Load on meta device first to check structure without RAM usage
    try:
        with sys.stdout:
             model = AutoModelForCausalLM.from_pretrained(
                "models/gpt-oss-120b-20L", 
                config=config, 
                trust_remote_code=True,
                device_map="meta" 
            )
        print("Model structure loaded on meta device successfully.")
    except Exception as e:
        print(f"Meta load failed: {e}")
        # Fallback to normal load if meta fails (some older transformers)
        print("Attempting normal CPU load (might be slow/OOM)...")
        # model = AutoModelForCausalLM.from_pretrained("models/gpt-oss-120b-20L", config=config, trust_remote_code=True)
        # print("Model loaded on CPU.")

except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
