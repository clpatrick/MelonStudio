
import torch
from transformers import AutoModelForCausalLM

try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("c:/repos/MelonStudio/cache_dir/microsoft_Phi-3-mini-4k-instruct", trust_remote_code=False, low_cpu_mem_usage=True)
    layer = model.model.layers[0]

    batch_size = 1
    seq_len = 4
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")
    
    dummy_hidden = torch.randn(batch_size, seq_len, hidden_dim).to(model.dtype)
    dummy_mask = torch.ones(batch_size, 1, seq_len, seq_len).to(model.dtype)
    
    dummy_pos = torch.arange(seq_len).unsqueeze(0)

    import inspect
    print(f"Layer forward signature: {inspect.signature(layer.forward)}")

    # Try calling
    print("Calling layer...")
    # Note: Phi-3 might use 'rotary_pos_emb' instead of position_ids in some impls, 
    # but with trust_remote_code=False it uses LlamaForCausalLM or similar generic backing if available.
    # Actually Phi-3 is supported in transformers 4.40+.
    
    output = layer(dummy_hidden, attention_mask=dummy_mask, position_ids=dummy_pos)
    print(f"Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output len: {len(output)}")
        print(f"Item 0 type: {type(output[0])}")
    else:
        print("Output is not tuple")

except Exception as e:
    import traceback
    traceback.print_exc()
