import urllib.request 
from torch.utils.data import Dataset, DataLoader 
import torch 
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

from config import MODEL_CONFIG, TRAINING_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache, sync
from inference_with_optimized_kv_cache import MHAModelOptimizedKV

torch.manual_seed(123) # initializing randomness

### device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")


### initializing the tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")


prompt = "Every effort moves you"
token_ids = text_to_token_ids(prompt, tokenizer).to(device) # 1 x seq_len
max_new_tokens = 300


optimized_kv_model = MHAModelOptimizedKV(MODEL_CONFIG)
optimized_kv_model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
optimized_kv_model.to(device)
for block in optimized_kv_model.blocks:
    block.attention.kv_cache.K_cache = block.attention.kv_cache.K_cache.to(device)
    block.attention.kv_cache.V_cache = block.attention.kv_cache.V_cache.to(device)
optimized_kv_model.eval()
with torch.no_grad():
    sync()
    start = time.time()
    output3 = generate_with_kv_cache(optimized_kv_model, token_ids, max_new_tokens, context_size=MODEL_CONFIG["context_length"])
    sync()
    end = time.time()
t3 = end - start
text3 = token_ids_to_text(output3, tokenizer)
size3 = optimized_kv_model.get_total_kv_cache_size()
print(f"With optimized KV cache: {t3:.3f}s | Cache: {size3/1024:.1f} KB")
optimized_kv_model.clear_cache()

with open("inference_investigation.txt", "w") as f:
    f.write(f"Max new tokens: {max_new_tokens}\n")
    f.write(f"Context length: {MODEL_CONFIG['context_length']}\n")
    f.write(f"Device: {device}\n\n")

    
    f.write(f"With optimized KV cache: {t3:.3f}s | Cache: {size3/1024:.1f} KB\n")
    f.write(f"Generated:\n{text3}\n\n")

print("Results saved to inference_results.txt")