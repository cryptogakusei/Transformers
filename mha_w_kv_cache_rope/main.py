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
from model import MHAModel
from inference import MHAModelKV
from inference_w_opt_kv_cache import MHAModelOptimizedKV

torch.manual_seed(123) # initializing randomness

### device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")


### initializing the tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")


### Preprocessing the input data 
# file_path = "file.txt"
# urllib.request.urlretrieve(TRAINING_CONFIG["url"], file_path)
# with open(file_path, "r", encoding="utf-8") as f:
#     raw_data = f.read()
ds = load_dataset(TRAINING_CONFIG["dataset"], split="train[:500000]")
raw_data = "\n".join(ds["text"])

# print #
total_tokens = len(tokenizer.encode(raw_data))
print(f"Tokens: {total_tokens}")
# print #

split_idx = int(TRAINING_CONFIG["train_ratio"] * len(raw_data))
train_data = raw_data[:split_idx]
val_data = raw_data[split_idx:]


train_loader = create_dataloader(
    tokenizer,
    train_data,
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length=MODEL_CONFIG["context_length"],
    stride=MODEL_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    tokenizer,
    val_data,
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length=MODEL_CONFIG["context_length"],
    stride=MODEL_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# print #
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
# print #


### Training
model = MHAModel(MODEL_CONFIG)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_losses, val_losses, tokens_seen = train(
                                            model, train_loader, val_loader, optimizer, device,
                                            num_epochs=TRAINING_CONFIG["num_epochs"], 
                                            eval_freq=TRAINING_CONFIG["eval_freq"], 
                                            eval_num_batches=TRAINING_CONFIG["eval_num_batches"],
                                            start_context="Every effort moves you", 
                                            tokenizer=tokenizer,
                                            context_size=MODEL_CONFIG["context_length"]
                                        )
epochs_tensor = torch.linspace(0, TRAINING_CONFIG["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
torch.save(model.state_dict(), "model_weights.pth")


prompt = "Every effort moves you"
token_ids = text_to_token_ids(prompt, tokenizer).to(device)
max_new_tokens = 500

# Inference without KV cache
model.eval()
with torch.no_grad():
    sync()
    start = time.time()
    output1 = generate_text_simple(model, token_ids, max_new_tokens, MODEL_CONFIG["context_length"])
    sync()
    end = time.time()
t1 = end - start
text1 = token_ids_to_text(output1, tokenizer)
print(f"Without KV cache: {t1:.3f}s")

# Inference with KV cache
kv_model = MHAModelKV(MODEL_CONFIG)
kv_model.load_state_dict(torch.load("model_weights.pth"))
kv_model.to(device)
kv_model.eval()
with torch.no_grad():
    sync()
    start = time.time()
    output2 = generate_with_kv_cache(kv_model, token_ids, max_new_tokens, context_size=MODEL_CONFIG["context_length"])
    sync()
    end = time.time()
t2 = end - start
text2 = token_ids_to_text(output2, tokenizer)
size2 = kv_model.get_total_kv_cache_size()
print(f"With KV cache: {t2:.3f}s | Cache: {size2/1024:.1f} KB")
kv_model.clear_cache()

# Inference with optimized KV cache
optimized_kv_model = MHAModelOptimizedKV(MODEL_CONFIG)
optimized_kv_model.load_state_dict(torch.load("model_weights.pth"))
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

print(f"\nSpeedup KV vs No KV: {t1/t2:.2f}x")
print(f"Speedup Optimized KV vs No KV: {t1/t3:.2f}x")

with open("inference_results.txt", "w") as f:
    f.write(f"Max new tokens: {max_new_tokens}\n")
    f.write(f"Context length: {MODEL_CONFIG['context_length']}\n")
    f.write(f"Device: {device}\n\n")
    
    f.write(f"Without KV cache: {t1:.3f}s\n")
    f.write(f"Generated:\n{text1}\n\n")
    
    f.write(f"With KV cache: {t2:.3f}s | Cache: {size2/1024:.1f} KB\n")
    f.write(f"Generated:\n{text2}\n\n")
    
    f.write(f"With optimized KV cache: {t3:.3f}s | Cache: {size3/1024:.1f} KB\n")
    f.write(f"Generated:\n{text3}\n\n")
    
    f.write(f"Speedup KV vs No KV: {t1/t2:.2f}x\n")
    f.write(f"Speedup Optimized KV vs No KV: {t1/t3:.2f}x\n")

print("Results saved to inference_results.txt")