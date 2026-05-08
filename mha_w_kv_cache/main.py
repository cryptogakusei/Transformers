import urllib.request 
from torch.utils.data import Dataset, DataLoader 
import torch 
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

from config import MODEL_CONFIG, TRAINING_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache
from model import MHAModel
from inference import MHAModelKV

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
                                            tokenizer=tokenizer
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
    start = time.time()
    output1 = generate_text_simple(model, token_ids, max_new_tokens, MODEL_CONFIG["context_length"])
    end = time.time()
print(f"Without KV cache: {end - start:.3f}s")

# Inference with KV cache
kv_model = MHAModelKV(MODEL_CONFIG)
kv_model.load_state_dict(torch.load("model_weights.pth"))
kv_model.to(device)
kv_model.eval()
with torch.no_grad():
    start = time.time()
    output2 = generate_with_kv_cache(kv_model, token_ids, max_new_tokens=50, context_size=MODEL_CONFIG["context_length"])
    end = time.time()
print(f"With KV cache: {end - start:.3f}s")

print(f"Outputs match: {torch.equal(output1, output2)}")
print(token_ids_to_text(output2, tokenizer))

size = kv_model.get_total_kv_cache_size()
print(f"KV cache size: {size / 1024:.1f} KB ({size / (1024*1024):.2f} MB)")
kv_model.clear_cache()
