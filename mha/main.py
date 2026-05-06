import urllib.request 
from torch.utils.data import Dataset, DataLoader 
import torch 
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt

from config import MODEL_CONFIG, TRAINING_CONFIG
from utils import create_dataloader, train
from model import MHAModel

torch.manual_seed(123) # initializing randomness

### device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")


### initializing the tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")


### Preprocessing the input data 
file_path = "file.txt"
urllib.request.urlretrieve(TRAINING_CONFIG["url"], file_path)
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = f.read()

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


