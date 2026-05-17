import urllib.request 
from torch.utils.data import Dataset, DataLoader 
import torch 
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
from datasets import load_dataset
from uuid import uuid4 

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache, sync
from model import MHAModel
from inference_for_continuous_batching import MHAModelContinuousBatching
from continuous_batching import KVPool
from inference_w_paged_attention import MHAModelPagedAttention
from schedular import Schedular
from page_allocator import PageAllocator


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
# ds = load_dataset(TRAINING_CONFIG["dataset"], split="train[:500000]")
# raw_data = "\n".join(ds["text"])

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
    max_length=MODEL_CONFIG["max_seq_len"],
    stride=MODEL_CONFIG["max_seq_len"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    tokenizer,
    val_data,
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length=MODEL_CONFIG["max_seq_len"],
    stride=MODEL_CONFIG["max_seq_len"],
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


#====== Training ========
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
                                            context_size=MODEL_CONFIG["max_seq_len"]
                                        )
epochs_tensor = torch.linspace(0, TRAINING_CONFIG["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
torch.save(model.state_dict(), "model_weights.pth")





#======= INFERENCE ========


model = MHAModelPagedAttention(MODEL_CONFIG)
model.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
model.to(device)
model.eval()

# Continuous batching setup
context_length = min(MODEL_CONFIG["rope_limit"], MODEL_CONFIG["kvcache_limit"])
num_layers = MODEL_CONFIG["num_layers"]
max_new_tokens = 50
schedular = Schedular(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])
page_allocator = PageAllocator(
    num_pages=INFERENCE_CONFIG["num_pages"], 
    num_heads=MODEL_CONFIG["num_heads"], 
    tokens_per_page=INFERENCE_CONFIG["tokens_per_page"], 
    head_dim=MODEL_CONFIG["emb_dim"]//MODEL_CONFIG["num_heads"],
    device=device
    )
prompt1 = text_to_token_ids("Every effort moves you", tokenizer).to(device)
prompt2 = text_to_token_ids("Once upon a time there was", tokenizer).to(device)


# do som stuff with first prompt
schedular.queue(context_length, num_layers, prompt1, max_new_tokens, request_id=str(uuid4()))
for step in range(5):
    inference_seq, partition, mask, active_requests = schedular.next_seq(page_allocator)

    request_ids = []
    num_new_tokens_per_id = []
    for key, value in partition.items():
        request_ids.append(active_requests[value].request_id)
        (start_pos, end_pos) = key
        num_new_tokens_per_id.append(end_pos - start_pos)
    print(f'Is space available?: {page_allocator.check_space_availability(request_ids, num_new_tokens_per_id, MODEL_CONFIG["num_layers"])}')

    with torch.no_grad():
        logits = model(inference_seq, partition, mask.to(device), active_requests, page_allocator)

    for (start, end), req_id in partition.items():
        request = active_requests[req_id]
        next_logits = logits[:, end-1, :]
        probas = torch.softmax(next_logits, dim=-1)
        next_token = torch.argmax(probas, dim=-1, keepdim=True)
        if request.generated_token_id is None:
            request.generated_token_id = next_token
        else:
            request.generated_token_id = torch.cat([request.generated_token_id, next_token], dim=-1)
        
        if request.prefill_tok_left == 0:
            request.status = "decoding"

        if request.status == "decoding":
            request.tokens_generated += 1
            if request.tokens_generated >= request.max_new_tokens:
                request.status = "completed"

# now add second request mid-flight
schedular.queue(context_length, num_layers, prompt2, max_new_tokens, request_id=str(uuid4()))
while any(request.status != "completed" for request in schedular.active_requests):
    inference_seq, partition, mask, active_requests = schedular.next_seq(page_allocator)

    request_ids = []
    num_new_tokens_per_id = []
    for key, value in partition.items():
        request_ids.append(active_requests[value].request_id)
        (start_pos, end_pos) = key
        num_new_tokens_per_id.append(end_pos - start_pos)
    print(f"Is space available?: {page_allocator.check_space_availability(request_ids, num_new_tokens_per_id, MODEL_CONFIG['num_layers'])}")


    with torch.no_grad():
        logits = model(inference_seq, partition, mask.to(device), active_requests, page_allocator)
    for (start, end), req_id in partition.items():
        request = active_requests[req_id]
        next_logits = logits[:, end-1, :]
        probas = torch.softmax(next_logits, dim=-1)
        next_token = torch.argmax(probas, dim=-1, keepdim=True)
        if request.generated_token_id is None:
            request.generated_token_id = next_token
        else:
            request.generated_token_id = torch.cat([request.generated_token_id, next_token], dim=-1)

        if request.prefill_tok_left == 0:
            request.status = "decoding"
        
        if request.status == "decoding":
            request.tokens_generated += 1
            if request.tokens_generated >= request.max_new_tokens:
                request.status = "completed"
        

for i, request in enumerate(schedular.active_requests):
    prompt_text = token_ids_to_text(request.prompt_token_id, tokenizer)
    generated_text = token_ids_to_text(request.generated_token_id, tokenizer)
    print(f"\nRequest {i}: {prompt_text}{generated_text}")

