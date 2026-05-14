import urllib.request 
from torch.utils.data import Dataset, DataLoader 
import torch 
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache, sync
from model import MHAModel
from inference_for_continuous_batching import MHAModelContinuousBatching
from continuous_batching import KVPool


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
                                            context_size=MODEL_CONFIG["max_seq_len"]
                                        )
epochs_tensor = torch.linspace(0, TRAINING_CONFIG["num_epochs"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
torch.save(model.state_dict(), "model_weights.pth")



kv_model = MHAModelContinuousBatching(MODEL_CONFIG)
kv_model.load_state_dict(torch.load("model_weights.pth"), strict=False)
kv_model.to(device)
kv_model.eval()

# Continuous batching setup
context_length = min(MODEL_CONFIG["rope_limit"], MODEL_CONFIG["kvcache_limit"])
num_layers = MODEL_CONFIG["num_layers"]
max_new_tokens = 50
pool = KVPool(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])
prompt1 = text_to_token_ids("Every effort moves you", tokenizer).to(device)
prompt2 = text_to_token_ids("Once upon a time there was", tokenizer).to(device)


# do som stuff with first prompt
pool.allocate_cache(context_length, num_layers, prompt1, max_new_tokens)
for step in range(5):
    inference_seq, partition, mask, active_requests = pool.next_seq()
    with torch.no_grad():
        logits = kv_model(inference_seq, partition, mask.to(device), active_requests)
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
pool.allocate_cache(context_length, num_layers, prompt2, max_new_tokens)
while any(r.status != "completed" for r in pool.active_requests):
    inference_seq, partition, mask, active_requests = pool.next_seq()
    with torch.no_grad():
        logits = kv_model(inference_seq, partition, mask.to(device), active_requests)
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
        

for i, request in enumerate(pool.active_requests):
    prompt_text = token_ids_to_text(request.prompt_token_id, tokenizer)
    generated_text = token_ids_to_text(request.generated_token_id, tokenizer)
    print(f"\nRequest {i}: {prompt_text}{generated_text}")


# # reset
# kv_model_test1 = MHAModelContinuousBatching(MODEL_CONFIG)
# kv_model_test1.load_state_dict(torch.load("model_weights.pth"), strict=False)
# kv_model_test1.to(device)
# kv_model_test1.eval()

# pool_test1 = KVPool(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])
# pool_test1.allocate_cache(context_length, num_layers, text_to_token_ids("Every effort moves you", tokenizer).to(device), max_new_tokens)

# while any(r.status != "completed" for r in pool_test1.active_requests):
#     inference_seq, partition, mask, active_requests = pool_test1.next_seq()
#     with torch.no_grad():
#         logits = kv_model_test1(inference_seq, partition, mask.to(device), active_requests)
#     for (start, end), req_id in partition.items():
#         request = active_requests[req_id]
#         next_logits = logits[:, end-1, :]
#         probas = torch.softmax(next_logits, dim=-1)
#         next_token = torch.argmax(probas, dim=-1, keepdim=True)
#         if request.generated_token_id is None:
#             request.generated_token_id = next_token
#         else:
#             request.generated_token_id = torch.cat([request.generated_token_id, next_token], dim=-1)
#         if request.prefill_tok_left == 0:
#             request.status = "decoding"
#         if request.status == "decoding":
#             request.tokens_generated += 1
#             if request.tokens_generated >= request.max_new_tokens:
#                 request.status = "completed"

# print(f"Standalone 1: {token_ids_to_text(pool_test1.active_requests[0].prompt_token_id, tokenizer)}{token_ids_to_text(pool_test1.active_requests[0].generated_token_id, tokenizer)}")


# kv_model_test2 = MHAModelContinuousBatching(MODEL_CONFIG)
# kv_model_test2.load_state_dict(torch.load("model_weights.pth"), strict=False)
# kv_model_test2.to(device)
# kv_model_test2.eval()

# pool_test2 = KVPool(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])
# pool_test2.allocate_cache(context_length, num_layers, text_to_token_ids("Once upon a time there was", tokenizer).to(device), max_new_tokens)

# while any(r.status != "completed" for r in pool_test2.active_requests):
#     inference_seq, partition, mask, active_requests = pool_test2.next_seq()
#     with torch.no_grad():
#         logits = kv_model_test2(inference_seq, partition, mask.to(device), active_requests)
#     for (start, end), req_id in partition.items():
#         request = active_requests[req_id]
#         next_logits = logits[:, end-1, :]
#         probas = torch.softmax(next_logits, dim=-1)
#         next_token = torch.argmax(probas, dim=-1, keepdim=True)
#         if request.generated_token_id is None:
#             request.generated_token_id = next_token
#         else:
#             request.generated_token_id = torch.cat([request.generated_token_id, next_token], dim=-1)
#         if request.prefill_tok_left == 0:
#             request.status = "decoding"
#         if request.status == "decoding":
#             request.tokens_generated += 1
#             if request.tokens_generated >= request.max_new_tokens:
#                 request.status = "completed"

# print(f"Standalone 2: {token_ids_to_text(pool_test2.active_requests[0].prompt_token_id, tokenizer)}{token_ids_to_text(pool_test2.active_requests[0].generated_token_id, tokenizer)}")



# # Inference without KV cache
# model.eval()
# with torch.no_grad():
#     sync()
#     start = time.time()
#     output1 = generate_text_simple(model, token_ids, max_new_tokens, MODEL_CONFIG["max_seq_len"])
#     sync()
#     end = time.time()
# t1 = end - start
# text1 = token_ids_to_text(output1, tokenizer)
# print(f"Without KV cache: {t1:.3f}s")

# # Inference with KV cache
# kv_model = MHAModelKV(MODEL_CONFIG)
# kv_model.load_state_dict(torch.load("model_weights.pth"))
# kv_model.to(device)
# kv_model.eval()
# with torch.no_grad():
#     sync()
#     start = time.time()
#     output2 = generate_with_chunked_prefill(kv_model, token_ids, max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"], max_new_tokens=max_new_tokens)
#     sync()
#     end = time.time()
# t2 = end - start
# text2 = token_ids_to_text(output2, tokenizer)
# size2 = kv_model.get_total_kv_cache_size()
# print(f"With KV cache: {t2:.3f}s | Cache: {size2/1024:.1f} KB")
# kv_model.clear_cache()

# # Inference with optimized KV cache
# optimized_kv_model = MHAModelOptimizedKV(MODEL_CONFIG)
# optimized_kv_model.load_state_dict(torch.load("model_weights.pth"))
# optimized_kv_model.to(device)
# for block in optimized_kv_model.blocks:
#     block.attention.kv_cache.K_cache = block.attention.kv_cache.K_cache.to(device)
#     block.attention.kv_cache.V_cache = block.attention.kv_cache.V_cache.to(device)
# optimized_kv_model.eval()
# with torch.no_grad():
#     sync()
#     start = time.time()
#     output3 = generate_with_chunked_prefill(kv_model, token_ids, max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"], max_new_tokens=max_new_tokens)
#     sync()
#     end = time.time()
# t3 = end - start
# text3 = token_ids_to_text(output3, tokenizer)
# size3 = optimized_kv_model.get_total_kv_cache_size()
# print(f"With optimized KV cache: {t3:.3f}s | Cache: {size3/1024:.1f} KB")
# optimized_kv_model.clear_cache()

# # print(f"\nSpeedup KV vs No KV: {t1/t2:.2f}x")
# # print(f"Speedup Optimized KV vs No KV: {t1/t3:.2f}x")

# with open("inference_results.txt", "w") as f:
#     f.write(f"Max new tokens: {max_new_tokens}\n")
#     f.write(f"Context length: {MODEL_CONFIG['max_seq_len']}\n")
#     f.write(f"Device: {device}\n\n")
    
#     f.write(f"Without KV cache: {t1:.3f}s\n")
#     f.write(f"Generated:\n{text1}\n\n")
    
#     f.write(f"With KV cache: {t2:.3f}s | Cache: {size2/1024:.1f} KB\n")
#     f.write(f"Generated:\n{text2}\n\n")
    
#     f.write(f"With optimized KV cache: {t3:.3f}s | Cache: {size3/1024:.1f} KB\n")
#     f.write(f"Generated:\n{text3}\n\n")
    
#     # f.write(f"Speedup KV vs No KV: {t1/t2:.2f}x\n")
#     # f.write(f"Speedup Optimized KV vs No KV: {t1/t3:.2f}x\n")

# print("Results saved to inference_results.txt")