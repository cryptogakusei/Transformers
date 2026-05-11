import torch
import torch.nn as nn



def generate_with_chunked_prefill(model, idx, max_num_batched_tokens, max_new_tokens):
    model.eval()
    model.clear_cache()

    seq_len = idx.shape[1]
    for start in range(0, seq_len, max_num_batched_tokens):
        end = min(start + max_num_batched_tokens, seq_len)
        print(f"Prefill chunk: tokens [{start}:{end}]")
        logits = model(idx[:, start:end])
    
    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1) # only for the the first new token generated
    idx_next = torch.argmax(probas,dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=-1)

    for _ in range(max_new_tokens-1):
        logits = model(idx_next)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas,dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


    