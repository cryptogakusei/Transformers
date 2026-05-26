from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import torch
import tiktoken
from uuid import uuid4

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache, sync
from model import MHAModel
from mha_w_speculative_decoding.old_files.inference_for_continuous_batching import MHAModelContinuousBatching
from mha_w_speculative_decoding.old_files.continuous_batching import KVPool
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

tokenizer = tiktoken.get_encoding("gpt2")

model = MHAModelContinuousBatching(MODEL_CONFIG)
model.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
model.to(device)
model.eval()

context_length = min(MODEL_CONFIG["rope_limit"], MODEL_CONFIG["kvcache_limit"])
num_layers = MODEL_CONFIG["num_layers"]
pool = KVPool(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])



# MOST of the code in THIS FILE (ONLY) has been taken from someone else's codebase and I modified it a bit for my work. 
# I didn't come up the logic by myself.


app = FastAPI()

request_queue = asyncio.Queue()
results = {}

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = str(uuid4())
    results[request_id] = {"tokens": [], "timestamps": [], "done": False}
    await request_queue.put((request_id, req))
    return {"request_id": request_id}


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    return results[request_id]


@app.on_event("startup")
async def start_inference_loop():
    asyncio.create_task(inference_loop())


async def inference_loop():
    while True:
        while not request_queue.empty():
            request_id, req = await request_queue.get()
            token_ids = text_to_token_ids(req.prompt, tokenizer).to(device)
            pool.allocate_cache(context_length, num_layers, token_ids, req.max_new_tokens, request_id=request_id)

        if not pool.active_requests or all(r.status == "completed" for r in pool.active_requests):
            completed_indices = [i for i, r in enumerate(pool.active_requests) if r.status == "completed"]
            if completed_indices:
                pool.clear_completed(completed_indices)
            await asyncio.sleep(0.01)
            continue

        inference_seq, partition, mask, active_requests = pool.next_seq()

        with torch.no_grad():
            logits = model(inference_seq, partition, mask.to(device), active_requests)

        for (start, end), req_id in partition.items():
            request = active_requests[req_id]
            next_logits = logits[:, end - 1, :]
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
                results[request.request_id]["tokens"].append(next_token.item())
                results[request.request_id]["timestamps"].append(time.time())
                if request.tokens_generated >= request.max_new_tokens:
                    request.status = "completed"
                    results[request.request_id]["done"] = True
                    results[request.request_id]["text"] = token_ids_to_text(request.generated_token_id, tokenizer)

        await asyncio.sleep(0)  


