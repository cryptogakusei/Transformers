from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import torch
import tiktoken
from uuid import uuid4

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG
from utils import create_dataloader, train, plot_losses, text_to_token_ids, token_ids_to_text, generate_text_simple, generate_with_kv_cache, sync
from model import MHAModel
from inference_w_paged_attention import MHAModelPagedAttention
from schedular import Schedular
from page_allocator import PageAllocator
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

tokenizer = tiktoken.get_encoding("gpt2")

model = MHAModelPagedAttention(MODEL_CONFIG)
model.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
model.to(device)
model.eval()

context_length = min(MODEL_CONFIG["rope_limit"], MODEL_CONFIG["kvcache_limit"])
num_layers = MODEL_CONFIG["num_layers"]

schedular = Schedular(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])
page_allocator = PageAllocator(
    num_pages=INFERENCE_CONFIG["num_pages"], 
    num_heads=MODEL_CONFIG["num_heads"], 
    tokens_per_page=INFERENCE_CONFIG["tokens_per_page"], 
    head_dim=MODEL_CONFIG["emb_dim"]//MODEL_CONFIG["num_heads"],
    device=device
    )

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
            schedular.queue(context_length, num_layers, token_ids, req.max_new_tokens, request_id=request_id)

        if not schedular.active_requests or all(request.status == "completed" for request in schedular.active_requests):
            completed_indices = [i for i, request in enumerate(schedular.active_requests) if request.status == "completed"]
            if completed_indices:
                schedular.clear_completed(completed_indices)
            await asyncio.sleep(0.01)
            continue

        inference_seq, partition, mask, active_requests = schedular.next_seq(page_allocator)

        request_ids = []
        num_new_tokens_per_id = []
        for key, value in partition.items():
            request_ids.append(active_requests[value].request_id)
            (start_pos, end_pos) = key
            num_new_tokens_per_id.append(end_pos - start_pos)

        # generate an artifical delay until enough space is available
        while page_allocator.check_space_availability(request_ids, num_new_tokens_per_id, MODEL_CONFIG["num_layers"]) is False:
            completed_request_ids = [active_requests[i].request_id for i, request in enumerate(schedular.active_requests) if request.status == "completed"]            
            for completed_request_id in completed_request_ids:
                page_allocator.reclaim(completed_request_id) 
            await asyncio.sleep(0.1) # if not space available we wait until enought space is obtained

        with torch.no_grad():
            logits = model(inference_seq, partition, mask.to(device), active_requests, page_allocator)

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


