from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import torch
import tiktoken
from uuid import uuid4

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG
from utils import text_to_token_ids, token_ids_to_text
from schedular import Schedular
from page_allocator import PageAllocator
from draft_model import DraftModel
from target_model import TargetModel
from speculative_decoding import SpeculativeDecodingManager
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

tokenizer = tiktoken.get_encoding("gpt2")

# Load draft and target models
draft_model = DraftModel(MODEL_CONFIG)
draft_model.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
draft_model.to(device)
draft_model.eval()

target_model = TargetModel(MODEL_CONFIG)
target_model.load_state_dict(torch.load("model_weights.pth", map_location=device), strict=False)
target_model.to(device)
target_model.eval()

context_length = min(MODEL_CONFIG["rope_limit"], MODEL_CONFIG["kvcache_limit"])
num_layers = MODEL_CONFIG["num_layers"]

schedular = Schedular(max_num_batched_tokens=INFERENCE_CONFIG["max_num_batched_tokens"])

draft_page_allocator = PageAllocator(
    num_pages=INFERENCE_CONFIG["num_pages"],
    num_heads=MODEL_CONFIG["num_heads"],
    tokens_per_page=INFERENCE_CONFIG["tokens_per_page"],
    head_dim=MODEL_CONFIG["emb_dim"] // MODEL_CONFIG["num_heads"],
    device=device,
)

target_page_allocator = PageAllocator(
    num_pages=INFERENCE_CONFIG["num_pages"],
    num_heads=MODEL_CONFIG["num_heads"],
    tokens_per_page=INFERENCE_CONFIG["tokens_per_page"],
    head_dim=MODEL_CONFIG["emb_dim"] // MODEL_CONFIG["num_heads"],
    device=device,
)

speculative_decoding_manager = SpeculativeDecodingManager(
    draft_model=draft_model,
    draft_page_allocator=draft_page_allocator,
    target_model=target_model,
    target_page_allocator=target_page_allocator,
    num_speculations=INFERENCE_CONFIG["num_speculations"],
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

        # queue incoming requests
        while not request_queue.empty():
            request_id, req = await request_queue.get()
            token_ids = text_to_token_ids(req.prompt, tokenizer).to(device)
            schedular.queue(context_length, num_layers, token_ids, req.max_new_tokens, request_id=request_id)

        # if no active requests currently, keep waiting
        if not schedular.active_requests or all(
            request.status == "completed" for request in schedular.active_requests.values()
        ):
            completed_request_ids = [
                request_id
                for request_id, request in schedular.active_requests.items()
                if request.status == "completed"
            ]
            if completed_request_ids:
                schedular.clear_completed(completed_request_ids)
            await asyncio.sleep(0.01)
            continue

        # get next sequence to process
        inference_seq, partition, active_requests = schedular.next_seq()

        request_ids = []
        num_new_tokens_per_id = []
        for locs, request_id in partition.items():
            request_ids.append(request_id)
            (start_pos, end_pos) = locs
            num_new_tokens_per_id.append(end_pos - start_pos)

        # generate an artificial delay until enough space is available in both page allocators
        while (
            draft_page_allocator.check_space_availability(
                request_ids, num_new_tokens_per_id, MODEL_CONFIG["num_layers"], INFERENCE_CONFIG["num_speculations"]
            )
            is False
            or target_page_allocator.check_space_availability(
                request_ids, num_new_tokens_per_id, MODEL_CONFIG["num_layers"], INFERENCE_CONFIG["num_speculations"]
            )
            is False
        ):
            # reclaim the pages assigned to completed requests in both page allocators
            completed_request_ids = [
                request_id
                for request_id, request in schedular.active_requests.items()
                if request.status == "completed"
            ]
            for completed_request_id in completed_request_ids:
                draft_page_allocator.reclaim(completed_request_id, MODEL_CONFIG["num_layers"])
                target_page_allocator.reclaim(completed_request_id, MODEL_CONFIG["num_layers"])

            schedular.clear_completed(completed_request_ids)
            await asyncio.sleep(0.1)

        # track tokens before the run to detect newly generated ones
        tokens_before = {}
        for (_, _), request_id in partition.items():
            request = active_requests[request_id]
            tokens_before[request_id] = (
                request.generated_token_id.shape[-1] if request.generated_token_id is not None else 0
            )

        # run speculative decoding
        speculative_decoding_manager.run(inference_seq, partition, active_requests, num_layers)

        # update request status and stream results
        for (start, end), request_id in partition.items():
            request = active_requests[request_id]

            if request.prefill_tok_left == 0:
                request.status = "decoding"

            if request.status == "decoding":
                # figure out how many new tokens were added this round
                tokens_now = request.generated_token_id.shape[-1] if request.generated_token_id is not None else 0
                num_new = tokens_now - tokens_before[request_id]

                # stream the newly generated tokens to results
                if num_new > 0:
                    new_tokens = request.generated_token_id[:, -num_new:]
                    for t in range(num_new):
                        token_val = new_tokens[:, t].item()
                        results[request.request_id]["tokens"].append(token_val)
                        results[request.request_id]["timestamps"].append(time.time())

                if request.tokens_generated >= request.max_new_tokens:
                    request.status = "completed"
                    results[request.request_id]["done"] = True
                    results[request.request_id]["text"] = token_ids_to_text(
                        request.generated_token_id, tokenizer
                    )

        # reclaim the pages assigned to completed requests in both page allocators
        completed_request_ids = [
            request_id
            for request_id, request in schedular.active_requests.items()
            if request.status == "completed"
        ]
        for completed_request_id in completed_request_ids:
            draft_page_allocator.reclaim(completed_request_id, MODEL_CONFIG["num_layers"])
            target_page_allocator.reclaim(completed_request_id, MODEL_CONFIG["num_layers"])

        schedular.clear_completed(completed_request_ids)

        await asyncio.sleep(0.1)