import aiohttp
import asyncio
import time

import random
import urllib.request 

from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG

# MOST of the code in THIS FILE (ONLY) has been taken from someone else's codebase and I modified it a bit for my work. 
# I didn't come up the logic by myself.

file_path = "file.txt"
urllib.request.urlretrieve(TRAINING_CONFIG["url"], file_path)
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = f.read()

benchmark_requests = []
for i in range(100):
    prompt = random.choice([
        "Every effort moves you",
        "Once upon a time there was a little girl",
        raw_data[:500],   
        raw_data[:1000],
        "The",
        "I had always thought",
        raw_data[:200],
    ])
    delay = random.uniform(0, 10) 
    max_new_tokens = random.randint(20, 200)
    benchmark_requests.append({
        "prompt": prompt,
        "delay": delay,
        "max_new_tokens": max_new_tokens,
    })

async def send_request(session, prompt, delay, max_new_tokens):
    await asyncio.sleep(delay)
    t_submit = time.time()
    async with session.post("http://localhost:8000/generate",
                            json={"prompt": prompt, "max_new_tokens": max_new_tokens}) as resp:
        data = await resp.json()
    
    request_id = data["request_id"]
    while True:
        async with session.get(f"http://localhost:8000/result/{request_id}") as resp:
            result = await resp.json()
            if result["done"]:
                return {
                    "ttft": result["timestamps"][0] - t_submit,
                    "tpot": sum(t2-t1 for t1,t2 in zip(result["timestamps"], result["timestamps"][1:])) / max(len(result["timestamps"])-1, 1),
                    "total": result["timestamps"][-1] - t_submit,
                }
        await asyncio.sleep(0.01)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, req["prompt"], req["delay"], req["max_new_tokens"]) 
            for req in benchmark_requests
        ]
        results = await asyncio.gather(*tasks)
    
    ttfts = [r["ttft"] for r in results]
    tpots = [r["tpot"] for r in results]
    print(f"Avg TTFT: {sum(ttfts)/len(ttfts):.3f}s")
    print(f"Avg TPOT: {sum(tpots)/len(tpots):.3f}s")
    print(f"P99 TTFT: {sorted(ttfts)[int(len(ttfts)*0.99)]:.3f}s")

    total_tokens = sum(r["total"] / r["tpot"] for r in results if r["tpot"] > 0)
    wall_time = max(r["total"] + benchmark_requests[i]["delay"] for i, r in enumerate(results))
    print(f"Throughput: {total_tokens / wall_time:.1f} tokens/sec")

asyncio.run(main())