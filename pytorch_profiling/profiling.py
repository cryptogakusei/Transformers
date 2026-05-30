import argparse
import os
import torch

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--trace_dir", default="./traces/01_matmul_add")
    return p.parse_args()


def main():
    args = parse_arguments()

    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    x = torch.randn(args.size, args.size, device=device, dtype=dtype)
    w = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    wait, warmup, active, repeat = 0, 1, 20, 1


    def fn(x, w, b):
        return torch.add(torch.matmul(x, w), b)
    
    fn = torch.compile(fn) if args.compile else fn

    # 
    def step():
        with torch.profiler.record_function("matmul_add"):
            return fn(x, w, b)
        
    # for some wramup
    if args.warmup:
        for _ in range(3):
            step()
        torch.cuda.synchronize()


    os.makedirs(args.trace_dir, exist_ok=True)
    compile_tag = "compile" if args.compile else "eager"
    warmup_tag = "warm" if args.warmup else "cold"
    tag = f"{args.size}_{args.dtype}_{warmup_tag}_{compile_tag}"

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    # describing the scheduler for running the profiler
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(wait + warmup + active):
            step()
            prof.step()

    torch.cuda.synchronize()


    print(f"saving traces ... {trace_path}")
    prof.export_chrome_trace(trace_path)

    with open(table_path, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()