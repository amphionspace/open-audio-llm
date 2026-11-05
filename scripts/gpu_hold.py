"""Fill every visible GPU for AmphionASR- holder jobs on ACP (scripts/acp_holder.py): full utilization and most of the memory.

One process per GPU runs back-to-back bf16 matmuls (the cluster is all A800 80G, 312 TFLOPS bf16) and keeps a
resident tensor of --memory-fraction of the free memory; a worker that fails is restarted. Stop it by stopping
the ACP job.
usage: gpu_hold.py [--seconds N] [--memory-fraction F]   (N = 0 runs until stopped)
"""
import argparse
import multiprocessing as mp
import time

import torch

SIZE = 16384  # 512 MiB per bf16 matrix; large enough to saturate the tensor cores
CHECK_EVERY = 64  # matmuls queued between end-time checks


def burn(index, seconds, memory_fraction):
    torch.cuda.set_device(index)
    free, _ = torch.cuda.mem_get_info()
    ballast = torch.empty(int(free * memory_fraction), device="cuda", dtype=torch.uint8)  # noqa: F841
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds if seconds else float("inf")
    while time.time() < end:
        for _ in range(CHECK_EVERY):
            a @ b
        torch.cuda.synchronize()


def supervised(index, seconds, memory_fraction):
    deadline = time.time() + seconds if seconds else None
    while deadline is None or time.time() < deadline:
        try:
            burn(index, max(1, int(deadline - time.time())) if deadline else 0, memory_fraction)
            return
        except Exception as exc:  # noqa: BLE001 - keep the card held through transient CUDA errors
            print(f"gpu {index}: {type(exc).__name__}: {exc}; restarting", flush=True)
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    mp.set_start_method("spawn")
    workers = [mp.Process(target=supervised, args=(i, args.seconds, args.memory_fraction)) for i in range(torch.cuda.device_count())]
    for worker in workers:
        worker.start()
    print(f"holding {len(workers)} GPUs", flush=True)
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
