"""Keep every visible GPU busy and holding memory, for ASR- holder jobs on ACP (scripts/acp_holder.py).

One process per GPU runs bf16 matmuls for BUSY seconds out of every BUSY + REST, so utilization stays high
without running flat out; a worker that fails is restarted. Stop it by stopping the ACP job.
usage: gpu_hold.py [--seconds N]   (N = 0 runs until stopped)
"""
import argparse
import multiprocessing as mp
import time

import torch

SIZE = 8192  # 128 MiB per bf16 matrix
BALLAST_GIB = 8  # resident memory so the card shows as occupied
BUSY, REST = 0.8, 0.2


def burn(index, seconds):
    torch.cuda.set_device(index)
    a = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(SIZE, SIZE, device="cuda", dtype=torch.bfloat16)
    ballast = torch.empty(BALLAST_GIB * 2**30, device="cuda", dtype=torch.uint8)  # noqa: F841
    end = time.time() + seconds if seconds else float("inf")
    while time.time() < end:
        start = time.time()
        while time.time() - start < BUSY:
            a @ b
        torch.cuda.synchronize()
        time.sleep(REST)


def supervised(index, seconds):
    deadline = time.time() + seconds if seconds else None
    while deadline is None or time.time() < deadline:
        try:
            burn(index, max(1, int(deadline - time.time())) if deadline else 0)
            return
        except Exception as exc:  # noqa: BLE001 - keep the card held through transient CUDA errors
            print(f"gpu {index}: {type(exc).__name__}: {exc}; restarting", flush=True)
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=0)
    args = parser.parse_args()
    mp.set_start_method("spawn")
    workers = [mp.Process(target=supervised, args=(i, args.seconds)) for i in range(torch.cuda.device_count())]
    for worker in workers:
        worker.start()
    print(f"holding {len(workers)} GPUs", flush=True)
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
