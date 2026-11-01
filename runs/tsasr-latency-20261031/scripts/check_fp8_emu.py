"""How close the emulated sm90 FP8 GEMM is to the exact result, next to the FP8 rounding error itself.

One Qwen3-1.7B-sized projection (K=2048 -> N=6144) on random activations with a few outlier
channels, quantized with vLLM's own ops (weights per tensor, activations per token):
- exact: float64 product of the dequantized FP8 operands, rounded once to bf16 (what an ideal FP8
  GEMM with a bf16 output returns; the sm90 kernel differs from it only by accumulation order);
- emulated: the bf16 GEMM with FP32 output used by fp8_sm90_emu;
- fp8 error: exact vs the bf16 model (x @ W in float64), i.e. what FP8 itself changes.
"""
import json
import sys

import torch
from vllm import _custom_ops as ops


def main():
    torch.manual_seed(0)
    m, k, n = 512, 2048, 6144
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda")
    x[:, torch.randperm(k)[:8]] *= 30  # LLM activations have a few large channels
    x = x.to(torch.bfloat16)

    w_q, w_s = ops.scaled_fp8_quant(w, scale=None)
    x_q, x_s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    acc = torch.mm(x_q.to(torch.bfloat16), w_q.t().to(torch.bfloat16), out_dtype=torch.float32)
    emulated = (x_s * (w_s * acc)).to(torch.bfloat16).double()

    dequant = (x_q.double() * x_s.double()) @ (w_q.double() * w_s.double()).t()
    exact = dequant.to(torch.bfloat16).double()
    model = x.double() @ w.double().t()

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    result = {
        "shape": [m, k, n],
        "emulated_vs_exact_rel_error": rel(emulated, exact),
        "emulated_elements_differing_from_exact": (emulated != exact).double().mean().item(),
        "fp8_vs_bf16_rel_error": rel(dequant, model),
    }
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
