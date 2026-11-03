"""H20 (sm90) numerics of vLLM 0.18 `--quantization fp8` on an A800 (sm80), for accuracy tests only.

On sm90, online FP8 (bf16 checkpoint) runs Fp8OnlineLinearMethod with:
- weights quantized once per tensor (`ops.scaled_fp8_quant`, E4M3, scale = amax / 448);
- activations quantized per token at run time (`QuantFP8`; per token because
  `cutlass_fp8_supported()` is true there);
- `CutlassFP8ScaledMMLinearKernel`: FP8 x FP8 products, FP32 accumulation,
  D = s_a * (s_b * acc) (+ bias), cast to bf16.
sm80 has no FP8 GEMM, so vLLM switches to Marlin weight-only FP8 with bf16 activations.

Here vLLM still makes those sm90 choices and quantizes the weights with its own code. The
activation quantization and the GEMM run in one custom op that torch.compile cannot see into:
- activations: the formula of `QuantFP8.forward_native`, which is what sm90 runs by default
  (`custom_ops` is ["none"] under Inductor, so Inductor compiles that formula into a Triton kernel;
  Triton on sm80 cannot emit E4M3, so here it runs eagerly in PyTorch);
- GEMM: every E4M3 value is exact in bf16 and a bf16 x bf16 product is exact in FP32, so a bf16
  GEMM over the FP8 values with an FP32 output equals the sm90 GEMM up to the accumulation order
  and the FP8 tensor-core accumulator width.
"""
import importlib.abc
import importlib.util
import os
import sys

TARGET = "vllm.model_executor.layers.quantization.fp8"
TAG = "[fp8-sm90-emu]"


class _AfterImport(importlib.abc.MetaPathFinder):
    """Patch the FP8 module as soon as vLLM imports it, in whichever process that happens."""

    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


def install_import_hook():
    sys.meta_path.insert(0, _AfterImport())


def _install(fp8):
    import torch
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.input_quant_fp8 import (
        _FP8_DTYPE, _FP8_MAX, _FP8_MIN, _FP8_MIN_SCALING_FACTOR)
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    @torch.library.custom_op("oal_fp8_sm90_emu::fp8_linear", mutates_args=())
    def fp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                   bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
        # QuantFP8.forward_native, dynamic per-token branch
        x_max = x.abs().max(dim=-1)[0].unsqueeze(-1).to(torch.float32)
        x_s = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        x_q = (x.to(torch.float32) * x_s.reciprocal()).clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
        # CUTLASS scaled_mm: FP32 accumulation, D = s_a * (s_b * acc) + bias
        acc = torch.mm(x_q.to(torch.bfloat16), weight, out_dtype=torch.float32)
        out = x_s * (weight_scale * acc)
        if bias is not None:
            out = out + bias
        return out.to(out_dtype)

    @fp8_linear.register_fake
    def _(x, weight, weight_scale, bias, out_dtype):
        return x.new_empty((x.shape[0], weight.shape[1]), dtype=out_dtype)

    # sm90 answer: per-token dynamic activation scales instead of one scale per tensor.
    fp8.cutlass_fp8_supported = lambda: True

    init = fp8.Fp8LinearMethod.__init__
    reported = []

    def __init__(self, quant_config):
        init(self, quant_config)
        self.use_marlin = False  # sm80 would fall back to weight-only Marlin
        quant = self.fp8_linear.quant_fp8
        if quant.static or quant.group_shape != GroupShape.PER_TOKEN:
            raise RuntimeError(f"{TAG} expected the sm90 dynamic per-token activation scales, got {quant.group_shape}")
        if not reported:
            reported.append(True)
            print(f"{TAG} activation scales: {self.fp8_linear.quant_fp8.group_shape}, "
                  f"kernel {type(self.fp8_linear).__name__}", file=sys.stderr, flush=True)

    fp8.Fp8LinearMethod.__init__ = __init__

    process = fp8.Fp8OnlineLinearMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        process(self, layer)
        if layer.weight.dtype != torch.bfloat16:
            # Keep the [K, N] FP8 weight values in bf16 (exact) for the emulated GEMM, in place of
            # the FP8 tensor, so the server needs no more memory than the bf16 model.
            fp8.replace_parameter(layer, "weight", layer.weight.to(torch.bfloat16))

    fp8.Fp8OnlineLinearMethod.process_weights_after_loading = process_weights_after_loading

    def apply_weights(self, layer, x, bias=None):
        if getattr(layer, "input_scale", None) is not None or getattr(layer, "input_scale_ub", None) is not None:
            raise RuntimeError(f"{TAG} static activation scales are not emulated")
        out = fp8_linear(x.view(-1, x.shape[-1]), layer.weight, layer.weight_scale, bias,
                         self.config.out_dtype or x.dtype)
        return out.view(*x.shape[:-1], out.shape[-1])

    CutlassFP8ScaledMMLinearKernel.apply_weights = apply_weights

    get = fp8.Fp8Config.get_quant_method

    def get_quant_method(self, layer, prefix):
        method = get(self, layer, prefix)
        if isinstance(method, fp8.Fp8LinearMethod):
            print(f"{TAG} quantize {prefix}", file=sys.stderr, flush=True)
        return method

    fp8.Fp8Config.get_quant_method = get_quant_method
    print(f"{TAG} installed in pid {os.getpid()}", file=sys.stderr, flush=True)
