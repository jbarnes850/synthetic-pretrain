#!/usr/bin/env python3
import sys
import time

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Spark runtime")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the active CUDA device")

    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    print(f"device_name={torch.cuda.get_device_name(dev)}")
    print(f"sm_capability={cap[0]}.{cap[1]}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    a = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"bf16_matmul_100x_512: {dt * 1000:.1f}ms, output_dtype={c.dtype}")
    q = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if out.dtype != torch.bfloat16:
        raise RuntimeError(f"Unexpected SDPA dtype: {out.dtype}")
    print(f"sdpa_shape={tuple(out.shape)} dtype={out.dtype}")
    print("SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
