"""公共工具：正确性对照、计时、FLOPs / 带宽估算。

每个 ch*.py 都可以独立运行：  python code/ch01_tensors_einsum.py
"""
import time
import torch

torch.manual_seed(0)


def check(name, a, b, atol=2e-2, rtol=2e-2):
    """对照两个实现，返回最大绝对误差。"""
    a, b = a.detach().float(), b.detach().float()
    err = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:<52s} max|Δ| = {err:.3e}")
    return ok


def timer(fn, warmup=2, repeat=5):
    """简单计时（CPU）。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3  # ms


def gemm_flops(M, N, K, batch=1):
    """矩阵乘法浮点运算数：2MNK（乘加各算一次）"""
    return 2.0 * batch * M * N * K


def bytes_of(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))


def title(s, ch=None):
    print("\n" + "=" * 78)
    print(f"  CH{ch} | {s}" if ch else f"  {s}")
    print("=" * 78)


def section(s):
    print(f"\n--- {s} " + "-" * max(0, 70 - len(s)))
