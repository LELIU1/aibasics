"""
CH05 FlashAttention：在线 softmax + 分块矩阵乘 = 把 O(N²) 显存压成 O(N)

数学核心（两件事）
  ① 在线 softmax（running max / running sum + 重缩放）
     把 softmax 的分母与分子拆成可增量维护的量：
        m_new = max(m_old, rowmax(S))
        ℓ_new = ℓ_old · e^{m_old - m_new} + rowsum(e^{S - m_new})
        O_new = O_old · e^{m_old - m_new} + e^{S - m_new} V
     正确性来自 softmax 的平移不变性 + 指数函数的可分解性。
  ② 分块矩阵乘法：S = Q_i K_j^T，O += P V_j，中间块只活在 SRAM 里，
     用【重缩放因子】把不同块的贡献按统一的 m 对齐后累加（就是分块矩阵乘法 + 行缩放）。

学习目标
  1. 手推在线 softmax，并验证与朴素实现数值等价
  2. 实现两级分块的 FlashAttention（纯 PyTorch），验证等价
  3. 算清 IO 复杂度的账：为什么它快
  4. 反向传播：不存 N×N 矩阵，靠重算换显存
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)

NEG = float("-inf")


# ------------------------------------------------------- 1. 在线 softmax 推导
def demo_online_softmax():
    section("1. 在线 softmax：把 softmax 改成可流式的递推")
    x = torch.randn(4, 10) * 8                       # 假装是一行 logits 被切成若干块

    ref = torch.softmax(x, -1)

    # 逐块（每块 3 个元素）流式更新
    m = torch.full((4, 1), NEG)
    l = torch.zeros(4, 1)
    o = torch.zeros(4, 10)                            # 这里 o 存的是未归一化的加权和（输入用 one-hot）
    ident = torch.eye(10)
    for j in range(0, 10, 3):
        xj = x[:, j:j + 3]
        m_new = torch.maximum(m, xj.max(-1, keepdim=True).values)
        corr = torch.exp(m - m_new)                   # 老 max 与新 max 的差 → 重缩放
        corr = torch.nan_to_num(corr, nan=0.0)        # m_old = -inf 时修正为 0
        p = torch.exp(xj - m_new)
        l = l * corr + p.sum(-1, keepdim=True)
        o = o * corr + p @ ident[j:j + 3]
        m = m_new
    check("在线 softmax(块=3) == 一次性 softmax", o / l, ref)

    # 更直观：用极端数值展示为什么必须维护 running max
    big = torch.tensor([[1e4, 2e4, 3e4]])
    m = torch.tensor([[NEG]]); l = torch.zeros(1, 1)
    for v_ in big[0].tolist():
        xv = torch.tensor([[v_]])
        m_new = torch.maximum(m, xv)
        corr = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
        l = l * corr + torch.exp(xv - m_new)
        m = m_new
    print(f"  逐元素在线 log-sum-exp(1e4,2e4,3e4) = {(m + l.log()).item():.4f}")
    print(f"  torch.logsumexp 参考               = {big.logsumexp(-1).item():.4f}")
    print(f"  天真 torch.exp(3e4)                = {big.exp().tolist()} ← 上溢")
    print("  → 关键恒等式：log Σ e^{x} = m + log Σ e^{x-m}，m 分块单调递增，用重缩放对齐")


# -------------------------------------------------- 2. 分块 FlashAttention
def flash_attention(Q, K, V, bq=64, bk=64, causal=False, scale=None):
    """两级分块：外层切 Q，内层切 K/V。中间块只有 bq×bk，永不落 HBM。
    Q,K,V: (N, d) 单头；返回 (N, d)
    """
    N, d = Q.shape
    s = scale if scale is not None else d ** -0.5
    O = torch.zeros(N, d)
    L = torch.zeros(N, 1)                              # 只额外存 O(N) 的 logsumexp，反向要用
    for i in range(0, N, bq):
        hi = min(i + bq, N)
        Qi = Q[i:hi]
        Oi = torch.zeros(hi - i, d)
        mi = torch.full((hi - i, 1), NEG)
        li = torch.zeros(hi - i, 1)
        j_end = (i + bq) if causal else N               # 因果：只看自己及之前
        for j in range(0, j_end, bk):
            hj = min(j + bk, N)
            Sij = (Qi @ K[j:hj].T) * s                  # (bq, bk) —— 只活在 SRAM
            if causal and hj > i:                       # 对角块内还要屏蔽"未来"
                rows = torch.arange(i, hi)[:, None]
                cols = torch.arange(j, hj)[None, :]
                Sij = Sij.masked_fill(cols > rows, NEG)
            m_new = torch.maximum(mi, Sij.max(-1, keepdim=True).values)
            corr = torch.nan_to_num(torch.exp(mi - m_new), nan=0.0)
            P = torch.exp(Sij - m_new)
            li = li * corr + P.sum(-1, keepdim=True)
            Oi = Oi * corr + P @ V[j:hj]
            mi = m_new
        O[i:hi] = Oi / li
        L[i:hi] = mi + li.log()                          # 保存 logsumexp 供反向使用
    return O, L


def naive_attention(Q, K, V, causal=False, scale=None):
    N, d = Q.shape
    s = scale if scale is not None else d ** -0.5
    S = (Q @ K.T) * s
    if causal:
        S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1), NEG)
    P = torch.softmax(S, -1)
    return P @ V, S.logsumexp(-1, keepdim=True)


def demo_flash_correctness():
    section("2. 分块 FlashAttention 与朴素实现的等价性")
    N, d = 512, 64
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    Q, K, V = Q * 3, K * 3, V * 3          # 放大数值，考验稳定性

    for causal in [False, True]:
        o_naive, l_naive = naive_attention(Q, K, V, causal)
        for bq, bk in [(128, 128), (64, 64), (32, 128), (16, 32)]:
            o_flash, l_flash = flash_attention(Q, K, V, bq, bk, causal)
            ok1 = check(f"causal={causal} bq={bq},bk={bk}: 输出", o_flash, o_naive, atol=1e-4)
            ok2 = check(f"{'':>13}logsumexp", l_flash, l_naive, atol=1e-3)
            if not (ok1 and ok2):
                return

    # 检验极端数值（1000 量级 logits，朴素实现会溢出）
    Q2, K2 = torch.randn(256, 64), torch.randn(256, 64)
    o_n, _ = naive_attention(Q2 * 50, K2 * 50, V[:256])
    o_f, _ = flash_attention(Q2 * 50, K2 * 50, V[:256], 64, 64)
    check("大 logits（±50×d 量级）下仍然一致", o_f, o_n, atol=1e-4)

    # 分块大小不影响结果（数学上是分块矩阵乘法，完全等价）
    a, _ = flash_attention(Q, K, V, 32, 32, causal=True)
    b, _ = flash_attention(Q, K, V, 256, 256, causal=True)
    check("不同分块大小结果一致（分块矩阵乘法）", a, b, atol=1e-5)


def demo_io_accounting():
    section("3. IO 复杂度：为什么 FlashAttention 快")
    print("  标准实现（HBM 读写次数，单位：元素数）")
    print("    ① 读 Q,K 写 S=QK^T        : O(Nd + N²)")
    print("    ② 读 S 写 P=softmax(S)    : O(N²)")
    print("    ③ 读 P,V 写 O=PV          : O(N² + Nd)")
    print("    ⇒ HBM 访问 Θ(N²)（三次 N×N 大矩阵的读与写）")
    print("  FlashAttention")
    print("    Q 分 N/Bq 块，每块遍历 K,V 的 N/Bk 块；块大小满足 (Bq+2·Bk)·d ≲ M（SRAM 容量）")
    print("    ⇒ 取 Bq·d ≈ M 时 HBM 访问 Θ(N²d²/M)，d²/M ≪ 1 时远小于 Θ(N²)")

    print("\n  具体算账（dtype=fp32, 4 bytes；设 SRAM M = 100 KB）:")
    M = 100 * 1024
    for N, d in [(1024, 64), (2048, 64), (8192, 128), (32768, 128)]:
        std = (3 * N * N + 2 * N * d) * 4 / 1e6
        flash = (N * N * d * d / M) * 4 / 1e6
        print(f"    N={N:>6d}, d={d:>3d}: 标准 {std:9.2f} MB  Flash {flash:8.2f} MB  "
              f"减少 {std/max(flash,1e-9):5.1f}x")
    print("  → 注意：Flash 的 FLOPs 一点没少（还是 O(N²d)），省的是【搬运】。")
    print("     这就是 'memory-bound' 算子的正确优化方向。")


def demo_backward_recompute():
    section("4. 反向传播：用重算换显存")
    N, d = 4096, 64
    elem = 4
    print(f"  N={N}, d={d}, fp32")
    print(f"    标准实现要保存 S(N×N) 与 P(N×N) 供反向: "
          f"{2*N*N*elem/1e6:.1f} MB")
    print(f"    FlashAttention 只保存 O(N×d) 与 logsumexp(N): "
          f"{(N*d + N)*elem/1e6:.2f} MB")
    print(f"    ⇒ 显存从 O(N²) 降到 O(N)，代价是反向时重算一次 QK^T（多花 ~1/3 FLOPs）")

    # 用 autograd 验证：分块前向 + 分块反向，梯度与朴素一致
    N2, d2 = 128, 32
    Q = torch.randn(N2, d2, requires_grad=True)
    K = torch.randn(N2, d2, requires_grad=True)
    V = torch.randn(N2, d2, requires_grad=True)
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)

    o1, _ = naive_attention(Q, K, V, causal=True)
    o2, _ = flash_attention(Q2, K2, V2, 32, 32, causal=True)
    g = torch.randn_like(o1)
    o1.backward(g); o2.backward(g)
    check("梯度 ∂L/∂Q", Q.grad, Q2.grad, atol=1e-4)
    check("梯度 ∂L/∂K", K.grad, K2.grad, atol=1e-4)
    check("梯度 ∂L/∂V", V.grad, V2.grad, atol=1e-4)
    print("  → 分块版本的自动微分梯度与朴素版本完全一致（分块只是重排了浮点加法顺序）")


TRITON_KERNEL_NOTE = r'''
# ---------------------------------------------------------------------------
# 附：Triton 版 FlashAttention forward（需要 NVIDIA GPU，本沙箱为 CPU 故不执行）
# 关键：q 块常驻 SRAM，k/v 块循环载入，全程不写 N×N 矩阵
# ---------------------------------------------------------------------------
import triton, triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, O, L, sm_scale, N, D: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):      # 因果：只到对角
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale                        # BLOCK_M×BLOCK_N，在寄存器里
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_new) * 1.4426950408889634)     # 2^x 比 exp 快
        p = tl.math.exp2((s - m_new[:, None]) * 1.4426950408889634)
        acc = acc * alpha[:, None]
        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc, mask=offs_m[:, None] < N)
    tl.store(L + offs_m, m_i + tl.log(l_i), mask=offs_m < N)
'''


def main():
    title("FlashAttention：在线 softmax · 分块 · IO 复杂度", ch="05")
    demo_online_softmax()
    demo_flash_correctness()
    demo_io_accounting()
    demo_backward_recompute()
    print("\nCH05 完成。要点：在线 softmax 让 softmax 可流式；分块让它不落 HBM；反向只存 O(N) 并重算。\n")
    print(TRITON_KERNEL_NOTE)


if __name__ == "__main__":
    main()
