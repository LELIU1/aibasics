"""
CH01 线性代数的"物理层"：张量布局、stride、einsum、算术强度

学习目标
  1. 同一个数学对象（矩阵/张量）在内存里有多种"摆放方式"，stride 就是摆放规则
  2. einsum 是描述一切线性代数运算的通用语言，注意力/MLP/卷积都能一行写清
  3. 大模型里"算得快"的瓶颈通常不是 FLOPs，而是搬运字节数 —— 算术强度
"""
import torch
from _utils import check, timer, gemm_flops, bytes_of, title, section

torch.manual_seed(0)


# ---------------------------------------------------------------------------
def demo_stride():
    """1. stride：数学上的转置 vs 内存里的拷贝"""
    section("1. stride 与内存布局")
    A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("A =\n", A)
    print("A.stride()      =", A.stride(), "  连续:", A.is_contiguous())
    print("A.T.stride()    =", A.T.stride(), "  连续:", A.T.is_contiguous())
    print("A.T 与 A 共享内存:", A.T.data_ptr() == A.data_ptr())

    # permute 只改元数据，contiguous() 才会真的搬运数据
    x = torch.randn(2, 3, 4)
    for name, t in [("x                 ", x),
                    ("x.transpose(0,2)  ", x.transpose(0, 2)),
                    ("x.permute(2,0,1)  ", x.permute(2, 0, 1))]:
        print(f"  {name} stride={str(t.stride()):<20s} contiguous={t.is_contiguous()}")

    # 教训：很多 kernel 假设连续输入，非连续张量会触发隐式拷贝（多一次读+写）
    v = torch.randn(2048, 2048)
    vt = v.t()
    t_contig = timer(lambda: v @ v)
    t_trans = timer(lambda: vt @ vt)
    print(f"  连续 GEMM: {t_contig:.2f} ms   转置视图 GEMM: {t_trans:.2f} ms  (BLAS 其实能直接吃转置)")

    # 但 einsum/自定义 kernel 就没这么好运
    print("  小结：transpose/permute 是 O(1) 的元数据操作，contiguous() 才是 O(n) 的搬运。")


# ---------------------------------------------------------------------------
def demo_einsum():
    """2. einsum：把所有线性代数运算统一到一套脚标记号"""
    section("2. einsum —— 线性代数的通用语言")
    B, N, d = 2, 6, 8            # batch, 序列长, 头维度
    Q = torch.randn(B, N, d)
    K = torch.randn(B, N, d)
    V = torch.randn(B, N, d)
    W = torch.randn(d, d)

    # (a) 矩阵乘法 / 线性层
    check("线性层 bij,jk->bik  ==  Q @ W", torch.einsum("bij,jk->bik", Q, W), Q @ W)

    # (b) batch 矩阵乘
    Xb, Yb = torch.randn(B, N, 4), torch.randn(B, 4, N)
    check("batch 矩阵乘 bnd,bdm->bnm == bmm", torch.einsum("bnd,bdm->bnm", Xb, Yb), torch.bmm(Xb, Yb))

    # (c) 注意力分数 = 批量双线性形式
    S = torch.einsum("bnd,bmd->bnm", Q, K)          # B,N,N
    check("注意力分数 bnd,bmd->bnm == Q@K^T", S, Q @ K.transpose(-2, -1))

    # (d) 加权求和
    O = torch.einsum("bnm,bmd->bnd", S.softmax(-1), V)
    check("加权求和 bnm,bmd->bnd", O, S.softmax(-1) @ V)

    # (e) 外积累加（线性注意力的核心！见 CH06）
    kv = torch.einsum("bnk,bnv->bkv", K, V)         # 把整段序列压成一个 d×d 状态矩阵
    print(f"  外积累加: K,V 形状 {tuple(K.shape)} -> 状态 {tuple(kv.shape)}  "
          f"（N 消失，这就是 O(N) 线性注意力的秘密）")

    # (f) 对角 / trace / 提取对角线
    M = torch.randn(d, d)
    check("trace ii->", torch.einsum("ii->", M), torch.trace(M))
    # (g) 双二次型 q^T M k
    q, k = torch.randn(d), torch.randn(d)
    check("双线性 i,ij,j->", torch.einsum("i,ij,j->", q, M, k), q @ M @ k)
    # (h) Hadamard 积与归约
    check("逐元素乘+求和 ij,ij->", torch.einsum("ij,ij->", M, M), (M * M).sum())
    # (i) 多头注意力一次写全：把 head 维当脚标
    H = 4
    Qh = torch.randn(B, H, N, d)
    Kh = torch.randn(B, H, N, d)
    Vh = torch.randn(B, H, N, d)
    Oh = torch.einsum("bhnd,bhmd->bhnm", Qh, Kh).softmax(-1)
    Oh = torch.einsum("bhnm,bhmd->bhnd", Oh, Vh)
    ref = torch.softmax(Qh @ Kh.transpose(-2, -1), -1) @ Vh   # 本例省略 1/sqrt(d) 缩放
    check("多头注意力 bhnd,bhmd->bhnm 全 einsum 版", Oh, ref)

    print("  einsum 速查：重复脚标=沿该轴相乘并求和；缺失脚标=求和掉；输出脚标=保留")


# ---------------------------------------------------------------------------
def demo_broadcast():
    """3. 广播 = 隐式外积。代价：中间张量被真实展开到内存"""
    section("3. 广播的内存代价（为什么 FlashAttention 要避免它）")
    # 真实规模：只算账，不真的分配（否则 OOM —— 这恰恰是重点）
    for N, d in [(4096, 64), (8192, 128), (32768, 128)]:
        temp = N * N * d * 4 / 1e9
        score = N * N * 4 / 1e6
        print(f"  seq={N:>6d}, d={d:>3d}: 广播中间张量 {temp:8.2f} GB, "
              f"注意力矩阵本身 {score:8.1f} MB  → 中间/结果 = {temp*1e9/(score*1e6):.0f}x")

    # 小规模实测：证明它真的会 materialize（用 chunk 方式避免，对比内存峰值）
    N, d, CH = 2048, 32, 256
    q, k = torch.randn(N, 1, d), torch.randn(1, N, d)
    out = torch.zeros(N, N)
    for s in range(0, N, CH):                       # 分块：一次只展开 CH×N×d
        out[s:s + CH] = (q[s:s + CH] - k).pow(2).sum(-1)
    naive = (q - k).pow(2).sum(-1)                  # 一次性展开 N×N×d
    check("分块广播 == 一次性广播", out, naive, atol=1e-4)
    print(f"  分块版峰值中间张量 {(CH*N*d*4)/1e6:.1f} MB vs 一次性 {(N*N*d*4)/1e6:.1f} MB "
          f"→ 省 {N/CH:.0f}x，数学结果完全一致")
    print("  → 这就是 attention 的 O(N^2) 内存墙，CH05 用 FlashAttention 彻底解决")


# ---------------------------------------------------------------------------
def demo_arithmetic_intensity():
    """4. 算术强度：FLOPs / Bytes = 每搬 1 字节能做多少次运算"""
    section("4. 算术强度与 roofline 直觉")
    N = 2048
    A = torch.randn(N, N)
    Bm = torch.randn(N, N)
    x = torch.randn(N * N)

    t_gemm = timer(lambda: A @ Bm)
    flops = gemm_flops(N, N, N)
    bytes_gemm = bytes_of(A, Bm) + N * N * 4
    ai_gemm = flops / bytes_gemm

    t_elt = timer(lambda: x * 2.0)
    flops_e = x.numel()
    bytes_e = bytes_of(x) * 2
    ai_elt = flops_e / bytes_e

    print(f"  GEMM   ({N}x{N}x{N}): {t_gemm:8.2f} ms  {flops/1e9:8.2f} GFLOP  "
          f"{flops/t_gemm/1e9*1e3:7.1f} GFLOP/s  算术强度={ai_gemm:7.1f} FLOP/Byte")
    print(f"  逐元素 ({x.numel():>8} 个): {t_elt:8.2f} ms  {flops_e/1e6:8.2f} MFLOP  "
          f"{flops_e/t_elt/1e9*1e3:7.2f} GFLOP/s  算术强度={ai_elt:7.2f} FLOP/Byte")
    print("  结论：GEMM 算术强度高 → 算力受限（好）；逐元素/softmax/归一化 → 带宽受限（坏）。")
    print("  Softmax、RMSNorm、RoPE、激活函数都属于后者，所以优化方向是【融合 kernel】。")


def demo_tiling_effect():
    """5. 分块为什么有效：把工作集塞进 cache"""
    section("5. 分块（tiling）的数值等价性与 cache 友好性")
    N, BS = 1024, 64
    A, B = torch.randn(N, N), torch.randn(N, N)
    C_ref = A @ B

    # 手写分块 GEMM：数学上完全等价（分块矩阵乘法）
    C = torch.zeros(N, N)
    for i in range(0, N, BS):
        for j in range(0, N, BS):
            for kk in range(0, N, BS):
                C[i:i + BS, j:j + BS] += A[i:i + BS, kk:kk + BS] @ B[kk:kk + BS, j:j + BS]
    check("分块 GEMM == A@B（分块矩阵乘法）", C, C_ref)
    print(f"  分块大小 {BS}：每块只需 resident A块+{BS}x{BS} 与 B块+{BS}x{BS}，"
          f"约 {2*BS*BS*4/1024:.0f} KB，能放进 L1/L2")
    print("  这正是 GPU 上 SRAM(≈几十MB共享内存) vs HBM(≈80GB) 的同一套故事 → CH05 FlashAttention")


def main():
    title("张量布局 · einsum · 算术强度", ch="01")
    demo_stride()
    demo_einsum()
    demo_broadcast()
    demo_arithmetic_intensity()
    demo_tiling_effect()
    print("\nCH01 完成。要点：stride 决定搬运成本，einsum 统一表达，算术强度决定你是算力受限还是带宽受限。\n")


if __name__ == "__main__":
    main()
