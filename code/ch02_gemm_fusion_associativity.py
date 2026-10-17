"""
CH02 GEMM 经济学：分块、结合律、矩阵吸收、算子融合

学习目标
  1. Transformer 里 90%+ 的 FLOPs 都是 GEMM，理解 GEMM 才能理解性能
  2. 矩阵乘法满足结合律 → 换乘法顺序能省一个数量级的 FLOPs（LoRA / 矩阵吸收）
  3. 对角矩阵、标量缩放可以"吸收"进邻近权重，代价为零
  4. 非 GEMM 的部分（softmax/norm/RoPE）是带宽瓶颈 → 必须融合
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, gemm_flops, bytes_of, title, section

torch.manual_seed(0)


def demo_gemm_census():
    """1. 数一数一个 Transformer 层里有多少 GEMM"""
    section("1. Transformer 的 GEMM 普查")
    B, N, d, h, ff = 8, 2048, 4096, 4096, 14336   # 8k 序列, 4k 隐层, MLP 14336
    H = 32
    D = d // H                                     # head_dim = 128
    rows = [
        ("QKV 投影",     3 * B * N * d * d, "x @ W_qkv"),
        ("Attention 分数", 2 * H * B * N * N * D, "Q @ K^T"),
        ("Attention 输出", 2 * H * B * N * N * D, "A @ V"),
        ("O 投影",        B * N * d * d, "attn_out @ W_o"),
        ("MLP up+gate",  2 * B * N * d * ff, "h @ W_up, h @ W_gate"),
        ("MLP down",     B * N * ff * d, "act @ W_down"),
    ]
    total = sum(f for _, f, _ in rows)
    for name, f, expr in rows:
        print(f"  {name:<16s} {f/1e9:9.2f} GFLOP  ({f/total*100:5.1f}%)   {expr}")
    print(f"  {'合计':<16s} {total/1e9:9.2f} GFLOP")

    print("\n  attention 占比随序列长度的变化（O(N²) vs O(N)）:")
    for N_ in [2048, 8192, 32768, 131072]:
        attn = 2 * (2 * H * B * N_ * N_ * D)
        other = (3 * B * N_ * d * d + B * N_ * d * d + 2 * B * N_ * d * ff + B * N_ * ff * d)
        print(f"    N={N_:>7d}: attention {attn/1e9:9.1f} GFLOP ({attn/(attn+other)*100:5.1f}%)  "
              f"其它 {other/1e9:8.1f} GFLOP")
    print("  → N 每翻倍，attention 涨 4 倍而其余只涨 2 倍，长上下文必然被 attention 主导")
    print("  → 这就是 FlashAttention / 线性注意力 / 稀疏 attention 存在的理由")


def demo_associativity():
    """2. 结合律：换乘法顺序，FLOPs 天差地别"""
    section("2. 结合律 (AB)C = A(BC) 的性能含义")

    # (a) LoRA: y = x (W + B A)^T
    d_in, d_out, r = 4096, 4096, 16
    B_n = 8 * 32
    x = torch.randn(B_n, d_in)
    W = torch.randn(d_out, d_in) * 0.02
    lora_B = torch.randn(d_out, r) * 0.02
    lora_A = torch.randn(r, d_in) * 0.02

    y_merge = x @ (W + lora_B @ lora_A).T          # 先合并权重，再乘
    y_chain = x @ W.T + (x @ lora_A.T) @ lora_B.T   # 链式：低秩路径单独走
    y_dense = x @ W.T + x @ (lora_B @ lora_A).T     # 先算 ΔW 再乘（最差）
    check("LoRA 权重合并 == 链式计算", y_merge, y_chain)
    check("LoRA 权重合并 == 先算 ΔW", y_merge, y_dense)

    f_merge = gemm_flops(B_n, d_out, d_in)
    f_chain = gemm_flops(B_n, d_out, d_in) + gemm_flops(B_n, r, d_in) + gemm_flops(B_n, d_out, r)
    f_dense = gemm_flops(d_out, r, d_in) + gemm_flops(B_n, d_out, d_in) * 2
    print(f"  合并权重: {f_merge/1e9:.2f} GFLOP | 链式(lora_A→lora_B): {f_chain/1e9:.2f} GFLOP "
          f"| 先算ΔW: {f_dense/1e9:.2f} GFLOP")
    print("  → 推理部署一定要 merge LoRA（W' = W + BA），服务时省掉低秩分支；")
    print("     训练时不能 merge（需要对 B、A 分别求梯度）")

    # (b) attention 换序失败案例 —— 这正是线性注意力存在的理由
    N, D = 4096, 64
    q, k, v = torch.randn(N, D), torch.randn(N, D), torch.randn(N, D)
    f_attn = gemm_flops(N, N, D) + gemm_flops(N, D, N)
    f_lin = gemm_flops(N, D, D) + gemm_flops(N, D, D)
    print(f"\n  (QK^T)V  顺序: {f_attn/1e6:.1f} MFLOP")
    print(f"  Q(K^T V) 顺序: {f_lin/1e6:.1f} MFLOP   （省 {f_attn/f_lin:.0f}x！）")
    print("  但 softmax(QK^T) 卡在中间，结合律用不上 → 去掉 softmax 就能换序 → CH06 线性注意力")
    ref = torch.softmax(q @ k.T / D ** .5, -1) @ v
    wrong = q @ (k.T @ v)
    err = (ref - wrong).abs().max().item()
    print(f"  （验证：直接换序结果不等价，最大偏差 {err:.3f} —— softmax 不可交换）")


def demo_absorb():
    """3. 对角矩阵与标量的吸收：零代价的数学技巧"""
    section("3. 矩阵吸收：把邻近运算揉进权重")
    B_n, d = 4, 512
    x = torch.randn(B_n, d)
    W = torch.randn(d, d)

    # (a) RMSNorm 的 gain 吸收进下一层权重:  (norm(x) ⊙ γ) @ W^T = norm(x) @ (diag(γ) W^T)
    def rmsnorm(x, g, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
    gamma = torch.randn(d)
    h = rmsnorm(x, gamma)
    y1 = h @ W.T
    # y = (n ⊙ γ) W^T = n (W·diag(γ))^T，其中 (W·diag(γ))[i,j] = W[i,j]·γ[j]  → 按列缩放
    y2 = rmsnorm(x, torch.ones_like(gamma)) @ (W * gamma[None, :]).T
    check("RMSNorm γ 吸收进 W（按列缩放 W·diag(γ)）", y1, y2)
    print("  注意方向：γ 作用在【输入维度】上 → 缩放 W 的列；若作用在输出维度则缩放行")

    # (b) attention 的 1/sqrt(d) 吸收进 W_q
    H, D = 8, 64
    N = 128
    q = torch.randn(B_n, H, N, D); k = torch.randn(B_n, H, N, D)
    s1 = torch.einsum("bhnd,bhmd->bhnm", q, k) / (D ** 0.5)
    s2 = torch.einsum("bhnd,bhmd->bhnm", q / (D ** 0.25), k / (D ** 0.25))
    s3 = torch.einsum("bhnd,bhmd->bhnm", q / (D ** 0.5), k)
    check("1/√d 拆成两个 1/d^0.25 吸收进 W_q/W_k", s1, s2)
    check("1/√d 整体吸收进 W_q", s1, s3)

    # (c) Softmax 的平移不变性 → 最大值可吸收
    logits = torch.randn(B_n, N, N) * 10
    p1 = torch.softmax(logits, -1)
    p2 = torch.softmax(logits - logits.max(-1, keepdim=True).values, -1)
    check("softmax 平移不变性（为在线 softmax 铺垫）", p1, p2)
    print("  → 同理，RoPE 不能吸收进 W（它依赖 token 位置 m），只能在线算 → CH03")


def demo_fusion():
    """4. 融合：数一数一次前向要读写多少字节"""
    section("4. 算子融合的账本：带宽 vs 算力")
    N, d = 4096, 4096
    x = torch.randn(N, d); W = torch.randn(d, d); b = torch.randn(d)

    flops = gemm_flops(N, d, d)
    # 不融合：matmul 写回 + 每个逐元素算子各读一次写一次
    bytes_unfused = bytes_of(x, W) + (N * d * 4) * (1 + 2 + 2 + 2)   # matmul出 + add + gelu + dropout
    bytes_fused = bytes_of(x, W) + N * d * 4                          # 只写回一次
    print(f"  GEMM {flops/1e9:.1f} GFLOP，输出 {N*d*4/1e6:.0f} MB")
    print(f"  不融合（matmul→add→gelu→mask 各读写一遍）: {bytes_unfused/1e6:.0f} MB  "
          f"算术强度 {flops/bytes_unfused:.1f}")
    print(f"  融合成一个 kernel                        : {bytes_fused/1e6:.0f} MB  "
          f"算术强度 {flops/bytes_fused:.1f}")
    print(f"  → 访存减少 {bytes_unfused/bytes_fused:.1f}x。这就是 torch.compile / Triton / CUDA 手写 kernel 的价值")

    # 实测：多次逐元素 pass vs 一次
    y = x @ W
    t_multi = timer(lambda: F.gelu(y + b))
    t_single = timer(lambda: y + b)
    print(f"  实测：gelu(add) 两个 pass {t_multi:.1f} ms vs 单个 pass {t_single:.1f} ms "
          f"（差 {t_multi/max(t_single,1e-6):.1f}x）")


def demo_head_merge():
    """5. head 合并：把小 GEMM 拼成大 GEMM"""
    section("5. 把 head 维塞进 batch 维 —— 小 GEMM → 大 GEMM")
    Bn, H, N, D = 4, 32, 2048, 128
    Q = torch.randn(Bn, H, N, D)
    W = torch.randn(H, D, D)
    # 写法 A：每个 head 单独一个 GEMM
    outA = torch.stack([Q[:, i] @ W[i].T for i in range(H)], dim=1)
    # 写法 B：把 head 挪到 batch 维，一次 bmm：(H, B*N, D) @ (H, D, D)
    def merged():
        q2 = Q.transpose(0, 1).reshape(H, Bn * N, D)
        return torch.bmm(q2, W.transpose(-2, -1)).reshape(H, Bn, N, D).transpose(0, 1)
    outB = merged()
    check("逐 head GEMM == 合并后 batched GEMM", outA, outB)
    tA = timer(lambda: torch.stack([Q[:, i] @ W[i].T for i in range(H)], dim=1))
    tB = timer(merged)
    print(f"  {H} 次小 GEMM: {tA:.1f} ms  →  1 次 bmm({H} 个 batch，每批 {Bn*N}x{D} @ {D}x{D}): "
          f"{tB:.1f} ms  (快 {tA/tB:.2f}x)")
    print("  → GPU 上差别更大：kernel launch 开销 + 每个小 GEMM 都读一遍权重")


def main():
    title("GEMM · 结合律 · 矩阵吸收 · 算子融合", ch="02")
    demo_gemm_census()
    demo_associativity()
    demo_absorb()
    demo_fusion()
    demo_head_merge()
    print("\nCH02 完成。要点：能合并的权重先合并（结合律），能吸收的对角/标量先吸收，非 GEMM 部分必须融合。\n")


if __name__ == "__main__":
    main()
