"""
CH11 分布式线性代数 + 综合实战：张量并行、Ring Attention 与组装一个小模型

数学主线
  分布式就是把矩阵【分块】放到不同设备上：
    · 张量并行（TP）：按权重矩阵的行/列切分 —— 利用 Y = X·[A₁ A₂] = [XA₁, XA₂]（列切，concat）
                                              Y = X·[A₁; A₂] = XA₁ + XA₂（行切，all-reduce）
    · 序列并行（SP）：沿序列维切分，norm/dropout 等"逐 token"算子天然可分
    · Ring Attention：沿序列维切 Q 和 K/V，环形传递 K/V 块 —— 与 CH05 的在线 softmax 完全同源

学习目标
  1. Megatron 的 MLP 并行：为什么 A 列切、B 行切只需一次 all-reduce
  2. Ring Attention：把 CH05 的在线 softmax 用到跨设备
  3. 把前 10 章的组件组装成一个能跑的小模型，并画出每一层的"线性代数账单"
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)
NEG = float("-inf")


# ------------------------------------------------------- 1. 张量并行
def demo_tensor_parallel():
    section("1. 张量并行：按行/列切分权重矩阵")
    d_in, d_out, world = 256, 512, 4
    x = torch.randn(64, d_in)
    W = torch.randn(d_in, d_out) * 0.02

    # (a) 列切分：W = [W_1, ..., W_n]，Y = [XW_1, ..., XW_n]（concat，无需通信）
    W_col = W.chunk(world, dim=-1)
    y_col = torch.cat([x @ w for w in W_col], dim=-1)
    check("列切分 + concat == X @ W", y_col, x @ W, atol=1e-5)

    # (b) 行切分：W = [W_1; ...; W_n]（按【输入维】切），则 X 也要按特征维切成 [X_1, ..., X_n]
    #     Y = Σ_i X_i W_i  ——  每卡只算一个部分和，最后 all-reduce 相加
    W_row = W.chunk(world, dim=0)          # 每块 (d_in/world, d_out)
    x_row = x.chunk(world, dim=-1)         # 每块 (T, d_in/world)
    y_row = sum(xi @ wi for xi, wi in zip(x_row, W_row))
    check("行切分（X 同步按特征切）+ 求和 == X @ W", y_row, x @ W, atol=1e-5)
    print("  列切：W 按输出维切 → Y = X·[A₁ A₂] = [XA₁, XA₂]   输出天然拼接，零通信")
    print("  行切：W 按输入维切 → Y = [X₁ X₂]·[A₁;A₂] = Σ XᵢAᵢ  输出需要相加 → 一次 all-reduce")
    print("        注意行切要求【输入 X 也按特征维切好】，这正是它能接在列切后面的原因")

    # (c) Megatron MLP：Y = GeLU(XA)B，A 列切、B 行切 ⇒ 全程只要一次 all-reduce
    A = torch.randn(d_in, 4 * d_in) * 0.02          # up 投影
    B = torch.randn(4 * d_in, d_in) * 0.02          # down 投影
    ref = F.gelu(x @ A) @ B
    A_chunks = A.chunk(world, dim=-1)               # 列切
    B_chunks = B.chunk(world, dim=0)                # 行切（与 A 的列块一一对应）
    partials = [F.gelu(x @ a) @ b for a, b in zip(A_chunks, B_chunks)]
    y_tp = sum(partials)                            # 一次 all-reduce
    check("Megatron MLP（A 列切 + B 行切）== 单卡结果", y_tp, ref, atol=1e-5)
    print("\n  关键：GeLU 是逐元素非线性，它要求【被切的那一维是完整的】")
    print("        A 按列切 ⇒ XA 的中间结果按列分成 world 块，每块内部完整 ⇒ GeLU 可以就地算")
    print("        B 按行切 ⇒ 每块的 (4d/w × d) 与该中间块对齐 ⇒ 各自得到一个 (·,d) 的部分和")
    print("        ⇒ 只在最后 all-reduce 一次。若 A 行切、B 列切，通信次数仍是 1 次，")
    print("          但 all-reduce 的是中间激活 T×4d（量是输出的 4 倍）")

    # (d) 通信量对比
    print("\n  一次 MLP 的通信量（T token）：")
    for name, elem in [("一次 all-reduce (输出)", 64 * d_in), ("一次 all-reduce (中间激活)", 64 * 4 * d_in)]:
        print(f"    {name:<28}: {elem*2/1024:.1f} KB (bf16)")
    print("    → 所以 Megatron 选择「列切 + 行切」把通信压到最小")


# --------------------------------------------------- 2. Ring Attention
def ring_attention(Q, K, V, world=4, causal=False):
    """Ring Attention：world 个设备各持一段 Q/K/V，环形传递 K/V，用在线 softmax 累加。
    与 CH05 的 FlashAttention 用的是同一套 m/ℓ/O 递推，只是"块"变成了"设备上的段"。
    """
    N, d = Q.shape
    assert N % world == 0
    b = N // world
    Qb = Q.reshape(world, b, d)
    Kb = K.reshape(world, b, d)
    Vb = V.reshape(world, b, d)

    O = torch.zeros(world, b, d)
    m = torch.full((world, b, 1), NEG)
    l = torch.zeros(world, b, 1)
    for step in range(world):
        # 第 step 步：设备 i 收到来自设备 (i - step) mod world 的 K/V
        src = torch.tensor([(i - step) % world for i in range(world)])
        Kj, Vj = Kb[src], Vb[src]                          # (world, b, d)
        S = torch.bmm(Qb, Kj.transpose(1, 2)) * d ** -0.5  # (world, b, b)
        if causal:
            # 设备 i 的 query 全局位置 [i*b, (i+1)*b)，key 来自设备 src，位置 [src*b, ...)
            qi = torch.arange(world).view(world, 1, 1) * b + torch.arange(b).view(1, b, 1)
            kj = src.view(world, 1, 1) * b + torch.arange(b).view(1, 1, b)
            S = S.masked_fill(kj > qi, NEG)
        m_new = torch.maximum(m, S.max(-1, keepdim=True).values)
        corr = torch.nan_to_num(torch.exp(m - m_new), nan=0.0)
        P = torch.exp(S - m_new)
        l = l * corr + P.sum(-1, keepdim=True)
        O = O * corr + torch.bmm(P, Vj)
        m = m_new
    return (O / l).reshape(N, d)


def demo_ring():
    section("2. Ring Attention：把在线 softmax 用到跨设备")
    N, d, world = 256, 32, 4
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

    ref_nc = torch.softmax(Q @ K.T * d ** -0.5, -1) @ V
    check("Ring Attention（非因果）== 全局注意力", ring_attention(Q, K, V, world), ref_nc, atol=1e-4)

    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
    ref_c = torch.softmax(Q @ K.T * d ** -0.5 + torch.where(mask, -1e9, 0.0), -1) @ V
    check("Ring Attention（因果）== 全局因果注意力", ring_attention(Q, K, V, world, causal=True),
          ref_c, atol=1e-4)

    print("  流程：设备 i 持有 Q_i,K_i,V_i；第 step 步收到来自 (i-step) 的 K/V 块，")
    print("        用 CH05 的 m/ℓ/O 在线 softmax 递推累加，world 步后归一化")
    print("  → 显存：每个设备只持有 N/world 的 Q/K/V，注意力矩阵永远只有 (N/w)² 大小")
    print("  → 因果时的负载不均：设备 i 需要 0..i 的 K/V，设备 0 只读 1 块而设备 world-1 要读全部")
    print("     解决：把序列块按『i 与 world-1-i 配对』重排（ZigZag / striped 分区）使各设备步数均衡")
    print("  → 通信量：每步传 (N/world)·d 的 K/V，总 world·(N/world)·d = N·d（与单卡 KV 总量同阶）")


# ------------------------------------------------------- 3. 综合实战
class MiniBlock(torch.nn.Module):
    """一个把前 10 章组件都用上的 Transformer 块：
    RMSNorm(γ 已吸收) → GQA 注意力(分块 Flash + RoPE) → SwiGLU MLP
    """

    def __init__(self, d_model=128, H=8, Hkv=2, d_h=16, ff=256, bq=64, bk=64):
        super().__init__()
        self.d_model, self.H, self.Hkv, self.d_h = d_model, H, Hkv, d_h
        self.bq, self.bk = bq, bk
        self.W_Q = torch.nn.Parameter(torch.randn(d_model, H * d_h) * 0.02)
        self.W_K = torch.nn.Parameter(torch.randn(d_model, Hkv * d_h) * 0.02)
        self.W_V = torch.nn.Parameter(torch.randn(d_model, Hkv * d_h) * 0.02)
        self.W_O = torch.nn.Parameter(torch.randn(H * d_h, d_model) * 0.02)
        self.W_up = torch.nn.Parameter(torch.randn(d_model, ff) * 0.02)
        self.W_gate = torch.nn.Parameter(torch.randn(d_model, ff) * 0.02)
        self.W_down = torch.nn.Parameter(torch.randn(ff, d_model) * 0.02)
        self.gamma1 = torch.nn.Parameter(torch.ones(d_model))
        self.gamma2 = torch.nn.Parameter(torch.ones(d_model))

    def rmsnorm(self, x, gamma, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma

    def rope(self, t, cos, sin):
        t1, t2 = t.chunk(2, -1)
        return torch.cat((t1 * cos - t2 * sin, t2 * cos + t1 * sin), -1)

    def flash_causal(self, q, k, v):
        """分块因果注意力（CH05），q,k,v: (N, H, d_h)；内部按 (H, 块, d) 布局以适配 einsum"""
        N, H, d = q.shape
        bq, bk = self.bq, self.bk
        O = torch.zeros(N, H, d)
        for i in range(0, N, bq):
            hi = min(i + bq, N)
            qi = q[i:hi]                                    # (bq, H, d)
            Oi = torch.zeros(H, hi - i, d)
            mi = torch.full((H, hi - i, 1), NEG)
            li = torch.zeros(H, hi - i, 1)
            for j in range(0, hi, bk):          # 因果：只需扫到当前 Q 块右端
                hj = min(j + bk, N)
                S = torch.einsum("qhd,khd->hqk", qi, k[j:hj]) * d ** -0.5
                rows = torch.arange(i, hi).view(1, -1, 1)   # 全局 query 位置
                cols = torch.arange(j, hj).view(1, 1, -1)   # 全局 key 位置
                S = S.masked_fill(cols > rows, NEG)
                m_new = torch.maximum(mi, S.max(-1, keepdim=True).values)
                corr = torch.nan_to_num(torch.exp(mi - m_new), nan=0.0)
                P = torch.exp(S - m_new)
                li = li * corr + P.sum(-1, keepdim=True)
                Oi = Oi * corr + torch.einsum("hqk,khd->hqd", P, v[j:hj])
                mi = m_new
            O[i:hi] = (Oi / li).transpose(0, 1)             # 回到 (块, H, d)
        return O

    def forward(self, x, cos, sin):
        N = x.shape[-2]                                       # 单条序列 (N, d_model)
        H, Hkv, d_h, g = self.H, self.Hkv, self.d_h, self.H // self.Hkv
        h = self.rmsnorm(x, self.gamma1)
        q = (h @ self.W_Q).view(N, H, d_h)
        k = (h @ self.W_K).view(N, Hkv, d_h)
        v = (h @ self.W_V).view(N, Hkv, d_h)
        q = self.rope(q, cos[:, None, :], sin[:, None, :])
        k = self.rope(k, cos[:, None, :], sin[:, None, :])
        k = k.unsqueeze(2).expand(N, Hkv, g, d_h).reshape(N, H, d_h)   # GQA 广播（CH04）
        v = v.unsqueeze(2).expand(N, Hkv, g, d_h).reshape(N, H, d_h)
        o = self.flash_causal(q, k, v).reshape(N, H * d_h) @ self.W_O
        x = x + o
        h2 = self.rmsnorm(x, self.gamma2)
        x = x + (F.silu(h2 @ self.W_gate) * (h2 @ self.W_up)) @ self.W_down
        return x


def demo_capstone():
    section("3. 综合实战：组装一个 MiniBlock 并核对每步的数学")
    B, N, d_model, H, Hkv, d_h = 2, 256, 128, 8, 2, 16
    blk = MiniBlock(d_model=d_model, H=H, Hkv=Hkv, d_h=d_h, ff=256, bq=64, bk=64)
    x = torch.randn(B, N, d_model)

    # 构造 RoPE（半分布局，CH03）
    th = 1.0 / (10000 ** (torch.arange(0, d_h, 2).float() / d_h))
    pos = torch.arange(N).float()
    # 半分布局：前 d_h/2 与后 d_h/2 配对成旋转平面 ⇒ cos/sin 只需 (N, d_h/2)
    cos = torch.cos(pos[:, None] * th)                       # (N, d_h/2)
    sin = torch.sin(pos[:, None] * th)

    # GQA 广播版的 GEMM 可以合并：KV 只需 Hkv 份权重（CH04）
    print("  组件清单:")
    print("    · RMSNorm 的 γ 按列吸收进 W_Q/W_K/W_V（CH02）—— 本例保留显式形式便于读数")
    print("    · RoPE：d_h/2 个旋转平面，半分布局（CH03）")
    print("    · GQA：K/V 头广播，KV Cache 省 %.0fx（CH04）" % (H / Hkv))
    print("    · 分块因果注意力：在线 softmax + 分块（CH05），bq=64, bk=64")
    print("    · SwiGLU MLP：三个 GEMM")

    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        y = torch.stack([blk(x[b], cos, sin) for b in range(B)])
    print(f"\n  前向完成: 输出 {tuple(y.shape)}, 耗时 {(time.perf_counter()-t0)*1e3:.0f} ms (CPU)")

    # 核对：分块注意力与朴素实现一致
    q = blk.rope((blk.rmsnorm(x[0], blk.gamma1) @ blk.W_Q).view(N, H, d_h), cos[:, None, :], sin[:, None, :])
    k = blk.rope((blk.rmsnorm(x[0], blk.gamma1) @ blk.W_K).view(N, Hkv, d_h), cos[:, None, :], sin[:, None, :])
    v = (blk.rmsnorm(x[0], blk.gamma1) @ blk.W_V).view(N, Hkv, d_h)
    k = k.unsqueeze(2).expand(N, Hkv, H // Hkv, d_h).reshape(N, H, d_h)
    v = v.unsqueeze(2).expand(N, Hkv, H // Hkv, d_h).reshape(N, H, d_h)
    o_flash = blk.flash_causal(q, k, v)
    S = torch.einsum("nhd,mhd->hnm", q, k) * d_h ** -0.5
    S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1), NEG)
    o_naive = torch.einsum("hnm,mhd->nhd", torch.softmax(S, -1), v)
    check("MiniBlock 的分块注意力 == 朴素实现", o_flash, o_naive, atol=1e-4)

    # 参数量与 FLOPs 账单
    nparam = sum(p.numel() for p in blk.parameters())
    flops_attn = 2 * 2 * H * N * N * d_h
    flops_proj = 2 * N * d_model * (H * d_h + 2 * Hkv * d_h + H * d_h)
    flops_mlp = 2 * N * d_model * 256 * 3
    print(f"\n  单序列账单 (N={N}, d_model={d_model}):")
    print(f"    参数量           : {nparam/1e3:.1f} K")
    print(f"    Attention FLOPs  : {flops_attn/1e6:.1f} MFLOP  (O(N²) 部分)")
    print(f"    投影 FLOPs       : {flops_proj/1e6:.1f} MFLOP")
    print(f"    MLP FLOPs        : {flops_mlp/1e6:.1f} MFLOP")
    print(f"    KV Cache(GQA)    : {2*Hkv*d_h*N*2/1024:.1f} KB/序列 (bf16)")
    print("  → 把 N 换成 128K，attention 项会涨 (128K/256)² = 512² = 26.2 万倍，其它只涨 500 倍")


def main():
    title("分布式线性代数 + 综合实战", ch="11")
    demo_tensor_parallel()
    demo_ring()
    demo_capstone()
    print("\nCH11 完成。要点：TP 用行/列切分换通信量；Ring Attention 是在线 softmax 的分布式版本。\n")


if __name__ == "__main__":
    main()
