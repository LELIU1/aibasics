"""
CH04 Softmax 注意力：双线性形式、数值稳定、掩码、GQA 与 KV Cache

数学主线
  注意力 = 一个【数据依赖的线性算子】作用于 value 序列：
      O = A(X) · (X W_V),  A(X) = softmax( (XW_Q)(XW_K)^T / √d + M )
  A 是行随机矩阵（每行和为 1 的非负矩阵），所以 O 的每行是 V 行的凸组合。

学习目标
  1. 为什么要除以 √d：点积方差 = d 的完整推导与实证
  2. softmax 的数值稳定（平移不变性）+ log-sum-exp
  3. 因果掩码 = 加性 0/-∞ 矩阵，等价于把矩阵乘限制到下三角
  4. GQA/MQA 的矩阵视角：K/V 头的重复 = 块对角/广播结构
  5. KV Cache = 自回归时对 K、V 矩阵做行追加
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)


def demo_variance():
    section("1. 为什么要除以 √d：点积方差推导")
    for d in [16, 64, 128, 256, 1024, 4096]:
        q = torch.randn(20000, d)          # 各分量 iid N(0,1)
        k = torch.randn(20000, d)
        dot = (q * k).sum(-1)
        print(f"  d={d:>5}: Var(q·k) = {dot.var().item():9.2f}  (理论 = d = {d})   "
              f"std = {dot.std().item():7.2f} (理论 √d = {d**0.5:.2f})")
    print("  推导：q·k = Σ_i q_i k_i，各项独立零均值，"
          "Var(q_i k_i) = E[q_i²]E[k_i²] = 1 → Var = d，标准差 √d")
    print("  → d 越大 logits 越分散，softmax 越接近 one-hot，梯度趋于 0（饱和）")

    # 实证：温度对 softmax 熵与梯度的影响
    d = 512
    logits = (torch.randn(8, 256, d) @ torch.randn(8, d, 256)).requires_grad_(True)   # 未缩放
    for scale in [1.0, 1 / d ** 0.5, 1 / d]:
        p = torch.softmax(logits * scale, -1)
        H = -(p * (p + 1e-12).log()).sum(-1).mean().item()       # 平均香农熵
        g = torch.autograd.grad(torch.softmax(logits * scale, -1).sum(), logits)[0]
        print(f"    scale={scale:8.4f}: 平均熵 = {H:6.3f} （均匀分布熵 = {torch.log(torch.tensor(256.0)).item():.3f}）"
              f"  梯度范数 = {g.norm().item():.4e}")
    print("  → 1/√d 让 softmax 保持合适熵值，梯度不饱和、训练稳定")


def demo_stability():
    section("2. 数值稳定：softmax 的平移不变性与 log-sum-exp")
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"  softmax(x)      = {torch.softmax(x, -1).tolist()}")
    print(f"  softmax(x+1000) = {torch.softmax(x + 1000, -1).tolist()}   ← 平移不变")

    big = torch.tensor([1000.0, 1001.0, 1002.0])
    naive = big.exp() / big.exp().sum()
    print(f"  天真实现 exp(1000+)/Σ: {naive.tolist()}   ← 上溢成 nan")
    print(f"  稳定实现（减最大值）:   {torch.softmax(big, -1).tolist()}")

    # log-sum-exp：所有对数运算的统一原语
    def lse(a, dim=-1):
        m = a.max(dim, keepdim=True).values
        return m + (a - m).exp().sum(dim, keepdim=True).log()
    a = torch.randn(4, 100) * 20
    check("log Σ exp(a) == max + log Σ exp(a-max)", a.logsumexp(-1), lse(a).squeeze(-1))
    check("softmax 对数 == a - logsumexp(a)", torch.log_softmax(a, -1), a - a.logsumexp(-1, keepdim=True))
    print("  → 交叉熵损失 = logsumexp(logits) - logits[target]，天然数值稳定")
    print("  → 同样的 '减最大值' 技巧会在 CH05 里变成【在线 softmax】，是 FlashAttention 的数学核心")


def demo_mask():
    section("3. 因果掩码：加性 0/-∞ 矩阵")
    N = 6
    scores = torch.randn(N, N)
    mask_strict = torch.triu(torch.ones(N, N), 1).bool()      # 禁止看未来
    NEG = torch.finfo(torch.float32).min                       # 用 -inf 在某些 kernel 里会产生 nan

    out1 = torch.softmax(scores.masked_fill(mask_strict, float("-inf")), -1)
    out2 = torch.softmax(scores + torch.triu(torch.full((N, N), float("-inf")), 1), -1)
    out3 = torch.softmax(scores.masked_fill(mask_strict, NEG), -1)
    ref = torch.softmax(scores + torch.tril(torch.zeros(N, N)).masked_fill(mask_strict, -1e9), -1)
    check("masked_fill(-inf) == 加性矩阵", out1, out3)
    print("  三种等价写法：masked_fill / 加 -inf 矩阵 / 加 -1e9 软掩码")
    print("  注意：用 float('-inf') 时若整行被 mask（如 padding 全 masked）会出现 nan，"
          "实践中常用 finfo.min")

    # 掩码本质是：把 softmax 的输入限制在下三角 —— 数学上 A 变成下三角行随机矩阵
    A = out1
    print(f"  行和: {A.sum(-1).tolist()[:3]} ... （行随机矩阵）")
    print(f"  上三角全零？ {(A.triu(1) == 0).all().item()}  （因果 ⇒ A 是下三角矩阵）")


def mha(q, k, v, causal=False, scale=None):
    """标准多头注意力: q,k,v: (B,H,N,D)"""
    d = q.shape[-1]
    s = torch.einsum("bhnd,bhmd->bhnm", q, k) * (scale or d ** -0.5)
    if causal:
        n = q.shape[2]
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    return torch.einsum("bhnm,bhmd->bhnd", torch.softmax(s, -1), v)


def demo_gqa():
    section("4. GQA / MQA：K/V 头共享的矩阵视角")
    B, Hq, N, D = 2, 8, 128, 64
    for Hkv in [8, 4, 2, 1]:
        q = torch.randn(B, Hq, N, D)
        k = torch.randn(B, Hkv, N, D)
        v = torch.randn(B, Hkv, N, D)
        g = Hq // Hkv
        # 写法 1：显式复制（浪费显存）
        k_rep = k.repeat_interleave(g, dim=1)
        v_rep = v.repeat_interleave(g, dim=1)
        o1 = mha(q, k_rep, v_rep, causal=True)
        # 写法 2：用 expand + reshape 零拷贝广播（B,Hkv,1,N,D → B,Hq,N,D）
        k_b = k.unsqueeze(2).expand(B, Hkv, g, N, D).reshape(B, Hq, N, D)
        v_b = v.unsqueeze(2).expand(B, Hkv, g, N, D).reshape(B, Hq, N, D)
        o2 = mha(q, k_b, v_b, causal=True)
        check(f"GQA(Hq={Hq},Hkv={Hkv}) 复制版 == broadcast 版", o1, o2)
        cache_bytes = 2 * B * Hkv * N * D * 4
        print(f"    Hkv={Hkv}: KV Cache 显存 {cache_bytes/1e6:.2f} MB "
              f"（相比 MHA 省 {Hq/Hkv:.0f}x）  分组数 g={g}")
    print("  → MQA(Hkv=1) 省显存最狠但质量下降；GQA 是折中；数学上等价于 K/V 头的块重复")


def demo_kv_cache():
    section("5. KV Cache：自回归 = 对 K、V 矩阵做行追加")
    B, H, N, D = 1, 4, 64, 32
    q_full = torch.randn(B, H, N, D)
    k_full = torch.randn(B, H, N, D)
    v_full = torch.randn(B, H, N, D)

    # 全量因果注意力（训练/预填充）
    o_full = mha(q_full, k_full, v_full, causal=True)
    print(f"  预填充: Q,K,V 形状 {tuple(q_full.shape)} → O {tuple(o_full.shape)}")

    # 增量解码：逐步追加行
    k_cache = torch.zeros(B, H, 0, D)
    v_cache = torch.zeros(B, H, 0, D)
    outs = []
    for t in range(N):
        k_cache = torch.cat([k_cache, k_full[:, :, t:t + 1]], dim=2)   # 行追加
        v_cache = torch.cat([v_cache, v_full[:, :, t:t + 1]], dim=2)
        qt = q_full[:, :, t:t + 1]
        s = torch.einsum("bhd,bhmd->bhm", qt.squeeze(2), k_cache) * D ** -0.5
        ot = torch.einsum("bhm,bhmd->bhd", torch.softmax(s, -1), v_cache).unsqueeze(2)
        outs.append(ot)
    o_inc = torch.cat(outs, dim=2)
    check("增量解码（KV Cache）== 全量因果注意力", o_inc, o_full)

    # 显存账：为什么长上下文服务被 KV Cache 卡脖子
    for N_ in [4096, 32768, 131072]:
        for H_, D_, L_ in [(32, 128, 80)]:
            b = 2 * 1 * H_ * N_ * D_ * L_ * 2   # K,V × 2 bytes(bf16) × 层数
            print(f"    N={N_:>7d}, {L_}层, {H_}头, d={D_}: KV Cache {b/1e9:7.2f} GB (bf16)")
    print("  → 于是有了 MLA（低秩压缩 KV，CH10）、GQA、KV 量化、分页管理")


def demo_attention_as_operator():
    section("6. 注意力 = 数据依赖的线性算子")
    N, D, H = 256, 64, 8
    V = torch.randn(H, N, D)
    A = torch.softmax(torch.randn(H, N, N) * 3, -1)     # 行随机矩阵
    O = torch.bmm(A, V)
    print(f"  A 行和 = {A[0].sum(-1)[:3].tolist()} ... （凸组合权重）")
    print(f"  每行 O 是 V 各行的凸组合 → O 的行空间 ⊂ V 的行空间")
    rank_A = torch.linalg.matrix_rank(A[0]).item()
    rank_V = torch.linalg.matrix_rank(V[0]).item()
    rank_O = torch.linalg.matrix_rank(O[0]).item()
    print(f"  rank(A)={rank_A}, rank(V)={rank_V}, rank(O)={rank_O}  "
          f"→ rank(O) ≤ min(rank A, rank V)")
    # 稳定秩 stable_rank(A) = Σσ_i² / σ_max²：衡量"信息混合的自由度"
    print("  稳定秩 stable_rank(A) = Σσᵢ²/σₘₐₓ²  （1 = 秩1，N = 满秩）")
    for scale in [0.2, 1.0, 3.0, 8.0]:
        At = torch.softmax(torch.randn(H, N, N) * scale, -1)
        sv = torch.linalg.svdvals(At[0])
        H_ent = -(At[0] * (At[0] + 1e-12).log()).sum(-1).mean().item()
        sr = ((sv ** 2).sum() / sv[0] ** 2).item()
        print(f"    logits×{scale:<4}: 每行熵 = {H_ent:5.2f} / {torch.log(torch.tensor(float(N))).item():.2f}"
              f"   稳定秩 = {sr:6.2f} / {N}")
    print("  → logits 缩放小 ⇒ 注意力均匀 ⇒ A 接近秩1（每行都是平均值，信息被抹平）")
    print("     缩放过大 ⇒ 接近置换矩阵 ⇒ 稳定秩高（每个 query 只搬一个 value，混合能力弱）")
    print("  → attention sink / massive activation 等现象都能从这个矩阵视角解释")


def main():
    title("Softmax 注意力：方差 · 稳定性 · 掩码 · GQA · KV Cache", ch="04")
    demo_variance()
    demo_stability()
    demo_mask()
    demo_gqa()
    demo_kv_cache()
    demo_attention_as_operator()
    print("\nCH04 完成。要点：注意力是行随机矩阵作用在 V 上；1/√d 控制熵；causal ⇒ 下三角；KV Cache 是行追加。\n")


if __name__ == "__main__":
    main()
