"""
CH08 GroupGEMM 与 MoE：块对角矩阵、置换路由与变长分组矩阵乘

数学主线
  MoE 层： y_t = Σ_{i ∈ TopK(x_t)} g_i(x_t) · E_i(x_t)
  把所有 token 按分配的专家排序后，整层可以写成一个【块对角矩阵】作用：
      Y = Pᵀ · blkdiag(E_0, E_1, ..., E_{E-1}) · P · X
  其中 P 是路由决定的置换矩阵。因此：
      · 路由 = 置换（对 token 做重排）
      · 专家计算 = 块对角矩阵乘法（每个块是一次 GEMM，块大小 = 该专家分到的 token 数）
      · GroupGEMM = 一组形状各异的 GEMM 打包成一个 kernel

学习目标
  1. 从置换矩阵 + 块对角矩阵理解 MoE 的代数结构
  2. 三种实现：逐专家循环 / padding 成定长 batched GEMM / 紧凑分组（GroupGEMM）
  3. 负载均衡为什么要加辅助损失：从块大小均匀性看 GPU 利用率
  4. 排序（permute）与还原（unpermute）的矩阵写法
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)


# ------------------------------------------------------------------ 数据
def make_moe(T=1024, d=256, ff=512, E=8, topk=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, d, generator=g)
    Wg = torch.randn(d, E, generator=g) * 0.1
    W1 = torch.randn(E, d, ff, generator=g) * 0.05      # 每个专家的 up 投影
    W2 = torch.randn(E, ff, d, generator=g) * 0.05      # 每个专家的 down 投影
    return x, Wg, W1, W2


def route(x, Wg, topk=2):
    logits = x @ Wg                                     # (T, E)
    prob = torch.softmax(logits, -1)
    topv, topi = prob.topk(topk, dim=-1)                # (T, topk)
    return topi, topv / topv.sum(-1, keepdim=True)      # 专家 id 与归一化权重


def expert_ffn(x, W1_i, W2_i):
    return F.gelu(x @ W1_i) @ W2_i


# --------------------------------------------------- 实现 1：逐专家循环
def moe_loop(x, Wg, W1, W2, topk=2):
    T, d = x.shape
    E = W1.shape[0]
    topi, w = route(x, Wg, topk)
    out = torch.zeros(T, d)
    for e in range(E):
        mask = (topi == e).any(-1)                      # 分到专家 e 的 token
        if not mask.any():
            continue
        xe = x[mask]                                    # gather
        ye = expert_ffn(xe, W1[e], W2[e])
        out[mask] += w[mask][topi[mask] == e].unsqueeze(-1) * ye   # scatter-add
    return out, topi


# -------------------------- 实现 2：排序 + 分组 GEMM（GroupGEMM 的语义）
def moe_grouped(x, Wg, W1, W2, topk=2, pad_to=None):
    """先按专家排序（置换），再对每个专家做一次 GEMM，最后加权还原（逆置换）。
    这是所有 MoE kernel 的通用套路；真实实现（DeepGEMM / Megablocks）用
    ragged 偏移量把这组变长 GEMM 塞进一个 kernel，避免 padding 浪费。
    """
    T, d = x.shape
    E = W1.shape[0]
    topi, w = route(x, Wg, topk)

    flat_id = topi.reshape(-1)                          # (T*topk,)
    flat_w = w.reshape(-1)
    flat_x = x.unsqueeze(1).expand(T, topk, d).reshape(T * topk, d)   # 每个 (token,专家) 一份

    order = flat_id.argsort()                           # 置换：按专家 id 排序
    xs = flat_x[order]
    ids_sorted = flat_id[order]
    counts = torch.bincount(flat_id, minlength=E)       # 每个专家的 token 数
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

    ys = torch.zeros_like(xs)
    for e in range(E):
        s, t = offsets[e].item(), offsets[e + 1].item()
        if t > s:
            ys[s:t] = expert_ffn(xs[s:t], W1[e], W2[e])

    # 逆置换 + 按 token 加权求和
    # xs[j] = flat_x[order[j]]，所以权重也要跟着置换：ys[j] 配 flat_w[order[j]]
    out = torch.zeros(T * topk, d)
    out[order] = ys * flat_w[order].unsqueeze(-1)       # scatter 回原顺序
    return out.view(T, topk, d).sum(1), counts, order, offsets


# --------------------------- 实现 3：块对角矩阵（纯矩阵视角，教学用）
def moe_blockdiag(x, Wg, W1, W2, topk=2):
    """把路由后的计算写成块对角矩阵乘法：
        Y = Σ_e (X ⊙ m_e) · W_e      ⟺   X_ext · blkdiag(W_e)  再按块求和
    其中 X_ext 把 token 复制 E 份并只在对应块放非零值。
    """
    T, d = x.shape
    E, ff = W1.shape[0], W1.shape[2]
    topi, w = route(x, Wg, topk)

    # 构造块对角权重 (E*d, E*ff)
    Wb1 = torch.zeros(E * d, E * ff)
    Wb2 = torch.zeros(E * ff, E * d)
    for e in range(E):
        Wb1[e * d:(e + 1) * d, e * ff:(e + 1) * ff] = W1[e]
        Wb2[e * ff:(e + 1) * ff, e * d:(e + 1) * d] = W2[e]

    out = torch.zeros(T, d)
    for k in range(topk):
        e_ids = topi[:, k]                                        # (T,) 第 k 个专家的选择
        mask = torch.zeros(T, E)
        mask.scatter_(1, e_ids[:, None], 1.0)                     # one-hot
        Xext = (mask.unsqueeze(-1) * x.unsqueeze(1)).reshape(T, E * d)   # 只在对应块非零
        H = Xext @ Wb1                                            # 块对角乘法 (T, E*ff)
        Y = F.gelu(H) @ Wb2                                       # (T, E*d)
        Yk = Y.view(T, E, d).gather(1, e_ids[:, None, None].expand(T, 1, d)).squeeze(1)
        out += Yk * w[:, k:k + 1]                                 # 加权求和
    return out


def demo_equivalence():
    section("1. 三种实现的等价性：循环 / 分组排序 / 块对角")
    x, Wg, W1, W2 = make_moe(T=256, d=64, ff=128, E=4, topk=2)
    o1, _ = moe_loop(x, Wg, W1, W2)
    o2, counts, order, offsets = moe_grouped(x, Wg, W1, W2)
    o3 = moe_blockdiag(x, Wg, W1, W2)
    check("逐专家循环 == 排序分组 GEMM", o1, o2, atol=1e-4)
    check("逐专家循环 == 块对角矩阵乘法", o1, o3, atol=1e-4)
    print(f"  各专家分到的 token 数: {counts.tolist()}  (总 = {counts.sum().item()} = T×topk)")
    print("  → 数学结构：路由是置换 P，专家是块对角 blkdiag(E_e)，MoE = Pᵀ·blkdiag·P")
    print("  → GroupGEMM 就是「一组 n_e 各不相同的 GEMM」：A_e (n_e×d) × W_e (d×ff)")


def demo_padding_waste():
    section("2. padding vs 紧凑（ragged）：算力浪费的账")
    import math
    print("  设 E=64 个专家，平均每个专家 256 个 token，考察不同负载不均程度下的 padding 浪费:")
    for skew in [0.0, 0.3, 0.6, 0.9]:
        torch.manual_seed(0)
        base = 256
        n = torch.tensor([base * (1 + skew * (torch.rand(()) * 2 - 1)) for _ in range(64)])
        n = n.round().long().clamp(min=1)
        total = n.sum().item()
        max_n = n.max().item()
        pad_total = max_n * 64
        print(f"    不均衡度={skew:.1f}: 实际 token {total:>6d}, padding 后 {pad_total:>6d}, "
              f"浪费 {(pad_total/total-1)*100:5.1f}%   (最大块 {max_n}, 最小块 {n.min().item()})")
    print("  → 这正是 GroupGEMM（ragged / 变长，不补齐）存在的理由：")
    print("     DeepGEMM、Megablocks 用偏移量数组索引，一个 kernel 处理所有变长组，零 padding。")


def demo_balance():
    section("3. 负载均衡：辅助损失在优化什么")
    T, E, topk = 4096, 8, 2
    print("  负载不均时，GPU 上不同专家组串行执行 → 总时间由【最大的块】决定")
    for name, probs in [("完美均衡", torch.ones(E) / E),
                        ("轻微倾斜", torch.tensor([.20, .18, .15, .13, .11, .09, .08, .06])),
                        ("严重倾斜", torch.tensor([.50, .20, .12, .08, .04, .03, .02, .01]))]:
        n = (probs * T * topk).round()
        util = n.sum().item() / (n.max().item() * E)
        # 辅助损失（Switch Transformer）：f·P，f 为 token 比例，P 为路由概率均值
        f = probs
        aux = (f * probs).sum().item() * E
        print(f"    {name:<8}: 每专家 token {[int(v) for v in n.tolist()]}")
        print(f"              利用率 {util*100:5.1f}%   辅助损失 f·P·E = {aux:.3f} (越均衡越小，下界=1)")
    print("  → 辅助损失在数学上就是让【块对角矩阵的各个块大小尽量相等】，")
    print("     等价于最大化 GPU 上 kernel 的算术强度与并行度。")


def demo_groupgemm_impl():
    section("4. GroupGEMM 的三种 PyTorch 写法与代价")
    E, d, ff = 8, 512, 1024
    torch.manual_seed(0)
    W = torch.randn(E, d, ff)
    counts = torch.tensor([64, 128, 32, 256, 96, 160, 48, 224])   # 变长
    xs = [torch.randn(int(c), d) for c in counts]
    ref = torch.cat([xs[e] @ W[e] for e in range(E)], 0)

    # (a) 逐组循环
    t_loop = timer(lambda: torch.cat([xs[e] @ W[e] for e in range(E)], 0))
    # (b) padding 成定长 → 一次 bmm
    maxn = int(counts.max())
    Xp = torch.zeros(E, maxn, d)
    for e in range(E):
        Xp[e, :int(counts[e])] = xs[e]
    def padded():
        out = torch.bmm(Xp, W)
        return torch.cat([out[e, :int(counts[e])] for e in range(E)], 0)
    out_pad = padded()
    t_pad = timer(padded)
    # (c) 紧凑拼接 + 一次大 GEMM（只在专家数=1 时等价；多专家需分组索引，这里用 mask 近似）
    check("padding 版与逐组循环结果一致", out_pad, ref, atol=1e-3)
    print(f"  逐组循环（{E} 次 GEMM）      : {t_loop:.1f} ms")
    print(f"  padding 成 ({E},{maxn},{d}) 一次 bmm : {t_pad:.1f} ms   "
          f"（多算 {(maxn*E/counts.sum().item()-1)*100:.0f}% 的 FLOPs）")
    print("  真实 GroupGEMM：一次 kernel launch，内部按 offsets 调度各组的 tile —— 无 padding、无多次 launch")
    print("  （PyTorch 层面无法完全表达，需要 Triton/CUDA；DeepGEMM 就是这个思路的 FP8 实现）")


def demo_permute():
    section("5. 置换的矩阵写法与反向")
    T, d = 8, 4
    x = torch.randn(T, d)
    perm = torch.randperm(T)
    # 置换矩阵 P：y = P x  ⟺  y[i] = x[perm[i]]
    P = torch.zeros(T, T)
    P[torch.arange(T), perm] = 1.0
    check("y = P·x == x[perm]", P @ x, x[perm])
    check("P 是正交矩阵（P⁻¹ = Pᵀ）", P @ P.T, torch.eye(T))
    # 还原：把 y 放回原位
    inv = torch.argsort(perm)
    check("逆置换 == Pᵀ", P.T @ (x[perm]), x)
    check("argsort(perm) 即逆置换", x[perm][inv], x)
    print("  → MoE 的 dispatch = P，combine = Pᵀ（加权后）")
    print("  → 置换矩阵是正交的，所以路由的反向就是把梯度按逆置换送回去（无需额外存储）")


def main():
    title("GroupGEMM / MoE：块对角矩阵 · 置换路由 · 变长分组 GEMM", ch="08")
    demo_equivalence()
    demo_padding_waste()
    demo_balance()
    demo_groupgemm_impl()
    demo_permute()
    print("\nCH08 完成。要点：MoE = 置换 + 块对角；GroupGEMM 用 ragged 索引消除 padding；均衡损失在优化块大小均匀度。\n")


if __name__ == "__main__":
    main()
