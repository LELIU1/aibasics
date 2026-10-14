"""
CH06 线性注意力：外积状态矩阵、秩1更新与 chunkwise 并行

数学主线
  去掉 softmax 后，(QK^T)V 可以用结合律改写成 Q(K^T V)：
      O_t = Σ_{j≤t} (q_t^T k_j) v_j = S_t q_t,   S_t = Σ_{j≤t} v_j k_j^T  (d_v × d_k)
  S_t 是一个【矩阵值状态】，每步做一次秩1更新（外积累加）：
      S_t = S_{t-1} + v_t k_t^T
  复杂度从 O(N²d) 降到 O(N d²)，且解码时是 O(d²) 的常数代价。

学习目标
  1. 状态矩阵 = 外积累加（秩1更新），理解它的代数含义
  2. 加入指数衰减 λ：遗忘机制，等价于对状态做谱收缩
  3. chunkwise 并行：序列分块后【块内二次 + 块间状态传递】，这是所有现代线性注意力的训练算法
  4. 核特征映射 φ：用随机特征逼近 softmax（Performer）
  5. 线性注意力失效的根源：状态秩增长 / 记忆冲突 → 引出 CH07 的 delta rule
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)


# ------------------------------------------------------------- 1. 外积状态
def demo_outer_product_state():
    section("1. 状态矩阵 = 外积累加（秩1更新）")
    d, N = 8, 32
    K = torch.randn(N, d); V = torch.randn(N, d); Q = torch.randn(N, d)

    # 线性注意力的两种等价写法
    P = torch.tril(torch.ones(N, N))
    O_quad = (Q @ K.T * P) @ V                      # O(N²d)：二次形式
    # 因果版：逐个 token 累加
    S_t = torch.zeros(d, d)
    O_rec = []
    for t in range(N):
        S_t = S_t + torch.outer(V[t], K[t])        # 秩1更新：S += v ⊗ k
        O_rec.append(S_t @ Q[t])
    O_rec = torch.stack(O_rec)
    check("循环递推 S_t = Σ_{j≤t} v_j k_j^T  ==  (tril(QK^T))V", O_rec, O_quad)

    print(f"  S 的形状 {tuple(S_t.shape)}：与序列长度无关 → 解码时状态是常数大小 O(d²)")
    print(f"  秩: rank(S_1)=1, 每步 +1 ⇒ rank(S_t) ≤ min(t, d)；t > d 后状态秩饱和")
    print("  → 秩被 d 卡住 ⇒ 存不下太多独立记忆 ⇒ 这就是线性注意力的表达力瓶颈")
    print("  → 但 Δ rule（CH07）会【主动擦除】，让状态保持在「有用的低秩」上")


# --------------------------------------------------------- 2. 衰减与遗忘
def demo_decay():
    section("2. 指数衰减 = 遗忘门 = 状态的谱收缩")
    d, N = 16, 64
    K, V, Q = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    lam = 0.95

    S = torch.zeros(d, d)
    O_rec = []
    for t in range(N):
        S = lam * S + torch.outer(V[t], K[t])
        O_rec.append(S @ Q[t])
    O_rec = torch.stack(O_rec)

    # 矩阵形式：decay 矩阵 D[t,j] = λ^{t-j} (j ≤ t)
    idx = torch.arange(N)
    D = torch.where(idx[:, None] >= idx[None, :], lam ** (idx[:, None] - idx[None, :]).float(), 0.0)
    O_mat = ((Q @ K.T) * D) @ V
    check("带衰减的递推 == (λ^{t-j} ⊙ QK^T)V", O_rec, O_mat)
    print(f"  λ={lam}: 距离 Δ 的记忆权重 λ^Δ → Δ=1:{lam:.3f}  Δ=10:{lam**10:.3f}  Δ=50:{lam**50:.4f}")
    print("  数学上 S ← λS 是对状态做谱缩放：σ_i(S) → λ·σ_i(S)，所有奇异值等比收缩")
    print("  → 数据依赖的 λ_t（Gated DeltaNet 的 α_t）让模型自己决定「忘多少」")


# ------------------------------------------------- 3. chunkwise 并行训练
def linear_attn_chunkwise(Q, K, V, lam=0.95, chunk=16):
    """chunkwise 线性注意力：块内用二次形式（可并行），块间只传状态矩阵。
    Q,K,V: (N, d) 单头。返回 (N, d)
    """
    N, d = Q.shape
    O = torch.zeros(N, d)
    S = torch.zeros(d, d)                                  # 跨 chunk 的状态
    idx = torch.arange(chunk)
    # 块内相对衰减矩阵：Δ[t,j] = λ^{t-j} for j ≤ t
    dec = torch.where(idx[:, None] >= idx[None, :],
                      lam ** (idx[:, None] - idx[None, :]).float(), 0.0)
    # 块内从状态衰减到每个位置的系数
    a_rel = lam ** (idx.float() + 1)                        # 位置 t 相对块起点的累积衰减 λ^{t+1}

    for c in range(0, N, chunk):
        h = min(chunk, N - c)
        Qc, Kc, Vc = Q[c:c + h], K[c:c + h], V[c:c + h]
        d_ = dec[:h, :h]
        # ① 块内（intra）：二次注意力形式，O(chunk² d)，全部并行
        intra = ((Qc @ Kc.T) * d_) @ Vc
        # ② 块间（inter）：状态与 query 的一次矩阵乘，O(chunk·d²)，乘上块内累积衰减
        inter = (Qc @ S.T) * a_rel[:h, None]                 # (h,d_v) ← λ^{t+1}·(S q_t)
        O[c:c + h] = intra + inter
        # ③ 推进状态到块末：S_{h-1} = λ^h S_start + Σ_j λ^{h-1-j} v_j k_jᵀ
        coeff = lam ** (h - 1 - idx[:h].float())             # [λ^{h-1}, ..., λ^0]
        S = (lam ** h) * S + (Vc * coeff[:, None]).T @ Kc
    return O


def demo_chunkwise():
    section("3. chunkwise 并行：所有现代线性注意力的训练算法")
    d, N, chunk = 32, 128, 16
    K, V, Q = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

    for lam in [1.0, 0.98, 0.9]:
        S = torch.zeros(d, d)
        O_rec = []
        for t in range(N):
            S = lam * S + torch.outer(V[t], K[t])
            O_rec.append(S @ Q[t])
        O_rec = torch.stack(O_rec)
        O_ch = linear_attn_chunkwise(Q, K, V, lam, chunk)
        check(f"λ={lam}: chunkwise == 逐步递推", O_ch, O_rec)

    print("  为什么必须分块？")
    print("    · 纯递推：序列方向有 N 步串行依赖，GPU 完全跑不满（只有 d×d 的矩阵乘）")
    print("    · 纯二次：O(N²d) 又回到平方复杂度")
    print("    · chunkwise：块内 O(C²d) 用 tensor core 打满，块间只有 N/C 步串行状态传递")
    print(f"    · 取 C=64 时，串行步数从 {N} 降到 {N//64}，块内还能用 matmul")

    # 复杂度对照
    print("\n  复杂度对照（N=序列长, d=头维, C=块大小）:")
    print("    Softmax 注意力  训练 O(N²d)      解码 O(Nd) / 步   状态 O(Nd)")
    print("    线性注意力(递推) 训练 O(Nd²)      解码 O(d²) / 步    状态 O(d²)")
    print("    chunkwise 线性   训练 O(N·C·d + N·d²)     解码 O(d²) / 步   状态 O(d²)")
    print("      （块内二次 N·C·d，块间 N/C 次状态更新各 O(C·d²)）")


# ------------------------------------------------------------ 4. 核特征
def demo_kernel_features():
    section("4. 核技巧：用 φ(q)·φ(k) 逼近 softmax")
    N, d, D = 128, 64, 256          # D = 随机特征维数
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

    def softmax_attn(Q, K, V):
        return torch.softmax(Q @ K.T / d ** 0.5, -1) @ V

    def relu_feature(x):
        return F.relu(x) + 1.0       # 保证非负

    def random_feature(x, D, seed=0):
        """FAVOR+（三角随机特征）：
        φ(x)_i = e^{‖x‖²/2} · √(2/D) · cos(w_iᵀx + b_i),  w_i~N(0,I), b_i~U(0,2π)
        则 E[φ(x)·φ(y)] = e^{‖x‖²/2+‖y‖²/2} · e^{-‖x-y‖²/2} = e^{xᵀy}
        """
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d, D, generator=g)
        b = torch.rand(D, generator=g) * 2 * 3.141592653589793
        return torch.exp((x ** 2).sum(-1, keepdim=True) / 2) * (2.0 / D) ** 0.5 * torch.cos(x @ W + b)

    def kernel_attn(qf, kf, V):
        """φ(Q)φ(K)ᵀV / φ(Q)φ(K)ᵀ1 —— 分母是核版的行归一化"""
        S = qf @ kf.T
        return (S @ V) / S.sum(-1, keepdim=True)

    # 为公平对比，把 q,k 归一化到单位球，用温度 T 控制 logits 尺度
    # 目标核：exp(T·q̂ᵀk̂)，即 x=√T q̂, y=√T k̂ 时近似 exp(xᵀy)
    T = 2.0
    Qn = Q / Q.norm(dim=-1, keepdim=True)
    Kn = K / K.norm(dim=-1, keepdim=True)
    o_soft = torch.softmax(T * (Qn @ Kn.T), -1) @ V

    x, y = (T ** 0.5) * Qn, (T ** 0.5) * Kn
    print(f"  归一化输入 + 温度 T={T}：与 softmax 注意力的相对误差")
    qf, kf = relu_feature(x), relu_feature(y)
    err = ((kernel_attn(qf, kf, V) - o_soft).norm() / o_soft.norm()).item()
    print(f"    {'ReLU 特征(φ=relu+1)':<28s} {err:.4f}")
    for D_ in [64, 256, 1024, 4096, 16384]:
        o_rf = kernel_attn(random_feature(x, D_), random_feature(y, D_), V)
        err = ((o_rf - o_soft).norm() / o_soft.norm()).item()
        print(f"    {'随机特征 FAVOR+(D=%d)' % D_:<28s} {err:.4f}")
    print("  注意：φ(x)·φ(y) 的方差 ∝ e^{‖x‖²+‖y‖²}，输入范数越大需要的特征维数越多。")
    print("  真实大模型 logits 尺度更宽（‖q‖²≈d），这正是 Performer 落地困难的原因之一。")
    print("  核技巧把 softmax 的 exp(q·k) 拆成 φ(q)·φ(k)，从而可以用结合律")
    print("  代价：需要 D ≫ d 才能逼近好，且非负性约束让表达力受限 → 后续工作转向直接用状态矩阵")


# --------------------------------------------------- 5. 记忆冲突 / 秩饱和
def demo_memory_conflict():
    section("5. 线性注意力为什么会失效：状态秩饱和与记忆冲突")
    d = 32
    torch.manual_seed(1)
    # 构造 d/2 个"应记住的 key-value 对"，然后看能否正确读出
    for n_pairs in [4, 8, 16, 32, 64]:
        dk = d
        Kmem = torch.randn(n_pairs, dk); Kmem = Kmem / Kmem.norm(dim=-1, keepdim=True)
        Vmem = torch.randn(n_pairs, dk)
        S = torch.zeros(dk, dk)
        for j in range(n_pairs):
            S += torch.outer(Vmem[j], Kmem[j])
        # 回读：q = k_j
        errs = []
        for j in range(n_pairs):
            got = S @ Kmem[j]
            errs.append((got - Vmem[j]).norm().item() / Vmem[j].norm().item())
        rank = torch.linalg.matrix_rank(S, tol=1e-4).item()
        print(f"  记忆 {n_pairs:>3} 对: 状态秩 = {rank:>3}/{d}   平均回读相对误差 = {sum(errs)/len(errs):.3f}")
    print("  → 记忆对数量超过 d 后，状态矩阵无法同时容纳所有 key-value（秩不够，互相干扰）")
    print("  → 解决思路①：增大 d（贵）  ②：加衰减（忘掉旧的）  ③：delta rule 主动擦除（CH07）")

    # DeltaNet 的 Δ rule 预览：n_pairs=32（已满秩）时对比，并展示多遍收敛
    print("\n  Δ rule 预览（n_pairs=32，状态已无空余秩）:")
    n_pairs = 32
    Kmem = torch.randn(n_pairs, d); Kmem = Kmem / Kmem.norm(dim=-1, keepdim=True)
    Vmem = torch.randn(n_pairs, d)
    S_lin = sum(torch.outer(Vmem[j], Kmem[j]) for j in range(n_pairs))
    lin_err = sum(((S_lin @ Kmem[j] - Vmem[j]).norm() / Vmem[j].norm()).item()
                  for j in range(n_pairs)) / n_pairs
    print(f"    线性累加（1 遍）                 : 回读相对误差 = {lin_err:.4f}")
    S = torch.zeros(d, d)
    for epoch in range(1, 9):
        for j in range(n_pairs):                              # Δ rule：先读、算误差、再修正
            S = S + torch.outer(Vmem[j] - S @ Kmem[j], Kmem[j]) / (Kmem[j] @ Kmem[j])
        if epoch in (1, 2, 4, 8):
            err = sum(((S @ Kmem[j] - Vmem[j]).norm() / Vmem[j].norm()).item()
                      for j in range(n_pairs)) / n_pairs
            print(f"    Δ rule（{epoch} 遍）               : 回读相对误差 = {err:.4f}")
    print("  → S ← S + (v - Sk)kᵀ/‖k‖²  = 先读出旧值、算残差、沿 k 方向修正（Widrow-Hoff / LMS）")
    print("  → 它【主动擦除】k 方向上已存的内容，因此不会被秩饱和拖垮 → 详见 CH07")


def main():
    title("线性注意力：外积状态 · 遗忘 · chunkwise · 核技巧", ch="06")
    demo_outer_product_state()
    demo_decay()
    demo_chunkwise()
    demo_kernel_features()
    demo_memory_conflict()
    print("\nCH06 完成。要点：状态矩阵是外积累加；衰减=谱收缩；chunkwise=块内二次+块间传状态；容量瓶颈靠 delta rule 解决。\n")


if __name__ == "__main__":
    main()
