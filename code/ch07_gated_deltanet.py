"""
CH07 Gated DeltaNet：Δ rule、Householder 变换、WY 表示与 chunkwise 并行

这是本教程数学密度最高的一章。核心递推（Gated Delta Rule）：

    S_t = α_t · S_{t-1} · (I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ
        = α_t S_{t-1} + β_t (v_t - α_t S_{t-1} k_t) k_tᵀ          （等价写法）

读法：
  · 先"读出"当前 key 关联的值：v̂ = α_t S_{t-1} k_t
  · 算残差：v_t - v̂
  · 沿 k_t 方向写入残差（秩1更新）
  这就是 Widrow-Hoff / LMS 的 delta rule，等价于在线最小二乘（RLS）的一步。

学习目标
  1. I - βkkᵀ 的谱结构：投影 / Householder 反射 / 收缩 —— 三种特例
  2. 为什么"擦除"能突破 CH06 的秩饱和
  3. chunkwise 并行：把 Σ_t 的转移矩阵 Π(I-βkkᵀ) 用【WY 表示】压缩
  4. 序列递推 → 三角线性系统求解（UT transform 的本质），这是能上 GPU 的关键
  5. 与 Mamba2 / SSD 的矩阵视角对比
"""
import torch
from _utils import check, timer, title, section

torch.manual_seed(0)


# ============================================================ 参考实现
def gdn_naive(Q, K, V, beta, alpha):
    """逐 token 循环（慢，但最易读）—— 作为正确性基准。
    Q,K,V: (B, N, d)   beta, alpha: (B, N)   S: (B, d, d)
    """
    B, N, d = Q.shape
    S = torch.zeros(B, d, d)
    outs = []
    for t in range(N):
        kt = K[:, t:t + 1]                                   # (B,1,d)
        a = alpha[:, t].view(B, 1, 1)
        b = beta[:, t].view(B, 1, 1)
        v_hat = a * torch.bmm(S, kt.transpose(1, 2)).transpose(1, 2)   # α_t · S k_t
        delta = V[:, t:t + 1] - v_hat                        # 残差
        S = a * S + b * torch.bmm(delta.transpose(1, 2), kt)  # 秩1更新 (d,1)@(1,d)
        outs.append(torch.bmm(S, Q[:, t:t + 1].transpose(1, 2)).transpose(1, 2))
    return torch.cat(outs, dim=1)


# ============================================================ chunkwise
def gdn_chunkwise(Q, K, V, beta, alpha, chunk=32):
    """chunkwise 并行版：块内全部用矩阵运算（无 token 循环），块间传递状态 S。

    块内推导（块起点状态记为 S₀，块内索引 t=0..C-1）
      ① 归一化衰减：c_t = Π_{i≤t} α_i，令 S_t = c_t·(S₀A_t + Z_t) 可把递推化为无衰减形式
         ⇒ 只需把写入项的 β 换成 β'_t = β_t / c_t
      ② 转移矩阵 A_t = Π_{i≤t}(I - β_i k_i k_iᵀ) 用 WY 表示：A_t = I - W_{0:t} K_{0:t}ᵀ
         W 由三角系统 (I + diag(β)·tril(G,-1)) W = diag(β)·K 解出，G = K Kᵀ
      ③ Z_t 的递推里出现 X_t = Z_{t-1}k_t，同样构成三角系统：
         (I + tril(G,-1)·diag(β)) X = tril(G,-1)·diag(β')·V，然后 D_t = β'_t v_t - β_t X_t
      ④ 输出：o_t = c_t·[ S₀q_t - Σ_{i≤t}(S₀w_i)(k_iᵀq_t) + Σ_{j≤t} D_j (k_jᵀq_t) ]
      ⑤ 块末状态：S_end = c_{C-1}·( S₀ - (W S₀ᵀ)ᵀ Kᵀ + Dᵀ K )
    """
    B, N, d = Q.shape
    assert N % chunk == 0, "N 必须是 chunk 的整数倍（实际实现会做 padding）"
    Nc = N // chunk
    Qc = Q.view(B, Nc, chunk, d)
    Kc = K.view(B, Nc, chunk, d)
    Vc = V.view(B, Nc, chunk, d)
    bc = beta.view(B, Nc, chunk)
    ac = alpha.view(B, Nc, chunk)

    c = torch.cumprod(ac, dim=-1)                       # 块内累积衰减 c_t
    bp = bc / c                                         # β'_t = β_t / c_t

    G = torch.matmul(Kc, Kc.transpose(-1, -2))          # (B,Nc,C,C) Gram 矩阵
    L1 = torch.tril(torch.ones(chunk, chunk), -1)       # 严格下三角
    Gt = G * L1                                         # tril(G, -1)

    # ② WY 表示：解 (I + diag(β)·tril(G,-1)) W = diag(β)·K
    A1 = torch.eye(chunk) + bc.unsqueeze(-1) * Gt       # [t,j] = δ_tj + β_t·G[t,j]
    W = torch.linalg.solve_triangular(A1.reshape(-1, chunk, chunk),
                                      (bc.unsqueeze(-1) * Kc).reshape(-1, chunk, d),
                                      upper=False).view(B, Nc, chunk, d)
    # ③ X 的三角系统：(I + tril(G,-1)·diag(β)) X = tril(G,-1)·diag(β')·V
    A2 = torch.eye(chunk) + Gt * bc.unsqueeze(-2)       # [t,j] = δ_tj + G[t,j]·β_j
    rhs2 = torch.matmul(Gt * bp.unsqueeze(-2), Vc)
    X = torch.linalg.solve_triangular(A2.reshape(-1, chunk, chunk),
                                      rhs2.reshape(-1, chunk, d),
                                      upper=False).view(B, Nc, chunk, d)
    D = bp.unsqueeze(-1) * Vc - bc.unsqueeze(-1) * X    # 写入项 (B,Nc,C,d)

    # ④ 输出所需的因果 mask：[t,j] = q_tᵀk_j，只保留 j ≤ t（key 在 query 之前）
    MmT = torch.matmul(Qc, Kc.transpose(-1, -2)) * torch.tril(torch.ones(chunk, chunk))

    O = torch.zeros_like(Qc)
    S = torch.zeros(B, d, d)
    for ci in range(Nc):                                 # 块间：只有 Nc 步串行
        WS = torch.matmul(W[:, ci], S.transpose(-1, -2))         # (B,C,d) = W·S₀ᵀ
        QS = torch.matmul(Qc[:, ci], S.transpose(-1, -2))        # (B,C,d) = Q·S₀ᵀ
        inter = QS - torch.matmul(MmT[:, ci], WS)                # S₀ 经转移矩阵后的贡献
        intra = torch.matmul(MmT[:, ci], D[:, ci])               # 块内新写入的贡献
        O[:, ci] = c[:, ci].unsqueeze(-1) * (inter + intra)
        # ⑤ 推进状态
        S = c[:, ci, -1].view(B, 1, 1) * (
            S - torch.matmul(WS.transpose(-1, -2), Kc[:, ci])    # -(W S₀ᵀ)ᵀ Kᵀ
              + torch.matmul(D[:, ci].transpose(-1, -2), Kc[:, ci]))  # + Dᵀ K
    return O.view(B, N, d)


# ============================================================ 1. 谱结构
def demo_householder():
    section("1. I - βkkᵀ 的谱结构：投影 / 反射 / 收缩")
    d = 8
    k = torch.randn(d); k = k / k.norm()
    I = torch.eye(d)
    Kn = k / k.norm()

    print("  取 ‖k‖=1，考察 T = I - β k kᵀ，作用在 k 方向上：T k = (1-β‖k‖²)k")
    for name, beta_val in [("β‖k‖²=0.5  收缩", 0.5),
                           ("β‖k‖²=1    正交投影", 1.0),
                           ("β‖k‖²=2    Householder 反射", 2.0)]:
        T = I - beta_val * torch.outer(Kn, Kn)
        eig = torch.linalg.eigvalsh(T)
        print(f"  {name:<28s} 特征值: {[f'{e:.2f}' for e in eig.sort().values.tolist()]}")
        if abs(beta_val - 1.0) < 1e-9:
            check("  P² = P（幂等 ⇒ 投影矩阵）", T @ T, T, atol=1e-5)
        if abs(beta_val - 2.0) < 1e-9:
            check("  HᵀH = I 且 H² = I（正交对合 ⇒ Householder）", T.T @ T, I, atol=1e-5)
            check("  H² = I", T @ T, I, atol=1e-5)
    print("  → Gated DeltaNet 用 β∈(0,1) 且 k 归一化 ⇒ 特征值 1-β ∈ (0,1)：沿 k 方向【收缩/擦除】")
    print("    其余 d-1 个正交方向特征值恒为 1：只动 k 这一个方向，不影响其它记忆")
    print("  → 这就是「擦除」的数学：沿 k 方向的分量被削弱 1-β 倍，再写入新值 v")


# ============================================================ 2. WY 表示
def demo_wy():
    section("2. WY 表示：把一长串秩1修正压成一个 d×C 矩阵")
    d, C = 6, 5
    torch.manual_seed(3)
    K = torch.randn(C, d); K = K / K.norm(dim=-1, keepdim=True)
    beta = torch.rand(C) * 0.9 + 0.05

    # 直接累乘得到前缀转移矩阵 A_t = T_0 T_1 ... T_t
    I = torch.eye(d)
    As = []
    A = I.clone()
    for t in range(C):
        A = A @ (I - beta[t] * torch.outer(K[t], K[t]))
        As.append(A.clone())

    # WY 表示：A_t = I - W_{0:t} K_{0:t}ᵀ，W 由三角系统解出
    G = K @ K.T                                      # (C,C)
    Gt = torch.tril(G, -1)
    Aw = torch.eye(C) + beta[:, None] * Gt           # (C,C): I + diag(β)·tril(G,-1)
    W = torch.linalg.solve_triangular(Aw, beta[:, None] * K, upper=False)   # (C,d)

    for t in range(C):
        Wy = I - W[:t + 1].T @ K[:t + 1]             # (d,d): I - Σ_{i≤t} w_i k_iᵀ
        check(f"t={t}: Π_{{i≤t}}(I-β_i k_i k_iᵀ) == I - W_{{0:t}}K_{{0:t}}ᵀ", Wy, As[t], atol=1e-5)

    print("  WY 表示（Bischof–Van Loan / LAPACK 紧凑 Householder）:")
    print("    Π_{i≤t}(I - β_i k_i k_iᵀ) = I - W_{0:t} K_{0:t}ᵀ")
    print("    w_t = β_t ( k_t - Σ_{i<t} (k_iᵀk_t) w_i )   ← 下三角递推，等价于三角线性系统")
    print("  → 意义：C 次 d×d 矩阵乘法（O(C d²)，串行） 变成 一次 C×C 三角求解（可 batched、可并行）")
    print("  → 这是 Householder QR 分块化的同一个技巧，被搬到了序列建模里")


# ============================================================ 3. 正确性
def demo_correctness():
    section("3. chunkwise 实现 vs 逐 token 循环")
    B, H = 2, 4                                     # 当成 (B*H) 个独立序列
    for N, d, chunk in [(64, 16, 16), (128, 32, 32), (256, 32, 64)]:
        Q = torch.randn(B * H, N, d)
        K = torch.randn(B * H, N, d)
        V = torch.randn(B * H, N, d)
        K = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-6)     # 关键：k 归一化
        beta = torch.sigmoid(torch.randn(B * H, N)) * 0.9 + 0.05
        alpha = torch.sigmoid(torch.randn(B * H, N) * 0.3) * 0.2 + 0.8   # α ∈ (0.8, 1.0)
        o_ref = gdn_naive(Q, K, V, beta, alpha)
        o_ch = gdn_chunkwise(Q, K, V, beta, alpha, chunk=chunk)
        check(f"N={N},d={d},chunk={chunk}: chunkwise == naive", o_ch, o_ref, atol=1e-4)

    # 极端设置：α=1（无衰减，纯 DeltaNet）、β=1（完全替换）
    N, d = 64, 16
    Q, K, V = torch.randn(4, N, d), torch.randn(4, N, d), torch.randn(4, N, d)
    K = K / K.norm(dim=-1, keepdim=True)
    for a_val, b_val, tag in [(1.0, 1.0, "α=1,β=1（纯 DeltaNet，完全替换）"),
                              (1.0, 0.5, "α=1,β=0.5（部分擦除）"),
                              (0.9, 0.9, "α=0.9,β=0.9（强遗忘+强写入）")]:
        alpha = torch.full((4, N), a_val); beta = torch.full((4, N), b_val)
        check(tag, gdn_chunkwise(Q, K, V, beta, alpha, 16),
              gdn_naive(Q, K, V, beta, alpha), atol=1e-4)


# ============================================================ 4. 擦除的威力
def demo_erase_power():
    section("4. 为什么「擦除」能突破秩饱和（对比 CH06）")
    d = 32
    torch.manual_seed(1)
    print("  任务：往状态里塞 n 个 key-value 对，然后回读，看相对误差")
    print(f"  {'n_pairs':>8} {'线性累加':>10} {'gated(α=0.9)':>13} {'Δrule':>10} {'gated Δrule':>13}")
    for n_pairs in [8, 16, 32, 64]:
        Km = torch.randn(n_pairs, d); Km = Km / Km.norm(dim=-1, keepdim=True)
        Vm = torch.randn(n_pairs, d)
        alpha_v = 0.9

        def err(S):
            return sum(((S @ Km[j] - Vm[j]).norm() / Vm[j].norm()).item()
                       for j in range(n_pairs)) / n_pairs

        # ① 线性累加（无擦除、无衰减）
        S1 = sum(torch.outer(Vm[j], Km[j]) for j in range(n_pairs))
        # ② 只有衰减（Mamba2 式：S ← αS + v kᵀ）
        S2 = torch.zeros(d, d)
        for j in range(n_pairs):
            S2 = alpha_v * S2 + torch.outer(Vm[j], Km[j])
        # ③ 只有 Δ rule（无衰减）
        S3 = torch.zeros(d, d)
        for j in range(n_pairs):
            S3 = S3 + torch.outer(Vm[j] - S3 @ Km[j], Km[j])
        # ④ Gated Δ rule（衰减 + 擦除，多遍收敛）
        S4 = torch.zeros(d, d)
        for _ in range(8):
            for j in range(n_pairs):
                v_hat = alpha_v * (S4 @ Km[j])
                S4 = alpha_v * S4 + torch.outer(Vm[j] - v_hat, Km[j])
        print(f"  {n_pairs:>8} {err(S1):>10.3f} {err(S2):>13.3f} {err(S3):>10.3f} {err(S4):>13.3f}")
    print("  → 线性累加在 n > d 时彻底失效；衰减能缓解但会遗忘早期记忆；")
    print("  → Δ rule 主动擦除 k 方向的旧内容，让有限秩反复「复用」，容量显著提升")
    print("  → 注意真实的权衡：α<1 的遗忘会牺牲一部分远期记忆容量，")
    print("     换来处理超长序列时的稳定性。Gated DeltaNet 让 α_t、β_t 都由数据决定，"
          "由模型自己学这个权衡。")


# ============================================================ 5. 复杂度
def demo_complexity():
    section("5. 三种实现的复杂度与并行度")
    print("  " + f"{'实现':<22}{'串行步数':>10}{'每步工作量':>16}{'总 FLOPs':>16}")
    rows = [
        ("逐 token 循环", "O(N)", "O(d²)", "O(N d²)"),
        ("chunkwise(C)", "O(N/C)", "O(C²d + C d²)", "O(NCd + N d²)"),
        ("纯二次(不可行)", "O(1)", "—", "O(N²d)"),
    ]
    for r in rows:
        print(f"  {r[0]:<22}{r[1]:>10}{r[2]:>16}{r[3]:>16}")
    print("  chunkwise 里的关键并行化：C×C 三角系统用 solve_triangular 一次解出，")
    print("  不再有 token 级串行依赖 —— 这是 DeltaNet / Mamba2 / RWKV 能训练的共同套路。")

    B, N, d, chunk = 8, 512, 64, 64
    Q, K, V = torch.randn(B, N, d), torch.randn(B, N, d), torch.randn(B, N, d)
    K = K / K.norm(dim=-1, keepdim=True)
    beta = torch.sigmoid(torch.randn(B, N)) * 0.9 + 0.05
    alpha = torch.sigmoid(torch.randn(B, N) * 0.3) * 0.2 + 0.8
    t_naive = timer(lambda: gdn_naive(Q, K, V, beta, alpha), warmup=0, repeat=1)
    t_chunk = timer(lambda: gdn_chunkwise(Q, K, V, beta, alpha, chunk), warmup=1, repeat=2)
    print(f"\n  实测（CPU, B={B}, N={N}, d={d}, chunk={chunk}）: "
          f"naive {t_naive:.0f} ms  chunkwise {t_chunk:.0f} ms  加速 {t_naive/t_chunk:.1f}x")
    print("  （CPU 上 Python 循环开销被放大；GPU 上差距主要来自能否用 tensor core）")


# ============================================================ 6. 对比 Mamba2
def demo_vs_mamba2():
    section("6. 矩阵视角：Gated DeltaNet / Mamba2 / 线性注意力 是一家人")
    print("  统一写成 O = M · V，其中 M 是 N×N 下三角的「数据依赖混合矩阵」：")
    print("""
    线性注意力    : M[i,j] = q_iᵀ k_j
    带衰减( RetNet): M[i,j] = λ^{i-j} · q_iᵀ k_j
    Mamba2 / SSD  : M[i,j] = q_iᵀ (Π_{l=j+1}^{i} α_l) k_j      ← 标量衰减，A_t = α_t·I（对角）
    DeltaNet      : M[i,j] = q_iᵀ (Π_{l=j+1}^{i} (I-β_l k_l k_lᵀ)) k_j   ← 矩阵值转移，秩1修正
    Gated DeltaNet: 上面再叠加 α_t 的逐元素门控
    """)
    d = 8
    k = torch.randn(d); k = k / k.norm()
    beta = 0.7; alpha = 0.9
    T_delta = alpha * (torch.eye(d) - beta * torch.outer(k, k))
    T_mamba = alpha * torch.eye(d)
    print(f"  一步转移矩阵的谱半径对比（α={alpha}, β={beta}, ‖k‖=1）:")
    print(f"    Mamba2 型 A_t = α·I        : 全部特征值 = {alpha:.2f}（各方向等比遗忘）")
    ev = torch.linalg.eigvalsh(T_delta).sort().values
    print(f"    DeltaNet A_t = α(I-βkkᵀ)   : 特征值 = {[f'{e:.2f}' for e in ev.tolist()]}")
    print("    → DeltaNet 只在一个方向多衰减 (1-β) 倍，是【各向异性】的遗忘；")
    print("      Mamba2 是【各向同性】的遗忘。前者能精确擦掉某个 key 的记忆，后者只能整体变淡。")
    print("  → 代价：DeltaNet 的转移矩阵不再是标量×I，无法像 SSD 那样直接用对角矩阵乘法，")
    print("     必须用 WY 表示 + 三角求解（本章 ②③），这就是它实现更复杂的原因。")


def main():
    title("Gated DeltaNet：Δ rule · Householder · WY 表示 · chunkwise", ch="07")
    demo_householder()
    demo_wy()
    demo_correctness()
    demo_erase_power()
    demo_complexity()
    demo_vs_mamba2()
    print("\nCH07 完成。要点：I-βkkᵀ 是沿 k 方向的各向异性擦除；WY 表示 + 三角求解把串行递推变成可并行矩阵运算。\n")


if __name__ == "__main__":
    main()
