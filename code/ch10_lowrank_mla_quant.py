"""
CH10 低秩与压缩：SVD、LoRA、MLA 的矩阵吸收与权重量化

数学主线
  低秩近似的核心定理（Eckart–Young）：
      min_{rank(B) ≤ r} ‖A - B‖_F  =  ‖A - U_r Σ_r V_rᵀ‖_F  =  sqrt(Σ_{i>r} σ_i²)
  也就是说【截断 SVD 是所有秩 r 矩阵里最好的近似】，且误差恰好是尾部奇异值的平方和。

学习目标
  1. Eckart–Young 定理：截断 SVD 的最优性（并验证没有更好的随机低秩矩阵）
  2. LoRA：ΔW = BA 的秩选择，以及"微调更新是否真的低秩"
  3. MLA：用低秩 latent 压缩 KV Cache，并通过【矩阵吸收】避免把它展开回来
  4. 权重量化与 KV Cache 量化的误差账
"""
import torch
import torch.nn.functional as F
from _utils import check, title, section

torch.manual_seed(0)


# ------------------------------------------------------------ 1. SVD
def demo_svd():
    section("1. Eckart–Young：截断 SVD 是最优低秩近似")
    torch.manual_seed(2)
    m, n = 64, 64
    # 构造一个谱衰减很快的矩阵（真实权重/激活常近似如此）
    A = torch.randn(m, n) @ torch.diag(torch.tensor([1.0 / (i + 1) ** 1.2 for i in range(n)]))
    U, S, Vh = torch.linalg.svd(A)
    print("  前 12 个奇异值:", " ".join(f"{s:.3f}" for s in S[:12].tolist()))

    for r in [1, 2, 4, 8, 16, 32]:
        Ar = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
        err_svd = (A - Ar).norm() / A.norm()
        tail = (S[r:] ** 2).sum().sqrt().item() / A.norm().item()   # 理论误差下界
        # 随机秩 r 矩阵做对照（最优性检验）
        Br = torch.randn(m, r) @ torch.randn(r, n)
        Br = Br * ((A @ Br.T).trace() / (Br @ Br.T).trace())        # 给它最优缩放
        err_rand = (A - Br).norm() / A.norm()
        print(f"  r={r:>2}: 截断SVD 相对误差 {err_svd.item()*100:7.3f}%  "
              f"(理论下界 {tail*100:7.3f}%)   随机秩{r}矩阵 {err_rand.item()*100:7.2f}%")
    check("截断 SVD 误差 == 尾部奇异值平方和（Eckart–Young）",
          (A - U[:, :8] @ torch.diag(S[:8]) @ Vh[:8]).norm(),
          (S[8:] ** 2).sum().sqrt(), atol=1e-3)
    print("  → 谱衰减越快，低秩压缩越有效。这也是 LoRA / MLA / 权重压缩成立的前提")


# ------------------------------------------------------------ 2. LoRA
def demo_lora():
    section("2. LoRA：ΔW = BA 的秩要多大")
    d_in, d_out = 512, 512
    # 模拟一次"真实"微调：构造一个谱衰减的更新矩阵
    torch.manual_seed(5)
    W0 = torch.randn(d_out, d_in) * 0.02
    U, S, Vh = torch.linalg.svd(torch.randn(d_out, d_in))
    S_true = torch.exp(-torch.arange(d_in).float() / 6.0)      # 谱指数衰减（近似低秩）
    dW = U @ torch.diag(S_true) @ Vh * 0.05
    print(f"  ΔW 的谱：前 6 个奇异值 " + " ".join(f"{s:.4f}" for s in (torch.linalg.svdvals(dW)[:6]).tolist()))

    x = torch.randn(256, d_in)
    ref = x @ dW.T                                    # 只看"增量"对输出的影响
    print(f"  {'秩 r':>6} {'‖ΔW-ΔW_r‖_F/‖ΔW‖_F':>22} {'增量输出相对误差':>18} {'参数量占比':>12}")
    for r in [1, 2, 4, 8, 16, 64, 128]:
        dWr = U[:, :r] @ torch.diag(S_true[:r]) @ Vh[:r] * 0.05
        e_w = ((dW - dWr).norm() / dW.norm()).item() * 100
        e_y = ((x @ dWr.T - ref).norm() / ref.norm()).item() * 100
        params = r * (d_in + d_out) / (d_in * d_out) * 100
        print(f"  {r:>6} {e_w:>21.3f}% {e_y:>17.3f}% {params:>11.2f}%")
    print("  → 秩 8~16 就能复现绝大部分更新（因为 ΔW 的谱衰减快）")
    print("  → 但注意：LoRA 的 B、A 是【学出来的】，不是对已知 ΔW 做 SVD；")
    print("     实践中大 rank + 高 lr 或小 rank + 稳定训练各有适用场景")


# ------------------------------------------------------------ 3. MLA
class MLA(torch.nn.Module):
    """Multi-head Latent Attention（DeepSeek-V2/V3 的核心）

    训练/展开版（便于理解）:
        c_KV = x W_DKV            (d_model → d_c)    低秩 KV 压缩
        c_Q  = x W_DQ             (d_model → d_cq)
        K    = c_KV W_UK          (d_c  → H·d_h)     上采样
        V    = c_KV W_UV
        Q    = c_Q  W_UQ
    推理/吸收版（不展开 K、V，KV cache 只存 c_KV 与 k_r）:
        score = c_Qᵀ (W_UQᵀ W_UK) c_KV        ← W_UK 被吸收进 W_UQ
        out   = Σ_h (A_h c_KV) (W_UV_h W_O_h) ← W_UV 被吸收进 W_O
    """

    def __init__(self, d_model=256, H=8, d_h=32, d_c=64, d_cq=96, d_r=32):
        super().__init__()
        self.d_model, self.H, self.d_h, self.d_c, self.d_r = d_model, H, d_h, d_c, d_r
        self.W_DKV = torch.nn.Parameter(torch.randn(d_model, d_c) * 0.02)
        self.W_DQ = torch.nn.Parameter(torch.randn(d_model, d_cq) * 0.02)
        self.W_UQ = torch.nn.Parameter(torch.randn(d_cq, H * d_h) * 0.02)
        self.W_UK = torch.nn.Parameter(torch.randn(d_c, H * d_h) * 0.02)
        self.W_UV = torch.nn.Parameter(torch.randn(d_c, H * d_h) * 0.02)
        self.W_O = torch.nn.Parameter(torch.randn(H * d_h, d_model) * 0.02)
        # 解耦 RoPE 部分（不进 latent，避免位置信息被低秩压缩破坏）
        self.W_QR = torch.nn.Parameter(torch.randn(d_model, H * d_r) * 0.02)
        self.W_KR = torch.nn.Parameter(torch.randn(d_model, d_r) * 0.02)
        self.d_cq = d_cq

    def forward(self, x, rope_cos=None, rope_sin=None, absorbed=True):
        B, N, _ = x.shape
        H, d_h, d_r = self.H, self.d_h, self.d_r
        c_kv = x @ self.W_DKV                                   # (B,N,d_c)
        c_q = x @ self.W_DQ                                     # (B,N,d_cq)
        q_r = (x @ self.W_QR).view(B, N, H, d_r)                # 解耦 RoPE 的 q
        k_r = x @ self.W_KR                                     # (B,N,d_r)

        if rope_cos is not None:                                # 对解耦部分施加 RoPE
            def rot_half(t):
                t1, t2 = t.chunk(2, -1)
                return torch.cat((-t2, t1), -1)
            q_r = q_r * rope_cos[None, :, None, :] + rot_half(q_r) * rope_sin[None, :, None, :]
            k_r = k_r * rope_cos[None, :, :] + rot_half(k_r) * rope_sin[None, :, :]

        if not absorbed:                                        # 展开版（教学对照）
            q = (c_q @ self.W_UQ).view(B, N, H, d_h)
            k = (c_kv @ self.W_UK).view(B, N, H, d_h)
            v = (c_kv @ self.W_UV).view(B, N, H, d_h)
            s = torch.einsum("bnhd,bmhd->bhnm", q, k) * d_h ** -0.5
            s = s + torch.einsum("bnhd,bmd->bhnm", q_r, k_r) * (d_h + d_r) ** -0.5
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
            a = torch.softmax(s, -1)
            o = torch.einsum("bhnm,bmhd->bnhd", a, v).reshape(B, N, H * d_h)
            return o @ self.W_O

        # ---- 吸收版 ----
        # ① W_UK 吸收进 W_UQ：score_h = c_qᵀ (W_UQ_hᵀ W_UK_h) c_kv
        W_UQh = self.W_UQ.view(self.d_cq, H, d_h)
        W_UKh = self.W_UK.view(self.d_c, H, d_h)
        W_absorb = torch.einsum("chd,C hd->hcC", W_UQh, W_UKh)   # (H, d_cq, d_c)
        s = torch.einsum("bnc,hcC,bmC->bhnm", c_q, W_absorb, c_kv) * d_h ** -0.5
        s = s + torch.einsum("bnhd,bmd->bhnm", q_r, k_r) * (d_h + d_r) ** -0.5
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float("-inf"))
        a = torch.softmax(s, -1)                                 # (B,H,N,N)
        # ② W_UV 吸收进 W_O：out = Σ_h (A_h c_kv) (W_UV_h W_O_h)
        Z = torch.einsum("chd,hdD->hcD", self.W_UV.view(self.d_c, H, d_h),
                         self.W_O.view(H, d_h, self.d_model))    # (H, d_c, d_model)
        ac = torch.einsum("bhnm,bmc->bhnc", a, c_kv)             # (B,H,N,d_c)
        return torch.einsum("bhnc,hcD->bnD", ac, Z)


def demo_mla():
    section("3. MLA：低秩 KV 压缩 + 矩阵吸收")
    B, N, d_model = 2, 64, 256
    H, d_h, d_c, d_r = 8, 32, 64, 32
    x = torch.randn(B, N, d_model)
    mla = MLA(d_model=d_model, H=H, d_h=d_h, d_c=d_c, d_cq=96, d_r=d_r)

    # 简易 RoPE（半分布局）
    th = 1.0 / (10000 ** (torch.arange(0, d_r, 2).float() / d_r))
    pos = torch.arange(N).float()
    cos = torch.cat([torch.cos(pos[:, None] * th), torch.cos(pos[:, None] * th)], -1)
    sin = torch.cat([torch.sin(pos[:, None] * th), torch.sin(pos[:, None] * th)], -1)

    o_expanded = mla(x, cos, sin, absorbed=False)
    o_absorbed = mla(x, cos, sin, absorbed=True)
    check("MLA 展开版 == 吸收版（矩阵吸收不改变结果）", o_absorbed, o_expanded, atol=1e-4)

    print(f"  配置: H={H} 头, d_h={d_h}, latent d_c={d_c}, RoPE 解耦 d_r={d_r}")
    print("\n  KV Cache 每个 token 需要存的元素数:")
    mha = 2 * H * d_h
    mqa = 2 * 1 * d_h
    gqa4 = 2 * (H // 4) * d_h
    mla_c = d_c + d_r
    for name, v_ in [("MHA (8 头)", mha), ("GQA (2 组)", gqa4), ("MQA (1 头)", mqa), ("MLA", mla_c)]:
        print(f"    {name:<14}: {v_:>6d} 维/token   相对 MHA 省 {mha/v_:>5.1f}x")
    print("\n  真实规模对照（DeepSeek-V2 级别: H=128, d_h=128, d_c=512, d_r=64）:")
    mha_big, mla_big = 2 * 128 * 128, 512 + 64
    print(f"    MHA: {mha_big} 维/token → MLA: {mla_big} 维/token，省 {mha_big/mla_big:.1f}x"
          f"（且 MLA 的表达力还更强，因为 latent 是各头共享的高秩压缩）")
    print("  → MLA 把 KV 压到 latent，推理时靠【矩阵吸收】直接用 latent 计算，不需要展开回 H·d_h")
    print("  → 吸收的两个恒等式（都是结合律）：")
    print("     score: (W_UQ c_q)ᵀ(W_UK c_kv) = c_qᵀ (W_UQᵀ W_UK) c_kv      ← 预计算 (d_cq × d_c)")
    print("     out  : Σ_h W_O_h A_h (W_UV_h c_kv) = Σ_h (A_h c_kv)(W_UV_h W_O_h) ← 结合律换序")
    print("  → 代价：RoPE 不能进 latent（位置信息会被低秩抹掉），所以要额外解耦 d_r 维")


# ------------------------------------------------------------ 4. 量化
def demo_quant_weights():
    section("4. 权重与 KV Cache 量化")
    torch.manual_seed(0)
    d_in, d_out = 1024, 1024
    W = torch.randn(d_out, d_in) * 0.02
    x = torch.randn(512, d_in)
    ref = x @ W.T

    def q_int(t, bits, group):
        """分组仿射量化：每 group 个元素共享一个 scale"""
        qmax = 2 ** (bits - 1) - 1
        t2 = t.reshape(-1, group)
        s = t2.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        return (torch.round(t2 / s).clamp(-qmax - 1, qmax) * s).reshape(t.shape)

    print("  权重 W 量化后 y = xWᵀ 的相对误差:")
    print(f"    {'bits':>5} {'group=∞(per-tensor)':>22} {'group=128':>14} {'group=32':>14}")
    for bits in [8, 4, 3, 2]:
        row = [q_int(W, bits, g) for g in [W.numel(), 128, 32]]
        errs = [((x @ Wi.T - ref).norm() / ref.norm()).item() * 100 for Wi in row]
        print(f"    {bits:>5} {errs[0]:>21.3f}% {errs[1]:>13.3f}% {errs[2]:>13.3f}%")
    print("  → 均匀量化的理论误差 ≈ (step/√12)/σ，step = 2·amax/(2^b-1)：")
    print("     对高斯权重 INT8 ≈ 1%，INT4 ≈ 10% —— 光靠均匀量化 INT4 是不够的")
    print("  → 真实模型能跑 INT4/INT2，靠的是把目标从 ‖W-Ŵ‖ 换成【加权】目标：")
    print("     GPTQ/AWQ:  min ‖(W-Ŵ)X‖²_F  ≈  min Σ_j H_jj (W_j - Ŵ_j)²   (H = XᵀX 的 Hessian)")
    print("     即：重要的权重（对应激活方差大的通道）量化得更精细，不重要的可以更粗")

    print("\n  KV Cache 量化的特殊性（DeepSeek 的做法）:")
    print("    · K 沿【序列维】分布不均（有 massive activation），适合 per-token 量化")
    print("    · V 相对均匀，可以做更激进的压缩")
    print("    · 更激进的方案：把 KV 投影到低秩 latent 后再量化（MLA + FP8 叠加）")


def main():
    title("低秩与压缩：SVD · LoRA · MLA · 量化", ch="10")
    demo_svd()
    demo_lora()
    demo_mla()
    demo_quant_weights()
    print("\nCH10 完成。要点：截断 SVD 是最优低秩近似；MLA 用低秩 latent + 矩阵吸收把 KV Cache 压掉一个数量级。\n")


if __name__ == "__main__":
    main()
