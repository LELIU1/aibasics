"""
CH03 RoPE：旋转矩阵、正交群与相对位置编码

数学主线
  RoPE 的本质是把位置 m 映射为旋转群的一个元素：ρ(m) = R(mθ_0) ⊕ ... ⊕ R(mθ_{d/2-1})
  它是一个【分块对角正交矩阵】，且满足群同态 ρ(m)^T ρ(n) = ρ(n-m)
  → 于是 <ρ(m)q, ρ(n)k> = q^T ρ(n-m) k，注意力分数只依赖相对距离。

学习目标
  1. 2D 旋转矩阵的性质：正交、det=1、保范数、保内积
  2. 分块对角化：d 维空间 = d/2 个独立的旋转平面
  3. 两种内存布局（交错 vs 半分）通过置换矩阵共轭关联
  4. 复数形式：半分布局天然就是复平面上的相位旋转
  5. 频率基底与长上下文外推：NTK-aware scaling / YaRN
"""
import torch
from _utils import check, title, section

torch.manual_seed(0)


# ---------------------------------------------------------------- 1. 2D 旋转
def demo_rotation2d():
    section("1. 二维旋转矩阵 R(θ) 的代数性质")
    def R2(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    for deg in [0, 30, 90, 180, 270]:
        th = torch.tensor(deg * 3.141592653589793 / 180)
        R = R2(th)
        I = torch.eye(2)
        ortho = (R.T @ R - I).abs().max().item()
        det = torch.det(R).item()
        v = torch.tensor([1.0, 0.0])
        print(f"  θ={deg:>4}°  R^T R - I = {ortho:.2e}   det = {det:+.6f}   "
              f"|Rv| = {torch.norm(R @ v):.6f}   Rv = {R @ v @ torch.tensor([1.,1.]):+.3f}")
    print("  正交性 R^T R = I ⇒ 保范数 ‖Rv‖=‖v‖、保内积 ⟨Ru,Rv⟩=⟨u,v⟩（旋转不拉伸、不剪切）")
    print("  det=+1 ⇒ 属于特殊正交群 SO(2)，纯旋转不含镜像")

    # 群性质：R(a)R(b) = R(a+b)，这是 RoPE 相对性的根本来源
    a, b = torch.tensor(0.7), torch.tensor(-0.3)
    check("R(a)R(b) == R(a+b)（单参数子群）", R2(a) @ R2(b), R2(a + b))
    check("R(a)^T == R(-a)", R2(a).T, R2(-a))
    check("R(a)^T R(b) == R(b-a)（相对性恒等式）", R2(a).T @ R2(b), R2(b - a))


# ------------------------------------------------------- 2. 分块对角 + 布局
def freqs(base, d, device=None):
    """θ_i = base^{-2i/d},  i = 0..d/2-1"""
    i = torch.arange(0, d, 2, dtype=torch.float32)
    return 1.0 / (base ** (i / d))


def build_block_diag_R(m, d, base=10000.0, layout="interleaved"):
    """构造完整的 d×d 分块对角旋转矩阵（教学用，实践中绝不这么干）"""
    th = freqs(base, d) * m
    R = torch.eye(d)
    for j, t in enumerate(th):
        c, s = torch.cos(t), torch.sin(t)
        if layout == "interleaved":
            p, q_ = 2 * j, 2 * j + 1
        else:                                   # half-split
            p, q_ = j, j + d // 2
        blk = torch.tensor([[c, -s], [s, c]])
        R[p, p], R[p, q_], R[q_, p], R[q_, q_] = blk[0, 0], blk[0, 1], blk[1, 0], blk[1, 1]
    return R


def rotate_half(x):
    """半分布局：把后半取负搬到前面 —— 一次 cat 就完成 d/2 个平面的旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: (..., d) 半分布局；cos/sin: (..., d)"""
    return x * cos + rotate_half(x) * sin


def demo_block_diag():
    section("2. 分块对角化与两种内存布局")
    d, base = 8, 10000.0
    m = 3
    q = torch.randn(d)

    R_int = build_block_diag_R(m, d, base, "interleaved")
    R_half = build_block_diag_R(m, d, base, "half")
    check("R 是正交矩阵（交错）", R_int.T @ R_int, torch.eye(d))
    check("R 是正交矩阵（半分）", R_half.T @ R_half, torch.eye(d))

    # 置换矩阵：把半分布局换成交错布局
    perm = torch.empty(d, dtype=torch.long)
    perm[0::2] = torch.arange(d // 2)
    perm[1::2] = torch.arange(d // 2) + d // 2
    P = torch.zeros(d, d); P[torch.arange(d), perm] = 1.0   # (P x)[j] = x[perm[j]]
    check("半分/交错布局经置换共轭关联 R_half = P^T R_int P",
          R_half, P.T @ R_int @ P)

    # 高效实现（逐元素）vs 矩阵乘法
    th = freqs(base, d) * m
    cos = torch.cat([th.cos(), th.cos()])       # 半分布局：前后半共用同一组 cos/sin
    sin = torch.cat([th.sin(), th.sin()])
    check("apply_rope(逐元素) == 分块对角矩阵乘法", apply_rope(q, cos, sin), R_half @ q)

    print("  分块对角矩阵视角：ℝ^d = ⊕_{i} ℝ^2 (d/2 个旋转平面)，第 i 个平面以角速度 θ_i 转")
    print(f"  θ_i = base^(-2i/d): {[f'{t:.5f}' for t in freqs(base, d).tolist()]}")
    print("  → i 小 = 高频（转得快，编码近距离）；i 大 = 低频（转得慢，编码远距离）")
    print("  实际实现只做 2 次逐元素乘加，绝不 materialize d×d 矩阵")


# ------------------------------------------------------------ 3. 相对性验证
def demo_relativity():
    section("3. 相对位置：RoPE 的灵魂恒等式")
    d, base, N = 64, 10000.0, 512
    q = torch.randn(d); k = torch.randn(d)
    th = freqs(base, d)
    cos = torch.cat([th.cos(), th.cos()]); sin = torch.cat([th.sin(), th.sin()])

    def rope(x, m):
        t = th * m
        c = torch.cat([t.cos(), t.cos()]); s = torch.cat([t.sin(), t.sin()])
        return apply_rope(x, c, s)

    ok = True
    for m, n in [(0, 7), (13, 200), (100, 100), (512, 63)]:
        lhs = rope(q, m) @ rope(k, n)                 # ⟨ρ(m)q, ρ(n)k⟩
        rhs = q @ rope(k, n - m)                      # q^T ρ(n-m) k
        err = (lhs - rhs).abs().item()
        ok &= err < 1e-4
        print(f"  m={m:>4}, n={n:>4}: ⟨ρ(m)q,ρ(n)k⟩={lhs:+.6f}  q^Tρ(n-m)k={rhs:+.6f}  Δ={err:.2e}")
    print(f"  → {'恒等式成立' if ok else '失败'}：绝对位置 m,n 的内积只通过 n-m 依赖它们")
    print("  这就是 RoPE 外推性的来源，也是它比加性位置编码强的地方")

    # 直觉：随机 q,k 的内积分布本身与 Δ 无关，真正衰减的是【相关性】
    print("\n  相对位置相关性 Corr(⟨q,k⟩, ⟨q,ρ(Δ)k⟩) = tr(ρ(Δ))/d = (2/d)·Σ_i cos(Δθ_i):")
    for delta in [0, 1, 8, 64, 256, 1024, 4096, 16384]:
        analytic = (2.0 / d) * torch.cos(th * delta).sum().item()
        emp = []
        for _ in range(400):
            a, b_ = torch.randn(d), torch.randn(d)
            emp.append((a @ b_) * (a @ rope(b_, delta)))
        emp = torch.tensor(emp).mean().item()
        print(f"    Δ={delta:>6}: 解析 = {analytic:+.4f}   经验 E[⟨q,k⟩⟨q,ρ(Δ)k⟩]/‖q‖²‖k‖² = {emp/d:+.4f}")
    print("  → 相关性随距离衰减但有振荡（低频维度周期长），这就是 RoPE 的局部性先验")


# ---------------------------------------------------------------- 4. 复数
def demo_complex():
    section("4. 复数视角：旋转 = 相位乘法")
    d = 8
    m = 5
    base = 10000.0
    th = freqs(base, d) * m
    x = torch.randn(d)

    # 半分布局下：前半 = 实部，后半 = 虚部
    z = torch.complex(x[:d // 2], x[d // 2:])
    phase = torch.polar(torch.ones_like(th), th)      # e^{i m θ}
    zr = z * phase
    out_complex = torch.cat([zr.real, zr.imag])

    cos = torch.cat([th.cos(), th.cos()]); sin = torch.cat([th.sin(), th.sin()])
    check("复数乘 e^{imθ} == apply_rope", out_complex, apply_rope(x, cos, sin))

    # 相对性在复数下更显然：(q e^{imθ}) conj(k e^{inθ}) = q conj(k) e^{i(m-n)θ}
    q = torch.randn(d); k = torch.randn(d)
    zq = torch.complex(q[:d // 2], q[d // 2:]) * torch.polar(torch.ones_like(th), th)
    zk = torch.complex(k[:d // 2], k[d // 2:]) * torch.polar(torch.ones_like(th), th)
    print(f"  内积 = Re[ (q e^{{imθ}}) · conj(k e^{{inθ}}) ] = Re[ q conj(k) e^{{i(m-n)θ}} ]")
    print(f"  复数实现在 PyTorch 里可直接用 torch.view_as_complex，比 cat 少一次搬运")


# ------------------------------------------------------------ 5. 外推 / NTK
def demo_scaling():
    section("5. 长上下文外推：NTK-aware 与 YaRN")
    d = 64
    L_train, L_test = 4096, 32768
    s = L_test / L_train

    # 原始基底
    th = freqs(10000.0, d)
    # NTK-aware：放大 base（等价缩小所有 θ）→ 低频转得更慢，容纳更长距离
    base_ntk = 10000.0 * s ** (d / (d - 2))
    th_ntk = freqs(base_ntk, d)

    print(f"  外推倍数 s = {s:.0f}x   缩放后 base = {base_ntk:.0f}")
    print(f"  {'i':>3} {'θ_orig':>12} {'θ_ntk':>12} {'周期orig':>12} {'周期ntk':>12}")
    for i in [0, 1, 8, 16, 31]:
        p_orig = 2 * 3.141592653589793 / th[i].item()
        p_ntk = 2 * 3.141592653589793 / th_ntk[i].item()
        print(f"  {i:>3} {th[i].item():>12.6f} {th_ntk[i].item():>12.6f} {p_orig:>12.1f} {p_ntk:>12.1f}")

    # YaRN：分段 ramp —— 高频（短周期）不插值，低频（长周期）做线性插值
    def yarn_correction(th_i, L_train, L_test, alpha=1.0, beta=32.0):
        """YaRN 的 ramp 函数 γ(r)：r = L_train/λ，波长越短(r 越大)越不插值"""
        lam = 2 * 3.141592653589793 / th_i                     # 波长
        r = L_train / lam                                      # 训练长度内跑了几个周期
        gamma = torch.clamp((r - alpha) / (beta - alpha), 0.0, 1.0)
        return (1 - gamma) * (L_train / L_test) + gamma        # γ=1 不动，γ=0 纯插值 1/s

    lam = 2 * 3.141592653589793 / th
    corr = yarn_correction(th, L_train, L_test)
    print("\n  YaRN 插值系数（低频→接近 1/s 纯插值；高频→接近 1 不插值）:")
    for i in [0, 4, 16, 24, 31]:
        print(f"    i={i:>2}  波长={lam[i]:>12.1f}  系数={corr[i]:.4f}")
    print("  统一公式：θ'_i = θ_i · correction_i，其中 correction 是波长的分段线性函数")
    print("  直觉：高频维度在训练中已见过全部相位（周期短），不该改；低频维度没跑完一个周期，需要插值")


# ------------------------------------------------------------ 6. 梯度/一般化
def demo_grad_and_general():
    section("6. 反向传播与一般化视角")
    d, base, m = 16, 10000.0, 7
    th = freqs(base, d) * m
    cos = torch.cat([th.cos(), th.cos()]); sin = torch.cat([th.sin(), th.sin()])

    x = torch.randn(4, 8, d, requires_grad=True)
    grad_out = torch.randn(4, 8, d)
    y = apply_rope(x, cos, sin)
    y.backward(grad_out)
    manual = apply_rope(grad_out, cos, -sin)     # R(-m) = R(m)^T
    check("∂L/∂x == ρ(m)^T (∂L/∂y)（正交变换的 JVP 用转置）", x.grad, manual)

    print("  → 正交矩阵的反向就是它的转置，无需额外存储，这也是 RoPE 几乎零显存开销的原因")

    # 一般化：任何满足群同态 ρ(m)^Tρ(n)=ρ(n-m) 的正交族都能做相对位置编码
    print("\n  一般化视角：RoPE = 群同态 ρ: (ℤ,+) → SO(d)")
    print("    ρ(m) = ⊕_i R(m θ_i) 是 d/2 个 SO(2) 表示的直和")
    print("    若改用非正交矩阵（如可学习的一般矩阵），就不再保内积 → 相对性丢失")
    A = torch.randn(d, d) * 0.1
    q0, k0 = torch.randn(d), torch.randn(d)
    lhs = (A @ q0) @ (A @ k0)
    rhs = q0 @ (A @ k0)
    print(f"    对照：随机非正交矩阵 ⟨Aq,Ak⟩={lhs:+.4f} 与 ⟨q,Ak⟩={rhs:+.4f} 不等 → 相对性破坏")


def main():
    title("RoPE：旋转矩阵 · 正交群 · 相对位置", ch="03")
    demo_rotation2d()
    demo_block_diag()
    demo_relativity()
    demo_complex()
    demo_scaling()
    demo_grad_and_general()
    print("\nCH03 完成。要点：RoPE = 位置到 SO(d) 的群同态；实现上只是 2 次逐元素乘加；外推靠改频率基底。\n")


if __name__ == "__main__":
    main()
