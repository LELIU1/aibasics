"""
CH09 DeepEP 与专家并行：All-to-All 的置换语义、通信量下界与 FP8 量化

数学主线
  专家并行（EP）把一个 MoE 层拆到 W 张卡上，token 必须"飞"到它选中的专家所在的卡：

      dispatch :  X_local ──按目标专家分组──▶ All-to-All ──▶ X_expert  （矩阵：分块置换 P）
      计算     :  X_expert ──GroupGEMM(CH08)──▶ Y_expert
      combine :  Y_expert ──All-to-All 回来──▶ Σ_k g_k · Y_k        （矩阵：Pᵀ 加权求和）

  所以 EP 的通信本质是一个【跨设备的块置换矩阵】，而 DeepEP 是它的高性能实现。

学习目标
  1. 用张量模拟 All-to-All，验证它真的等价于"按路由重排"
  2. 通信量的下界：为什么是 T·topk·d 而不是 T·d
  3. FP8 量化的数学：仿射缩放 + 有限尾数，以及 block-wise scaling 为什么必要
  4. 计算/通信比决定能否重叠 —— DeepEP 双流水的原理
"""
import torch
import torch.nn.functional as F
from _utils import check, timer, title, section

torch.manual_seed(0)


# ------------------------------------------------- 1. 模拟 All-to-All
def simulate_alltoall(x_local, dst_expert, E, W_dev):
    """模拟 EP 的 dispatch：每个设备持有一批 token，按目标专家分组后"发送"。
    真实实现用 NCCL/RDMA All-to-All；这里用索引操作表达同样的语义。
    x_local: (W_dev, T_loc, d)   dst_expert: (W_dev, T_loc, topk)
    返回: 每个专家收到的 token 拼接 (E, *, d) 以及各专家的 token 数
    """
    Wd, T_loc, d = x_local.shape
    topk = dst_expert.shape[-1]
    # 每个 (token, 专家选择) 对都要发一份 ⇒ token 复制 topk 份
    flat_x = x_local.reshape(Wd * T_loc, d).repeat_interleave(topk, dim=0)
    flat_e = dst_expert.reshape(-1)
    order = flat_e.argsort()
    xs = flat_x[order]
    counts = torch.bincount(flat_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return xs, counts, offsets, order


def demo_alltoall():
    section("1. All-to-All = 跨设备的分块置换")
    W_dev, T_loc, d, E, topk = 4, 8, 6, 4, 2
    x = torch.randn(W_dev, T_loc, d)
    torch.manual_seed(1)
    dst = torch.stack([torch.stack([torch.randperm(E)[:topk] for _ in range(T_loc)])
                       for _ in range(W_dev)])                    # (W_dev, T_loc, topk)

    xs, counts, offsets, order = simulate_alltoall(x, dst, E, W_dev)
    # 校验：排序后同一段内的 token 都属于同一个专家
    flat_e = dst.reshape(-1)
    ids_sorted = flat_e[order]
    ok = all(ids_sorted[offsets[e]:offsets[e + 1]].eq(e).all().item() for e in range(E))
    print(f"  各专家收到 token 数: {counts.tolist()}，分组是否正确: {ok}")
    print(f"  总 token（含 topk 复制）: {int(counts.sum())} = W_dev×T_loc×topk = {W_dev*T_loc*topk}")

    # 等价性：直接 gather 与"走一遍 all-to-all 再回来"
    flat_x = x.reshape(W_dev * T_loc, d).repeat_interleave(topk, dim=0)
    experts = [torch.randn(d, d) for _ in range(E)]
    # 直接算（不跨设备）
    direct = torch.stack([flat_x[i] @ experts[int(flat_e[i])] for i in range(flat_x.shape[0])])
    direct = direct.view(W_dev * T_loc, topk, d).sum(1)
    # 走一遍 all-to-all
    ys = torch.zeros_like(xs)
    for e in range(E):
        s, t = offsets[e].item(), offsets[e + 1].item()
        if t > s:
            ys[s:t] = xs[s:t] @ experts[e]
    recv = torch.zeros_like(flat_x)
    recv[order] = ys                                     # combine：按逆置换送回
    combined = recv.view(W_dev * T_loc, topk, d).sum(1)
    check("dispatch→计算→combine == 直接本地计算", combined, direct, atol=1e-4)
    print("  → 数学上：dispatch 是分块置换 P，combine 是 Pᵀ（加权），两者都是正交/广义置换矩阵")
    print("  → DeepEP 做的事：把这个置换用 RDMA/NVLink 高效地实现，并与计算重叠")


def demo_comm_volume():
    section("2. 通信量下界与计算/通信比")
    print("  每个 token 要发往 topk 个专家所在设备 → 每层每 token 的通信量 = topk × d × bytes")
    print(f"  {'配置':<28}{'通信量/token':>14}{'计算量/token':>14}{'算通比':>10}")
    for d, ff, topk, name in [(4096, 14336, 2, "DeepSeek-V3 式 (d=4096,topk=2)"),
                              (4096, 14336, 8, "topk=8 大稀疏"),
                              (2048, 8192, 2, "中等规模 (d=2048)"),
                              (1024, 4096, 2, "小规模 (d=1024)")]:
        comm = topk * d * 2                      # bf16: 2 bytes
        comp = topk * (2 * d * ff * 3) / 1e0     # up/gate/down 三个 GEMM，FLOPs
        ratio = comp / comm
        print(f"  {name:<28}{comm/1024:>11.1f} KB{comp/1e6:>11.1f} MFLOP{ratio/1000:>9.1f} K")
    print("  → 算通比 ≫ 1 时，通信可以被计算掩盖（DeepEP 的双流水/流水线就是这样设计的）")
    print("  → 反例：decode 阶段 batch 小、计算少，通信占比高 → 必须用低延迟 kernel + FP8 压缩")

    print("\n  不同 dtype 下的 dispatch 带宽（T=8192 token, d=4096, topk=2）:")
    T, d, topk = 8192, 4096, 2
    for name, nb in [("bf16", 2), ("fp8 (e4m3)", 1), ("int8", 1), ("int4", 0.5)]:
        gb = T * topk * d * nb / 1e9
        print(f"    {name:<12}: {gb:6.3f} GB / 层   （相对 bf16 省 {2/nb:.1f}x）")


# --------------------------------------------------------- 3. FP8 量化
def fp8_e4m3_emulate(x, scale):
    """模拟 e4m3（4 指数位 + 3 尾数位）的量化：相对步长 2^-3，最大正规数 448。
    x_hat = scale · round_fp8(x / scale)
    """
    q = x / scale
    q = q.clamp(-448.0, 448.0)
    sign = torch.sign(q)
    a = q.abs().clamp(min=1e-30)
    # 正规区：步长 = 2^{e-3}（3 位尾数）；次正规区（|q| < 2^-6）：步长固定 2^-9，低于它直接归零
    e = torch.floor(torch.log2(a))
    step = torch.where(a >= 2.0 ** -6, torch.pow(2.0, e - 3), torch.full_like(a, 2.0 ** -9))
    return sign * torch.round(a / step) * step * scale


def blockwise_amax(x, block=128):
    """按连续 block 个元素分组，每组一个 amax（DeepGEMM/DeepEP 用 128 元素块）"""
    n = x.numel()
    pad = (-n) % block
    xf = torch.cat([x.reshape(-1), torch.zeros(pad, dtype=x.dtype)]) if pad else x.reshape(-1)
    amax = xf.view(-1, block).abs().amax(-1)
    return amax


def int8_quantize(x, scale):
    """均匀（仿射）量化：q = round(x/s)，s = amax/127。相对误差 ∝ 1/|x| —— 小值最吃亏"""
    q = torch.round(x / scale).clamp(-128, 127)
    return q * scale


def demo_quantization():
    section("3. FP8 / INT8 量化：动态范围、尾数截断与 block-wise scaling")
    torch.manual_seed(0)
    n = 65536
    x = torch.randn(n)                                  # 高斯

    def rel_err(xq, xin):
        return ((xq - xin).norm() / xin.norm()).item() * 100

    def s_fp8(t):   return t.abs().amax().item() / 448.0     # amax 映射到 e4m3 最大值 448
    def s_int8(t):  return t.abs().amax().item() / 127.0     # amax 映射到 int8 最大值 127

    print("  (a) 均匀高斯，per-tensor scale —— FP8 正规区的相对步长恒为 2^-3")
    print(f"    fp8 e4m3 : 相对误差 = {rel_err(fp8_e4m3_emulate(x, s_fp8(x)), x):.3f}%")
    print(f"    int8     : 相对误差 = {rel_err(int8_quantize(x, s_int8(x)), x):.3f}%")
    print("    → FP8 的相对误差与幅值无关（浮点自适应步长），INT8 对小值更不友好")

    print("\n  (b) 分布不均匀（真实激活常见：channel 间幅值差几个数量级）")
    x_mix = torch.cat([torch.randn(n // 2), torch.randn(n // 2) * 2e-5])
    print(f"    大幅值段 amax={x_mix[:n//2].abs().amax().item():.3f}，"
          f"小幅值段 amax={x_mix[n//2:].abs().amax().item():.3e}，"
          f"动态范围 {x_mix.abs().amax().item()/x_mix[n//2:].abs().amax().item():.2e}x")
    for tag, fn, s in [("fp8 e4m3", fp8_e4m3_emulate, s_fp8(x_mix)),
                       ("int8    ", int8_quantize, s_int8(x_mix))]:
        xq = fn(x_mix, s)
        print(f"    {tag}: 大值段 {rel_err(xq[:n//2], x_mix[:n//2]):7.3f}%   "
              f"小值段 {rel_err(xq[n//2:], x_mix[n//2:]):8.3f}%   ← 全局 scale 被大值主导")

    print("\n  (c) block-wise scaling（每 block 个元素一个 scale）")
    print(f"    {'block':>6} {'fp8 整体':>9} {'fp8 小值段':>11} {'int8 整体':>10} {'int8 小值段':>11}")
    for block in [65536, 4096, 512, 128, 32]:
        amax = blockwise_amax(x_mix, block)
        sv8 = (amax / 448.0).repeat_interleave(block)[:n]
        svi = (amax / 127.0).repeat_interleave(block)[:n]
        xq8, xqi = fp8_e4m3_emulate(x_mix, sv8), int8_quantize(x_mix, svi)
        nscale = n // block
        print(f"    {block:>6} {rel_err(xq8, x_mix):>8.3f}% {rel_err(xq8[n//2:], x_mix[n//2:]):>10.3f}% "
              f"{rel_err(xqi, x_mix):>9.3f}% {rel_err(xqi[n//2:], x_mix[n//2:]):>10.3f}%"
              f"   (scale 数 {nscale})")
    print("  → block 越小，scale 越贴合局部幅值；但元数据越多、kernel 越复杂")
    print("  → 128 是精度/开销的甜点 —— 这正是 DeepGEMM / DeepEP 采用 per-128-element scaling 的原因")

    print("\n  (d) 量化误差在 GEMM 中不会随 K 累积（量化能用于训练的关键）:")
    d = 256
    print(f"    {'K':>6} {'fp8×fp8 相对误差':>18}")
    for K in [256, 1024, 4096, 16384]:
        a = torch.randn(d, K); b = torch.randn(K, d)
        ref = a @ b
        err = ((fp8_e4m3_emulate(a, s_fp8(a)) @ fp8_e4m3_emulate(b, s_fp8(b)) - ref).norm()
               / ref.norm()).item() * 100
        print(f"    {K:>6} {err:>17.3f}%")
    print("    → 信号 ‖a·b‖ ∝ √K，量化噪声也 ∝ √K，比值恒定 ⇒ 误差不随维度放大")


def demo_overlap():
    section("4. 计算-通信重叠：为什么 DeepEP 要分双流水")
    print("  EP 的一次前向时间 ≈ max(通信, 计算)  if 完美重叠")
    print("  设 T=4096 token, d=4096, ff=14336, topk=2, 单卡带宽 50 GB/s, 算力 300 TFLOP/s(bf16)")
    T, d, ff, topk = 4096, 4096, 14336, 2
    bw, flops_dev = 50e9, 300e12
    for dtype_bytes, tag in [(2, "bf16"), (1, "fp8")]:
        comm_bytes = T * topk * d * dtype_bytes
        t_comm = comm_bytes / bw
        flops = T * topk * 3 * 2 * d * ff            # up/gate/down
        t_comp = flops / flops_dev
        print(f"    {tag}: 通信 {comm_bytes/1e6:7.1f} MB → {t_comm*1e3:6.2f} ms | "
              f"计算 {flops/1e9:8.1f} GFLOP → {t_comp*1e3:6.2f} ms | "
              f"算通比 {t_comp/t_comm:5.2f}")
    print("  → 算通比 > 1 ⇒ 通信可被掩盖；DeepEP 把 dispatch 切成多个 chunk，")
    print("     让 chunk i+1 的通信与 chunk i 的计算并行（双流水 / wave 流水）")


def main():
    title("DeepEP / 专家并行：All-to-All · 通信量 · FP8 量化", ch="09")
    demo_alltoall()
    demo_comm_volume()
    demo_quantization()
    demo_overlap()
    print("\nCH09 完成。要点：EP 通信是跨设备块置换；通信量 ∝ topk·d；FP8 + per-block scaling 是带宽与精度的折中。\n")


if __name__ == "__main__":
    main()
