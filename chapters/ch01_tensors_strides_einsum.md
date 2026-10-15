# CH01 线性代数的「物理层」：张量布局、stride、einsum、算术强度

> 代码：`code/ch01_tensors_strides_einsum.py`
> 前置：PyTorch 基本操作。本章不引入任何模型概念，只讲「同一个数学对象在内存里怎么摆」。

## 1. 为什么要从内存布局开始

大模型里 99% 的性能问题都能归结为一句话：**你在搬运字节，而不是在做浮点运算**。
而"搬运多少字节"这件事，完全由张量的**布局（layout）**决定——数学上完全相同的两个矩阵，
摆法不同，速度可以差几倍。

本章建立三个工具，后面 10 章反复使用：

| 工具 | 回答的问题 |
|---|---|
| `stride` | 一个逻辑坐标 $(i,j,k)$ 映射到哪个物理地址？ |
| `einsum` | 如何用一行符号把任意线性代数运算写清楚？ |
| 算术强度 (arithmetic intensity) | 这个算子是**算得慢**还是**搬得慢**？ |

## 2. stride：布局的数学定义

一个形状为 $(d_0,d_1,\dots,d_{n-1})$ 的张量，其 `stride`=$(s_0,\dots,s_{n-1})$ 表示：

$$\text{addr}(i_0,\dots,i_{n-1}) = \text{offset} + \sum_k i_k \cdot s_k$$

连续性（contiguous）的判据是 $s_k = \prod_{m>k} d_m$。

```python
A = torch.arange(12.).reshape(3, 4)
A.stride()      # (4, 1)    行主序：走一行跳 4 个数
A.T.stride()    # (1, 4)    转置后：走一列跳 4 个数
A.T.data_ptr() == A.data_ptr()   # True —— 共享内存！
```

**关键结论**：

- `transpose` / `permute` / `view`（某些情况）是 **$O(1)$ 的元数据操作**，只改 stride，不搬数据。
- `.contiguous()` 才是 **$O(n)$ 的真正拷贝**（一次读 + 一次写）。
- 很多手写 kernel 假设输入连续；喂给它非连续张量会触发隐式拷贝，白白多一次全量读写。
- BLAS（`torch.mm` / `@`）比较聪明，能直接吃转置视图（因为它把 stride 当成 lda/ldb 参数）。

运行 `ch01` 第 1 节会打印：连续 GEMM 与转置视图 GEMM 的耗时几乎相同，而
`einsum`/自定义 kernel 就没这么好运。

### 布局陷阱速查

| 操作 | 是否拷贝 | 说明 |
|---|---|---|
| `t.transpose(a,b)` / `t.permute(...)` | 否 | 只改 stride |
| `t.view(...)` | 否（若兼容） | 否则抛错，需先 `.contiguous()` |
| `t.reshape(...)` | **可能** | 必要时会悄悄拷贝 |
| `t.contiguous()` | 是 | 显式物化 |
| `t.expand(...)` | 否 | stride=0 的"广播视图"，见 §4 |

## 3. einsum：描述线性代数的通用语言

Einstein 求和约定：重复出现的下标被求和；`->` 右边是输出下标。

| 运算 | einsum 写法 | 形状 |
|---|---|---|
| 矩阵乘 | `ij,jk->ik` | $(M,K)\times(K,N)\to(M,N)$ |
| batch 矩阵乘 | `bij,bjk->bik` | 逐样本独立 |
| 注意力分数 | `bnd,bmd->bnm` | **对 $d$ 收缩，保留 $n,m$** |
| 多头注意力分数 | `bhnd,bhmd->bhnm` | 多一个 head 维 |
| 外积（秩1矩阵） | `i,j->ij` | 线性注意力的核心，见 CH06 |
| 外积累加 | `bnk,bnv->bkv` | **对 $n$ 收缩** → 状态矩阵，CH06 |
| bilinear form | `i,ij,j->` | 标量，二次型 |
| trace | `ii->` | 对角线和 |
| 对角缩放（左） | `i,ij->ij` | 等价于 `diag(a) @ B` |
| 对角缩放（右） | `ij,j->ij` | 等价于 `B @ diag(a)` |

**读 einsum 的三步法**（把它当成"维度记账"）：

1. 输入里出现、输出里消失的下标 → 被**收缩（求和）**，代价出现在 FLOPs 里；
2. 输出里出现的下标 → 被**保留**，其大小乘积 = 输出元素数；
3. FLOPs $= 2 \times \prod(\text{所有出现过的下标大小})$（若每个下标在每个输入中至多出现一次）。

例：`bhnd,bhmd->bhnm` 的 FLOPs $= 2 \cdot B \cdot H \cdot N \cdot D \cdot M$，
注意力里 $N=M$ 所以是 $2BHN^2D$ —— 这就是 $O(N^2)$ 的来源。

> **本书约定**：看到 `->nm` 就是"每对 token 之间算一个量"（二次复杂度）；
> 看到 `->kv` 或 `->bd` 这种不含 $n$ 的输出，就是"状态/汇总"（线性复杂度）。
> 这个区别是 CH05→CH07 全部优化的总纲。

## 4. 广播的代价：隐式的外积

**广播本身是免费的**：插入长度为 1 的维度只是把对应 stride 改成 0，是视图，不搬数据。
但一旦对两个**互补拉伸**的张量做逐元素运算，结果必须被**真实写出来**：

```python
q, k: (B, H, N, D)
q[:, :, :, None, :]    # (B,H,N,1,D)   stride 里 N 维被拉伸（重复 stride）
k[:, :, None, :, :]    # (B,H,1,N,D)
q5 - k5                # (B,H,N,N,D)   ← 真分配：B·H·N²·D 个数
```

结果的 stride 里不再有 0，说明 $N^2$ 个 $(n,m)$ 对的**差向量**每一条都被物化了：

$$ \text{字节数} = B\cdot H\cdot N^2 \cdot D \cdot 4\ \text{(fp32)} $$

> ⚠️ **常见索引错误**：写成 `q[:, :, None, :] - k[:, None, :, :]` 得到的是 $(B,H,H,N,D)$——
> 它在算"第 $h$ 个 q 头 vs 第 $h'$ 个 k 头"的交叉，通常不是你要的东西。
> 想表达"第 $n$ 个 token vs 第 $m$ 个 token"，`None` 必须插在 **N 之后**（`[:, :, :, None, :]`）。

### 数字是怎么算出来的

fp32 下一个元素 4 字节；单头 $B=H=1$，所以元素数就是 $N^2D$：

| $N,d$ | 元素数 $N^2D$ | 广播中间张量 | 分数矩阵 $N^2$ 个标量 | 倍数 |
|---|---|---|---|---|
| 4096, 64 | $4096^2\times64 = 1.07\times10^9$ | $1.07\times10^9\times4\text{B} =$ **4.29 GB** | 67.1 MB | **64×** |
| 8192, 128 | $8192^2\times128 = 8.59\times10^9$ | **34.36 GB** | 268.4 MB | **128×** |
| 32768, 128 | $32768^2\times128 = 1.37\times10^{11}$ | **549.76 GB** | 4295.0 MB | **128×** |

**倍数恰好等于 $d$**，跟序列长度无关：

$$ \frac{N^2 D\cdot 4}{N^2\cdot 4} = D $$

第一行是 64× 只因为那一行的 $d=64$。长序列让两个数字**一起**变大，比值不动。

### 一个重要澄清

标准 softmax attention **根本不会写 `q - k`**（那是欧氏距离注意力 / 相对位置编码的写法）。
它用 `einsum("bhnd,bhmd->bhnm")` 把 $d$ 直接**收缩掉**，峰值只有 $N^2$ 个标量
（$N{=}4096$ 时 = 67.1 MB）。所以：

- **4.29 GB** 是「在 pairwise 运算上滥用逐元素广播」的代价，真实踩坑场景是相对位置编码、距离注意力、手写 pairwise 特征；
- **67.1 MB** 才是标准 attention 的分数矩阵——而 $N{=}32768$ 时它已经涨到 4.3 GB，再乘 $B\cdot H$ 直接爆掉。

**$N^2$ 本身就是内存墙，这才是 CH05 FlashAttention 要解决的对象**（分块降到 $O(N)$）。

| 写法 | 峰值内存 | 说明 |
|---|---|---|
| `q5 - k5` 再逐元素算 | $O(BHN^2D)$ | ❌ 炸内存（pairwise 广播踩坑） |
| `einsum`/`matmul` 收缩 $d$ | $O(BHN^2)$ | ⚠️ 标准 attention，仍是 $O(N^2)$ |
| 分块（chunk）循环 | $O(BH\,b^2)$ | ✅ FlashAttention 的思路 |

实测（`ch01` 第 3 节，$N{=}2048,d{=}32,\text{chunk}{=}256$）：一次性广播峰值 536.9 MB，
分块后 67.1 MB，**省 8×（= $N/\text{chunk}$）且数值完全一致**。
（这里的 67.1 MB 和上面表格里的 67.1 MB 纯属数字巧合，不是同一个量。）

## 5. 算术强度与 Roofline

**算术强度（Arithmetic Intensity, AI）**：

$$ \mathrm{AI} = \frac{\text{FLOPs}}{\text{Bytes moved}} \quad [\text{FLOP/Byte}] $$

把它和硬件的"峰值算力 / 峰值带宽"比值比较，就能判断算子属于哪一类：

⚠️ 先分清三笔账，否则数字永远对不上：

| 量 | 公式里的常数 | 跟 dtype 的关系 |
|---|---|---|
| **FLOPs**（运算次数） | $2MNK$，那个 2 是"一乘一加" | **无关**，纯数学量 |
| **Bytes**（搬运量） | 元素数 $\times$ 字节宽（fp32=4, bf16/fp16=2, fp8/int8=1） | **强相关** |
| **峰值 FLOP/s**（算力） | 硬件电路决定 | **强相关**，必须选实际执行的 dtype |

统一按 **fp32（4 字节/元素）** 记账：

| 算子 | FLOPs | 搬运字节 | AI | 类型 |
|---|---|---|---|---|
| GEMM $(M,N,K)$，大形状 | $2MNK$ | $(MK+KN+MN)\times4$ | 随 $\min(M,N,K)$ 增长，常 > 100 | **计算受限** |
| 逐元素（+、ReLU、乘 mask） | $n$ | $2n$ 个元素 $=8n$ B | $1/8=0.125$ | **带宽受限** |
| softmax | $\approx 3n$ | $8n$ B（融合后可降到 $4n$ B） | $\approx 0.375$ | **带宽受限** |
| LayerNorm / RMSNorm | $\approx 4n$ | $8n$ B | $\approx 0.5$ | **带宽受限** |
| RoPE | $\approx 4n$ | $8n$ B | $\approx 0.5$ | **带宽受限** |

在 `ch01` 实测（CPU）：$2048^3$ GEMM 的 AI ≈ 341，逐元素算子 AI ≈ 0.12（就是那个 $1/8$）—— **差 2800 倍**。

### 5.1 峰值算力和带宽，分别取哪个精度？

这是算 roofline 时最容易搞混的一步：

- **峰值算力取「这个 kernel 实际跑在什么 dtype 上」的峰值。** 用 BF16 做 GEMM 就用 BF16 Tensor Core 的 989.5 TFLOPS，不要拿 FP32 的 66.9 去比。
- **带宽跟精度无关。** HBM 3.35 TB/s 是物理通道，**单位就是 Byte/s**，不存在"fp16 带宽"。低精度省的是**字节数**，不是带宽变宽——所以 FP8 通信时间减半，而不是通道翻倍。
- **分母要数这个实现真的搬了多少字节**，不是"理论最少字节"（不融合时中间结果要写回 HBM 再读出来，见 CH02）。

于是屋顶的**拐点**（machine balance）随 dtype 变（H100 SXM，dense 值）：

| 执行 dtype | 峰值算力 | 带宽 | 拐点 = 峰值/带宽 |
|---|---|---|---|
| FP32（CUDA core） | 66.9 TFLOPS | 3.35 TB/s | **20** FLOP/Byte |
| BF16（Tensor Core） | 989.5 TFLOPS | 3.35 TB/s | **295** |
| FP8（Tensor Core） | 1979 TFLOPS | 3.35 TB/s | **591** |

判断：`AI > 拐点` → 计算受限；`AI < 拐点` → 访存受限。

同一个 $4096^3$ GEMM（FLOPs 恒为 $2\times4096^3=137.4$ GFLOP）在三套屋顶下：

| 执行精度 | 搬运字节 | AI | 计算时间 | 访存时间 | 归属 |
|---|---|---|---|---|---|
| FP32 (CUDA core) | 201 MB | 683 | 2054 µs | 60 µs | 计算受限（慢 30×） |
| BF16 | 101 MB | 1365 | 139 µs | 30 µs | 计算受限 ✅ |
| FP8 | 50 MB | 2731 | 69 µs | 15 µs | 计算受限，又快 1× |

**注意方向**：精度越低，峰值涨得比字节数降得快（BF16→FP8：字节 ÷2，峰值 ×2），所以拐点右移，**越容易撞上算力天花板**——小形状 GEMM 在低精度下反而更容易变成访存受限。

> ⚠️ 三个坑：① 规格表上的大数字常是 2:1 **稀疏**值（H100 BF16 常被写成 1979，dense 是 989.5），算 roofline 一律用 dense；
> ② 除了 HBM 还有 SRAM/L2 这条带宽高一个数量级的屋顶，FlashAttention 就是把数据留在 SRAM 从而换到那条屋顶上（CH05）；
> ③ Tensor Core 峰值要求形状/地址对齐，达不到会打折，roofline 给的是上界。

**推论（贯穿全书）**：

1. GEMM 已经是"好算子"，别去优化它的 FLOPs，要去**减少非 GEMM 的访存次数**；
2. softmax / norm / RoPE / 掩码 / 类型转换这些"便宜"算子，在长序列下才是真正的大头；
3. 把多个带宽受限的算子**融合（fusion）** 成一个 kernel，让数据只进出 SRAM 一次 —— 这就是
   FlashAttention、融合 RMSNorm、融合 RoPE 的全部动机（CH02、CH05）。

## 6. 分块（tiling）为什么有用

设 SRAM 容量为 $M$ 字节。把 $N\times N$ 的注意力矩阵分块成 $b\times b$ 的块，
每个块需要的 SRAM 是 $O(b\,d)$，而总 HBM 访问次数是

$$ \text{HBM IO} = \Theta\!\left(\frac{N^2 d^2}{M}\right) \quad\text{vs}\quad \text{朴素实现 } \Theta(N^2 + Nd) $$

分块不改变 FLOPs，只改变**数据重用率**：同一块 K/V 被载入 SRAM 后，服务多个 Q 块。
`ch01` 第 5 节用纯 PyTorch 的分块 GEMM 验证了结果与整块 GEMM 一致（`max|Δ| ≈ 6e-5`，浮点结合律误差）。

## 7. 运行

```bash
cd code && python ch01_tensors_strides_einsum.py
```

输出包含 `[PASS]` 断言 10 条。

## 练习

1. 用 `torch.as_strided` 手写一个 $(3,4)$ 张量的"每 2 列取 1 列"视图，并验证 `stride`。
2. 不运行代码，估算 `torch.einsum("bhnd,bhmd->bhnm", q, k)` 在 $B{=}8,H{=}32,N{=}8192,D{=}128$ 时的
   输出元素数和 FLOPs，再与 `ch01` 实测对照。
3. 为什么 `t.reshape` 有时拷贝有时不拷贝？查 `t.is_contiguous()` 与 `t.stride()` 判断。

---

**下一章**：CH02 会把"减少访存"变成可操作的技巧——结合律、矩阵吸收、算子融合。
