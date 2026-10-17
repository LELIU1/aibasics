# CH09 DeepEP：专家并行的通信数学、All-to-All、FP8 与计算-通信重叠

> 代码：`code/ch09_deepep_communication.py`
> 前置：CH08（MoE / 置换路由）、CH01（算术强度）

## 1. 专家并行 = 跨设备的置换

CH08 里，MoE 在一个设备上是 $Y = P^\top \mathrm{blkdiag}(E_e)\, P X$。
**专家并行（EP）**把这些专家分到不同 GPU 上，于是 $P$ 变成**跨设备的置换**：

```
设备 0 的 token ──┐
设备 1 的 token ──┼─→ All-to-All ─→ 每个设备收到"属于自己专家"的 token
设备 2 的 token ──┤                 （各设备收到的数量不同 → 变长！）
设备 3 的 token ──┘
```

**All-to-All 的定义**：设备 $i$ 把发送缓冲的第 $j$ 块发给设备 $j$，并从设备 $j$ 收下 $j$ 的第 $i$ 块。
它就是一个**块置换矩阵**作用于分布式张量。

`ch09` 的 `simulate_alltoall` 在单进程里模拟：

```python
flat_x = x.repeat_interleave(topk, dim=0)    # ⚠️ token 被复制 topk 份！
order  = torch.argsort(expert_id)            # 按目标专家排序 → 即"发送顺序"
ys     = expert_ffn(flat_x[order], ...)      # 本地计算
recv   = torch.zeros_like(flat_x)
recv[order] = ys                             # 反置换 = 接收回本地
```

> ⚠️ 踩过的坑：必须先 `repeat_interleave(topk)`。
> topk=2 时每个 token 去 2 个专家，发送缓冲是 $2T$ 行而不是 $T$ 行。
> 漏掉会得到 `IndexError: index 60 is out of bounds for dimension 0 with size 32`。

## 2. 通信量的账

设：$T$ token、topk=$k$、隐层 $d$、EP 度 $W$。

每步发送量：$T \cdot k \cdot d \cdot (\text{bytes})$（bf16 = 2 字节）。
往返（dispatch + combine）是 **2 倍**。

`ch09` 实测（$T{=}4096$, topk=2, $d{=}4096$, bf16；下表统一取**单向 dispatch** 口径，与代码打印一致）：

| 项 | 数值 |
|---|---|
| 通信量（单向 dispatch） | 67.1 MB |
| 按 400 Gb/s (50 GB/s) 计 | 1.34 ms |
| 专家计算量 | 2886.2 GFLOP |
| 按 300 TFLOP/s 计 | 9.62 ms |
| **计算/通信比** | **7.17** |
| 换成 FP8（通信减半） | 14.34 |

**结论**：EP 的通信量只与 $T \cdot k \cdot d$ 有关，**与专家数 $E$、EP 度无关**。
所以可以放心地把专家数堆到 256（DeepSeek-V3）—— 代价是每步通信延迟（latency）不变，
但要保证 **batch 足够大**才能让通信被计算掩盖。小 batch 推理时 EP 就很不划算。

## 3. FP8 (e4m3) 量化：为什么是它

FP8 有两种格式：

| 格式 | 指数位 | 尾数位 | 用途 |
|---|---|---|---|
| **e4m3** | 4 | 3 | 权重/激活（前向） |
| e5m2 | 5 | 2 | 梯度（反向，动态范围优先） |

e4m3 的**相对精度**：3 位尾数（含隐含位共 4 位精度）⇒ 约 1.2 个十进制有效数字（$\log_{10}2^4$），即相对步长 $2^{-3} = 12.5\%$？
不对——尾数 $m$ 位意味着相对误差约 $2^{-(m+1)} = 2^{-4} = 6.25\%$（舍入到最近）。
实际表现：

- 正规数区间：相邻可表示数的**步长**是 $2^{e-3}$，相对步长 $2^{-3}$；
- **次正规（subnormal）区间**：步长被钉在 $2^{-9}$（e4m3 的最小次正规 = $2^{-9}$）。

`ch09` 的 `fp8_e4m3_emulate` 精确模拟了这两段：

```python
step = torch.where(a >= 2**-6, 2**(e - 3), 2**-9)    # 正规 / 次正规
```

这个"次正规地板"很重要：它让 FP8 在极小数上**绝对误差**有界，
也解释了为什么 FP8 在小幅度张量上比 INT8 好得多。

### FP8 vs INT8

| | FP8 e4m3 | INT8 |
|---|---|---|
| 表示 | 浮点（指数 + 尾数） | 定点仿射 $q = \mathrm{round}(x/s) + z$ |
| 相对误差 | 相对步长上界 6.25%（$2^{-4}$），高斯 RMS 实测 2.65% | 与幅度成反比（小数相对误差爆炸） |
| 动态范围 | 靠指数，天生很大 | 靠 scale，需分块 |
| 硬件 | Hopper/Blackwell tensor core 原生支持 | 老硬件也有 |

`ch09` 第 3 节用一个**混合幅度**张量（一小段极小值 + 一大段大值，动态范围 $5.19\times 10^4$）对比：

| 方法 | 小值段相对误差 | 大值段相对误差 |
|---|---|---|
| FP8 per-tensor | **27.7%** | 2.636% |
| INT8 per-tensor | **100.0%** | 0.998% |
| FP8 block-wise (128) | 2.56% | 2.56% |
| INT8 block-wise (128) | 0.65% | 0.64% |

**这就是 block-wise scaling 的全部动机**：一个 scale 管不住约 $5\times 10^4$ 的动态范围（实测 $5.19\times10^4$），
把张量切成 128 元素的小块、每块一个 scale，误差直接降一个数量级。

> DeepEP / DeepGEMM 用的正是 **per-128-element block-wise scaling**（配合 Hopper 的
> MX 格式思路）。注意 scale 的取值：$s = \mathrm{amax}/448$（448 = e4m3 的最大值），
> 不是 $448/\mathrm{amax}$ —— 这个方向搞反会得到 100% 误差（本教程踩过）。

## 4. 量化误差会不会随 $K$ 累积？

GEMM $c = \sum_{k=1}^K a_k b_k$。若每个 $a_k,b_k$ 有独立量化噪声 $\epsilon$，
则结果的**相对误差不随 $K$ 增长**：

$$ \lVert a\cdot b\rVert \propto \sqrt{K},\qquad \text{噪声} \propto \sqrt{K}\ \Rightarrow\ \text{相对误差} \propto O(1) $$

`ch09` 实测：不同 $K$ 下 GEMM 的相对误差稳定在 **~3.73%**，确实不累积。

这是量化能用在 GEMM 上的**理论基础**。（对比：如果误差是**乘性**的、或存在系统性偏置，
就会随 $K$ 或层数累积 —— 所以要特别小心"零点/偏置"的处理。）

## 5. 计算-通信重叠

EP 的瓶颈是：dispatch（通信）→ 计算 → combine（通信）。
如果串行执行，总时间 = $T_{comm} + T_{comp} + T_{comm}$。

**Dual-stream / chunk 流水线**：把 token 流切成小块，两块交替：

```
stream A:  [comm chunk0][comp chunk0][comm chunk0']
stream B:        [comm chunk1][comp chunk1][comm chunk1']
```

当 chunk 数足够多时，通信几乎完全被计算掩盖，总时间 $\to \max(T_{comm}, T_{comp})$。

`ch09` 的 `demo_overlap` 用"分块数 vs 总耗时"的表格演示了这个收敛过程。

工程要点：

- chunk 太小 → 通信的**延迟**（而非带宽）无法掩盖（每块都有固定开销）；
- chunk 太大 → 掩盖不完全；
- DeepEP 提供 normal（训练/prefill，高吞吐）和 low-latency（推理 decode，
  纯 RDMA、无 SM 占用）两种模式，就是针对这两个极端。

## 6. 运行

```bash
cd code && python ch09_deepep_communication.py
```

## 练习

1. 推导 EP 的通信量公式，说明为什么它与专家数无关。
2. 手算：FP8 e4m3 最大正规数是多少？最小次正规数是多少？（448 和 $2^{-9}$）
3. 为什么 INT8 在"一小段极小值 + 一大段大值"的张量上会 100% 失效？从仿射量化的 $s, z$ 推导。
4. 若把 block size 从 128 改成 32 或 512，误差和 scale 开销如何变化？改代码实测。
5. 计算：若 batch 从 4096 降到 256，计算/通信比变成多少？EP 还划算吗？

---

**下一章**：CH10 低秩、MLA 与权重量化 —— 回到"矩阵本身能不能更小"。
