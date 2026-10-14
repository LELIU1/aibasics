# CH05 FlashAttention：在线 softmax + 分块 = 把 $O(N^2)$ 显存压成 $O(N)$

> 代码：`code/ch05_flash_attention.py`
> 前置：CH01（分块/算术强度）、CH02（融合）、CH04（softmax 稳定性）

## 1. 要解决的问题

朴素注意力需要**物化** $N\times N$ 的分数矩阵：

| 步骤 | 显存 | HBM 访问 |
|---|---|---|
| $S = QK^\top$ | $N^2$ | 读 $Q,K$，**写** $S$ |
| $P = \mathrm{softmax}(S)$ | $N^2$ | 读 $S$，**写** $P$ |
| $O = PV$ | $N$ | 读 $P,V$，写 $O$ |

$N{=}8192$ 时 $N^2$ 个 fp32 = 268 MB（一层一头就这么多，实际还要乘 $B\cdot H$）。
**显存是 $O(N^2)$，HBM 访问也是 $O(N^2)$** —— 而且这些都是 CH01 说的"带宽受限"操作。

FlashAttention 的两个武器：

1. **在线 softmax**：不存 $S$、$P$，用递推维护 $(m, \ell, O)$；
2. **分块（tiling）**：$Q$ 分块外层循环、$K,V$ 分块内层循环，中间块只活在 SRAM。

## 2. 在线 softmax 的推导

目标：流式地算 $O = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}}$，每次只看到一块 $s$。

维护三个量：

- $m$：到目前为止见过的**行最大值**
- $\ell$：到目前为止的**（按当前 $m$ 归一化的）分母和**
- $O$：到目前为止的**（按当前 $m$ 归一化的）分子加权和**

看到新块 $S$ 时：

$$ \begin{aligned} m_{new} &= \max(m_{old},\ \mathrm{rowmax}(S))\\ \ell_{new} &= \ell_{old}\cdot e^{\,m_{old}-m_{new}} + \mathrm{rowsum}\big(e^{\,S-m_{new}}\big)\\ O_{new} &= O_{old}\cdot e^{\,m_{old}-m_{new}} + e^{\,S-m_{new}} V \end{aligned} $$

最后 $O_{final} = O / \ell$。

**正确性**来自两件事：

1. softmax 的**平移不变性**（CH02 §3.3）：分母分子同时乘 $e^{-m}$ 不变；
2. 指数的可分解性：$e^{s - m_{new}} = e^{s - m_{old}} \cdot e^{m_{old} - m_{new}}$，
   所以旧贡献只要乘一个**标量重缩放因子**就能对齐到新的 $m$。

于是"重缩放"就是：$O_{old}$ 和 $\ell_{old}$ 都乘同一个 $e^{m_{old}-m_{new}}$。

> 这正是 CH07 Gated DeltaNet 里"块间状态传递"的同款技巧，
> 也是 CH11 Ring Attention 跨设备累加的同款技巧。**学会一次，用三处。**

## 3. 两级分块

```
for i in range(0, N, bq):          # Q 块（外层）—— 结果写回 HBM
    load Q[i:i+bq]
    for j in range(0, N, bk):      # K,V 块（内层）—— 只活在 SRAM
        load K[j:j+bk], V[j:j+bk]
        S = Q_i K_j^T              # (bq, bk)
        online softmax update
    write O_i / ell_i
```

因果模式下：内层只扫到 $j \le i$（省一半 FLOPs），且完全跳过的块不算。

`ch05` 的 `flash_attention(Q, K, V, bq, bk, causal)` 就是这段的纯 PyTorch 实现，
返回 $(O, L)$（$L$ = logsumexp，反向要用）。

## 4. 验证结果

`ch05` 跑的断言（共 22 条）：

| 检查 | 结果 |
|---|---|
| 因果 / 非因果 × 4 组分块大小 vs 朴素 | `max|Δ| ≈ 4e-6` |
| 大 logits（$× 20$）数值稳定 | PASS |
| 不同 `bq/bk` 之间互相一致 | PASS |
| 梯度 $\partial L/\partial Q,\partial L/\partial K,\partial L/\partial V$ | PASS |

## 5. IO 复杂度的账

设 SRAM = $M$。分块大小受限于 $b\,d \lesssim M$。

- 朴素实现 HBM 访问：$\Theta(Nd + N^2 + N^2 + Nd) = \Theta(N^2)$
- FlashAttention：$\Theta\!\left(\frac{N^2 d^2}{M}\right)$

实测（$N{=}8192, d{=}128$，$M$ 按 100 KB 估）：

| 实现 | HBM 访问 |
|---|---|
| 朴素 | 813.69 MB |
| FlashAttention | 42.95 MB |
| **加速比** | **18.9×** |

注意：**FLOPs 完全没变**（甚至因为要重算，反向更多）。省的全是访存。

## 6. 反向传播：用重算换显存

前向只保存：

- 输出 $O$：$O(Nd)$
- logsumexp $L$：$O(N)$

反向时**重算** $S$、$P$（在 SRAM 内），代价是多一次 $O(N^2d)$ 的 GEMM，
但换来的是显存从 $O(N^2)$ 降到 $O(N)$。

$$ \frac{\partial L}{\partial S} = P \odot \big(dO\, V^\top - \mathrm{rowsum}(dO \odot O)\big) $$

（这个式子是标准的 softmax 反向 + $O$ 的行内积修正项。）

## 7. Triton kernel 速写

`ch05` 里附了一段（只打印、不执行，因为沙箱无 GPU）的 Triton 伪代码，要点：

```python
q = tl.load(Q_block)                      # (BLOCK_M, HEAD_DIM)
for j in range(...):
    k = tl.load(K_block)
    qk = tl.dot(q, k) * qk_scale
    qk = tl.where(mask, qk, -inf)
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_new)     # 用 exp2 而非 exp：硬件上更快
    p = tl.math.exp2(qk - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha + tl.dot(p.to(fp8/bf16), v)
```

关键技巧：

- **用 `exp2` 代替 `exp`**：$e^x = 2^{x\log_2 e}$，把 $\log_2 e$ 吸收进 $1/\sqrt d$ 里（CH02 的标量吸收！）；
- `tl.dot` 走 tensor core，要求块大小是 16 的倍数；
- mask 用 `tl.where`，且完全被 mask 的块直接跳过（省算力）。

## 8. 运行

```bash
cd code && python ch05_flash_attention.py
```

22 条 `[PASS]`。

## 练习

1. 手推在线 softmax 的递推式，说明为什么 $\ell$ 和 $O$ 的重缩放因子**必须相同**。
2. 若 $S$ 的某些行整行被 mask（全 $-inf$），$\ell = 0$ 会怎样？代码里怎么兜住？（看 `nan_to_num`）
3. 把 `flash_attention` 改成支持 GQA（K/V 头数不同），验证一致。
4. 为什么反向"重算"反而更快？（提示：重算的 GEMM 是计算受限，而存/读 $N^2$ 是带宽受限）

---

**下一章**：CH06 线性注意力——如果干脆不要 softmax，结合律就回来了。
