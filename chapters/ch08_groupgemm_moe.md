# CH08 GroupGEMM 与 MoE：块对角矩阵、置换路由与变长分组矩阵乘

> 代码：`code/ch08_groupgemm_moe.py`
> 前置：CH01（stride / 分块）、CH02（GEMM 经济学）

## 1. MoE 的矩阵写法

MoE 层：每个 token 选 $k$ 个专家（TopK），输出是加权和

$$ y_t = \sum_{i \in \mathrm{TopK}(x_t)} g_i(x_t)\, E_i(x_t) $$

把 $T$ 个 token 按它们被分配到的专家**排序**后，整层可以写成一个矩阵表达式：

$$ \boxed{Y = P^\top\, \mathrm{blkdiag}\big(E_0, E_1, \dots, E_{E-1}\big)\, P\, X} $$

其中 $P$ 是路由决定的**置换矩阵**（每行每列恰一个 1）。拆开看：

| 步骤 | 数学 | 工程 |
|---|---|---|
| ① 路由 | 置换 $P$ | `argsort` + `repeat_interleave(topk)` |
| ② 专家计算 | 块对角矩阵乘 | **GroupGEMM**：每个块一次 GEMM，块高 = 该专家分到的 token 数 |
| ③ 反置换 + 加权 | $P^\top$ + 逐行乘 $g_i$ | `scatter` + 乘权重 |

置换矩阵的好性质：$P^{-1} = P^\top$，且**逆置换可以用 `argsort` 直接得到**
（因为 $P$ 是正交矩阵，$P^\top$ 也是一个置换）。

## 2. 三种实现及其等价性

`ch08` 实现了三种，并互相验证：

### (a) `moe_loop`：按专家循环

```python
for e in range(E):
    idx = (expert_id == e)
    out[idx] += g[idx] * expert_ffn(x[idx], E_e)
```

- 优点：简单、零 padding；
- 缺点：$E$ 次 kernel launch，$E$ 很大（DeepSeek 有 256 个专家）时调度开销爆炸；
  且每个 GEMM 的 $M$ 很小 → 算术强度低。

### (b) `moe_grouped`：排序 + GroupGEMM

```python
order   = torch.argsort(expert_id)            # 稳定排序，同专家聚在一起
counts  = torch.bincount(expert_id, minlength=E)
offsets = torch.cumsum(counts) - counts       # 每个专家块的起始位置
xs      = x[order]                            # 重排后的输入（连续！）
# 对每个专家切一个 [offset: offset+count] 的连续切片做一次 GEMM
```

- 优点：输入连续，能用上高效 GEMM；kernel 数量可控（甚至可以打包成一个 kernel）；
- 关键：`order` 是**稳定排序**（`argsort` 对相同 key 保持原顺序），所以反置换 `argsort(order)` 能还原。

> ⚠️ 踩过的坑：路由权重 $g$ 必须跟着一起重排（或按 `order` 索引）。
> 写成 `flat_w[order]` 才对；漏掉这一步会得到 `max|Δ| = 3.4e-01`（看起来"差不多"但其实是错的）。

### (c) `moe_blockdiag`：显式块对角

把 $\mathrm{blkdiag}(E_0,\dots,E_{E-1})$ 真的构造出来（或用 one-hot mask 索引），
一次大矩阵乘。适合**教学理解**，实际不用（因为块对角矩阵是 $Ed \times Ed$ 的巨大稀疏矩阵）。

`ch08` 验证三者数值一致（`max|Δ| = 0`）。

## 3. Padding vs Ragged：GroupGEMM 的两个流派

各专家分到的 token 数**不均**（这就是负载不均衡），设最大块大小为 $M_{\max}$：

| 流派 | 做法 | waste |
|---|---|---|
| **Padded** | 所有块补齐到 $M_{\max}$（或对齐到 8/16/64 的倍数），填 0 | $(M_{\max} - \bar M)/\bar M$ |
| **Ragged** | 用 offsets 描述变长，kernel 内部按 ragged 索引 | 0，但 kernel 复杂 |

`ch08` 实测（专家块大小 $[135, 129, 122, 126]$，即均衡时）：

| 对齐粒度 | padding 浪费 |
|---|---|
| 无对齐 | 0% |
| 对齐到 128 的倍数 | 32.1% |
| 对齐到 256 的倍数 | 65.7% |
| 对齐到 512 的倍数 | 100.2% |

**结论**：对齐粒度是把双刃剑——tensor core 要求 $M$ 是 8/16 的倍数才高效，
但粒度太大会浪费一半以上算力。工程上一般取 **$M$ 对齐到 64 或 128**，
或者用 ragged kernel（如 DeepGEMM 的做法）。

## 4. 负载均衡的辅助损失

如果不加约束，路由会塌缩（所有 token 都去少数几个专家）。经典辅助损失：

$$ \mathcal{L}_{aux} = \alpha \cdot E \sum_{e=1}^{E} f_e\, P_e $$

其中：

- $f_e$ = 分到专家 $e$ 的 token 比例（**实际负载**）；
- $P_e$ = 专家 $e$ 的平均路由概率（**软负载**）。

**为什么是 $f \cdot P$ 而不是别的？**
在完全均衡时 $f_e = 1/E$，$P_e = 1/E$，则 $\sum_e f_e P_e = E \cdot (1/E^2) = 1/E$，
乘 $E$ 得 **1** —— 所以损失值 1 是"完美均衡"的基准线，越大于 1 越不均。

`ch08` 实测：

| 场景 | $E\cdot\sum f_e P_e$ |
|---|---|
| 均衡路由 | 1.000 |
| 轻度不均 | 1.136 |
| 严重不均 | 2.510 |

这个损失在**可微**上是必要的：$f_e$ 本身是离散的（不可导），所以用连续的 $P_e$ 作为它的"代理"，
再让梯度通过 $P_e$ 传回路由器。

## 5. 置换矩阵的性质验证

`ch08` 检查：

- $P^\top P = I$（正交）
- $P^{-1} = P^\top$，且 `argsort(order)` 就是 $P^\top$
- 正置换 + 反置换 = 恒等

这也是 CH09 DeepEP 里 all-to-all 的数学基础：跨设备的 token 分发**就是一个跨设备的置换**。

## 6. GroupGEMM 与 Attention 的联系

有意思的是：**变长 attention（varlen）用的也是同一套东西**。
一个 batch 里有多条不同长度的序列，把它们拼接成一个大序列 + `cu_seqlens`（累积长度 offsets），
然后每个序列段做一次 attention —— 这就是 ragged GroupGEMM 在 attention 上的应用。

| | MoE | Varlen Attention |
|---|---|---|
| 分组依据 | expert_id | sequence_id |
| offsets | expert 边界 | `cu_seqlens` |
| 块内运算 | FFN GEMM | FlashAttention |
| 实现 | `argsort` + `bincount` | `cumsum(lengths)` |

## 7. 运行

```bash
cd code && python ch08_groupgemm_moe.py
```

7 条 `[PASS]`。

## 练习

1. 证明置换矩阵满足 $P^{-1} = P^\top$，并说明为什么 `argsort(order)` 是逆置换。
2. 实现 ragged 版本（不 padding），与 padded 版对比结果。
3. 推导辅助损失在完全均衡时 $\mathcal L_{aux} = \alpha$，在完全塌缩时（所有 token 去 1 个专家）的值。
4. 若 topk=2，token 会被复制两份，`repeat_interleave(topk, dim=0)` 为什么是必须的？（CH09 会再遇到）

---

**下一章**：CH09 DeepEP —— 把这个置换搬到**跨设备**，就是 all-to-all 通信。
