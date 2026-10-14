# CH07 Gated DeltaNet：Δ rule、Householder 变换、WY 表示与 chunkwise 并行

> 代码：`code/ch07_gated_deltanet.py`
> 前置：CH05（在线 softmax / 分块）、CH06（外积状态 / chunkwise）
> ⚠️ 这是本教程数学密度最高的一章，建议先跑一遍代码再回来读推导。

## 1. 递推式与它的直觉

**Gated Delta Rule**：

$$ \boxed{S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t\, v_t k_t^\top} $$

等价写法（更直观）：

$$ S_t = \alpha_t S_{t-1} + \beta_t\big(v_t - \alpha_t S_{t-1}k_t\big)k_t^\top $$

读法（三步）：

1. **读出**：$\hat v_t = \alpha_t S_{t-1} k_t$ —— 当前状态认为 $k_t$ 对应什么值；
2. **残差**：$v_t - \hat v_t$ —— 新值和旧记忆的差；
3. **写入**：沿 $k_t$ 方向做秩1更新，把残差写进去。

这就是 Widrow-Hoff / LMS 的 **delta rule**，也等价于**在线最小二乘（RLS）的一步**：
$S$ 始终在拟合"最近的 $(k, v)$ 配对"。

与 CH06 纯外积累加的区别：

| | 纯线性注意力 | Gated DeltaNet |
|---|---|---|
| 写入 | $S \mathrel{+}= v k^\top$（只加） | 先**擦除** $Sk^\top$ 方向，再加残差 |
| 转移矩阵 | $I$（恒等） | $\alpha_t(I - \beta_t k_t k_t^\top)$ |
| 秩饱和 | 严重（CH06 §5） | 被擦除缓解 |

## 2. $I - \beta kk^\top$ 的谱结构

设 $\lVert k\rVert = 1$，$T = I - \beta kk^\top$。它在 $k$ 方向和正交补上的作用完全不同：

$$ T k = (1-\beta\lVert k\rVert^2) k,\qquad T u = u \ \ (\forall u \perp k) $$

所以特征值：$\{1-\beta\}$（1 个）和 $\{1\}$（$d-1$ 个）。取不同 $\beta$：

| $\beta$ | 矩阵性质 | 验证 |
|---|---|---|
| $0.5$ | **收缩**：沿 $k$ 方向压到 0.5 | —— |
| $1.0$ | **正交投影** $P$：$k$ 方向清零 | $P^2 = P$（幂等） |
| $2.0$ | **Householder 反射** $H$：$k$ 方向取反 | $H^\top H = I,\ H^2 = I$（正交对合） |

Gated DeltaNet 用 $\beta \in (0,1)$ ⇒ 沿 $k$ 方向**部分擦除**，其余 $d-1$ 个方向**完全不动**。

> 这句话很重要：**擦除是各向异性的**。
> 对比 Mamba2 / 状态空间对偶（SSD）的 $S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^\top$，
> 那里的衰减 $\alpha_t I$ 是**各向同性**的（所有方向一起缩）。
> Gated DeltaNet 的 $\alpha_t(I - \beta_t k_t k_t^\top)$ 只在 $k_t$ 这一个方向上额外收缩。
> 这让它可以"精确删除某条记忆"而不误伤其它记忆 —— 这正是联想召回任务需要的。

## 3. 为什么串行递推上不了 GPU

$S_t$ 依赖 $S_{t-1}$，$N$ 步完全串行。GPU 上这意味着 $N$ 次 kernel launch，
每次只做一个 $d\times d$ 的外积更新 —— 算术强度极低，完全跑不起来。

必须 **chunkwise**：块内用矩阵运算并行，块间只做 $N/c$ 步状态传递。
难点在于：块内第 $t$ 个输出依赖**块内所有前面的转移矩阵的乘积**：

$$ S_t = \Big(\prod_{i\le t} \alpha_i\big(I-\beta_i k_i k_i^\top\big)\Big) S_0 + \cdots $$

那个 $\prod_{i\le t}(I - \beta_i k_i k_i^\top)$ 是 $t$ 个 $d\times d$ 矩阵相乘 —— 必须**压缩**。

## 4. WY 表示：把一串秩1修正压成一个 $d\times C$ 矩阵

这是从 **Householder QR 的分块化**（Bischof–Van Loan，LAPACK 的紧凑 WY 表示）搬过来的技巧。

**定理**：对任意 $\{(k_i,\beta_i)\}_{i\le t}$，存在 $W \in \mathbb{R}^{d\times (t+1)}$ 使得

$$ \prod_{i\le t}\big(I - \beta_i k_i k_i^\top\big) = I - W_{0:t}\, K_{0:t}^\top $$

其中 $W_{0:t}$ 的列 $w_i$ 满足下三角递推：

$$ w_t = \beta_t\Big(k_t - \sum_{i<t}(k_i^\top k_t)\, w_i\Big) $$

写成矩阵就是一个**下三角线性系统**（记 Gram 矩阵 $G_{ij} = k_i^\top k_j$）：

$$ \big(I + \mathrm{diag}(\beta)\cdot \mathrm{tril}(G, -1)\big)\, W = \mathrm{diag}(\beta)\, K $$

用 `torch.linalg.solve_triangular(..., upper=False)` 一次解出全部 $W$。

**收益**：

| | 直接累乘 | WY 表示 |
|---|---|---|
| 运算 | $C$ 次 $d\times d$ 矩阵乘，$O(Cd^2)$，串行 | 一次 $C\times C$ 三角求解，可 batched |
| 并行度 | 无（严格串行） | 沿 batch / head 完全并行 |

`ch07` 第 2 节对 $t = 0..4$ 逐个验证了 $\prod_{i\le t}(I - \beta_i k_ik_i^\top) = I - W_{0:t}K_{0:t}^\top$。

## 5. 加上门控 $\alpha_t$：累积衰减归一化

带 $\alpha_t$ 时，先做**累积衰减**：

$$ c_t = \prod_{i\le t}\alpha_i,\qquad \beta'_t = \frac{\beta_t}{c_t} $$

这样可以把 $\alpha$ 从递推里"除掉"，剩下的结构与 §4 完全一样（只是 $\beta$ 换成 $\beta'$）。
块内最后再乘回 $c_t$。

## 6. 完整的 chunkwise 算法

设块长 $C$，块起始状态 $S_0$。

**Step 1（并行）**：Gram 矩阵 $G = K K^\top \in \mathbb{R}^{C\times C}$

**Step 2（并行，三角求解）**：

$$ \big(I + \mathrm{diag}(\beta)\mathrm{tril}(G,-1)\big)W = \mathrm{diag}(\beta)K \tag{求 $W$} $$

$$ \big(I + \mathrm{tril}(G,-1)\mathrm{diag}(\beta)\big)X = \mathrm{tril}(G,-1)\mathrm{diag}(\beta')V \tag{求 $X$} $$

**Step 3（并行）**：写入项

$$ D = \mathrm{diag}(\beta')V - \mathrm{diag}(\beta) X $$

**Step 4（块内输出）**：令 $M = \mathrm{tril}(QK^\top)$（因果掩码，索引 $[t,j] = q_t^\top k_j$，保留 $j\le t$）

$$ O_{\text{block}} = \mathrm{diag}(c)\Big(\underbrace{Q S_0^\top - M\,(W S_0^\top)}_{\text{块间}} \;+\; \underbrace{M\, D}_{\text{块内}}\Big) $$

**Step 5（推进状态）**：

$$ S \leftarrow c_{C-1}\Big(S - (WS_0^\top)^\top K + D^\top K\Big) $$

> ⚠️ **本教程踩的最大的坑**：Step 4 的 $M$ 必须是 $\mathrm{tril}(Q K^\top)$（索引 $[t,j]$），
> 不能写成 $\mathrm{tril}(K Q^\top)$ 再转置——虽然看起来"转置一下就行"，
> 但 `tril` 的**方向**在转置后是反的，会导致 `max|Δ| ~ 4e+03`。
> 写这类代码时，把每个矩阵的**索引含义**写进注释：`[t,j] = 第 t 个 query 对第 j 个 key`。

## 7. 验证与性能

`ch07` 用 `gdn_naive`（逐 token 循环，最易读的参考实现）作为基准：

| 配置 | `max|Δ|` |
|---|---|
| $N{=}64, d{=}16, C{=}16$ | ~1.9e-06 |
| $N{=}128, d{=}32, C{=}32$ | ~2.7e-06 |
| $N{=}256, d{=}32, C{=}64$ | ~3.8e-06 |
| $\alpha{=}1,\beta{=}1$ / $\alpha{=}1,\beta{=}0.5$ / $\alpha{=}0.9,\beta{=}0.9$ | 全部 PASS |

速度（CPU，$B{=}8, N{=}512, d{=}64, C{=}64$）：

| 实现 | 耗时 |
|---|---|
| `gdn_naive`（逐 token） | 284 ms |
| `gdn_chunkwise` | 12 ms |
| **加速** | **24.5×** |

（GPU 上这个差距会更大，因为 chunkwise 能用上 tensor core。）

## 8. 与 Mamba2 / SSD 的对照

| | Mamba2 / SSD | Gated DeltaNet |
|---|---|---|
| 转移矩阵 | $\alpha_t I$（标量 × 单位阵） | $\alpha_t(I - \beta_t k_t k_t^\top)$ |
| 衰减 | 各向同性（所有方向同速遗忘） | 各向异性（额外擦除 $k_t$ 方向） |
| 并行化 | 矩阵视角（SSD）= 1-semiseparable 矩阵乘 | WY 表示 + 三角求解 |
| 擅长 | 长程平滑依赖 | 联想召回 / 精确修改记忆 |

两者都是"线性注意力家族"，都可以写成**序列变换矩阵** $M \in \mathbb{R}^{N\times N}$ 的形式：
$O = (M \odot \text{mask}) V$。Mamba2 的 $M$ 是 1-semiseparable，
DeltaNet 的 $M$ 是"1-semiseparable + WY 修正"。

## 9. 运行

```bash
cd code && python ch07_gated_deltanet.py
```

14 条 `[PASS]`。

## 练习

1. 证明 $T = I - \beta kk^\top$ 的特征值是 $\{1-\beta, 1,\dots,1\}$（用秩1扰动的特征值性质）。
2. 验证 $\beta{=}1$ 时 $P^2 = P$；$\beta{=}2$ 时 $H^2=I$ 且 $H^\top H = I$。
3. 手动展开 $C{=}3$ 的 WY 表示，写出 $w_0,w_1,w_2$ 的显式表达式。
4. 把 `gdn_chunkwise` 改成支持 padding（$N$ 不整除 $C$），验证与 naive 一致。
5. 思考：为什么 $\beta$ 要用 `sigmoid` 约束到 $(0,1)$？$\beta > 1$ 会发生什么？（看 §2 的谱）

---

**下一章**：CH08 GroupGEMM / MoE —— 从序列建模切回"怎么把一堆形状各异的矩阵乘跑快"。
