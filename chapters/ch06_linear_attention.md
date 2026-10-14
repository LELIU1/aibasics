# CH06 线性注意力：外积状态、秩1更新与 chunkwise 并行

> 代码：`code/ch06_linear_attention.py`
> 前置：CH02（结合律）、CH05（分块/在线 softmax）

## 1. 去掉 softmax，结合律就回来了

CH02 说过：$(QK^\top)V$ 是 $O(N^2d)$，而 $Q(K^\top V)$ 是 $O(Nd^2)$ ——
但因为 softmax 卡在中间，两者不等价。

**如果去掉 softmax**（只保留因果掩码）：

$$ o_t = \sum_{j\le t} (q_t^\top k_j)\, v_j = \underbrace{\Big(\sum_{j\le t} v_j k_j^\top\Big)}_{S_t}\, q_t $$

定义**状态矩阵** $S_t \in \mathbb{R}^{d_v\times d_k}$：

$$ \boxed{S_t = \sum_{j\le t} v_j k_j^\top = S_{t-1} + v_t k_t^\top} $$

它有两个名字，理解它要同时用这两个名字：

| 视角 | 说法 |  einsum |
|---|---|---|
| 外积累加 | 每步加一个**秩1矩阵** $v_t k_t^\top$ | `bnk,bnv->bkv` |
| 联想记忆 | $S$ 是 key→value 的**线性联想记忆** | $S k \approx v$ |

**复杂度**：

| | 训练（$N$ 个 token 一起） | 解码（单步） |
|---|---|---|
| softmax attention | $O(N^2 d)$ | $O(Nd)$（要扫全 KV） |
| 线性注意力 | $O(N d^2)$ | $O(d^2)$（**常数**） |

当 $N \gg d$ 时线性注意力完胜；解码更是从"每步读整个 KV Cache"变成"更新一个 $d\times d$ 矩阵"。

## 2. 加入衰减：遗忘 = 谱收缩

$$ S_t = \lambda\, S_{t-1} + v_t k_t^\top,\qquad 0<\lambda\le 1 $$

展开：$S_t = \sum_{j\le t} \lambda^{\,t-j}\, v_j k_j^\top$ —— 越老的记忆权重越小。

**谱视角**：乘 $\lambda$ 是对 $S$ 的所有奇异值做均匀收缩：$\sigma_i \to \lambda \sigma_i$。
所以 $\lambda$ 是"遗忘率"：$\lambda$ 越小，旧记忆衰减越快。

稳态下 $S$ 的有效记忆长度约 $1/(1-\lambda)$。

## 3. Chunkwise 并行（训练时怎么上 GPU）

纯递推 $S_t = \lambda S_{t-1} + v_t k_t^\top$ 是**串行**的，没法并行——训练时不能这么算。
解决办法是**分块**：

把序列切成 $C$ 个长度为 $c$ 的块。对块 $[l]$：

$$ O^{[l]} = \underbrace{Q^{[l]} K^{[l]\top} \odot M}_{\text{块内：二次，可并行}} V^{[l]}
\;+\; \underbrace{Q^{[l]}\,(\text{块间状态})}_{\text{块间：线性}} $$

块间状态递推（块级别）：

$$ S^{[l+1]} = \lambda^{c}\, S^{[l]} + \sum_{j\in\text{block }l} \lambda^{\,c-1-\mathrm{idx}(j)}\, v_j k_j^\top $$

代码要点（这是 `ch06` 里最容易写错的地方）：

```python
# 块内系数：块内第 idx 个 token，出块时还剩下 c-1-idx 次衰减
coeff = lam ** (c - 1 - idx[:c].float())
S = (lam ** c) * S + (Vc * coeff[:, None]).T @ Kc
```

**对比 CH05**：FlashAttention 是"块内二次 + 无块间状态"（因为 softmax 的归一化是全局的，
用 $m/\ell$ 递推）；线性注意力是"块内二次 + 块间状态传递"。
CH07 的 Gated DeltaNet 会把两者**都用上**。

复杂度：块内 $O(C c^2 d)$，块间 $O(C d^2)$，总 $O(N c d + N d^2/c)$，
取 $c \approx \sqrt{N}$ 最优。

## 4. 核特征映射：用特征逼近 softmax

如果还想要 softmax 的"尖锐"性质，可以用核技巧：

$$ \mathrm{softmax}(q^\top k) \approx \phi(q)^\top \phi(k) $$

于是 $\mathrm{softmax}(QK^\top)V \approx \phi(Q)\big(\phi(K)^\top V\big)$，结合律又成立了。

### ReLU 特征（最简单）

$$ \phi(x) = \mathrm{ReLU}(x)\quad \text{（或 } \mathrm{ReLU}(x)^2 \text{，对应二阶核）} $$

### FAVOR+（Performer）

$$ \phi(x) = \frac{e^{\lVert x\rVert^2/2}}{\sqrt{D}}\sqrt{2}\,\big(\cos(xW + b)\big),\qquad W_{ij}\sim \mathcal N(0,1),\ b_i\sim U(0,2\pi) $$

这是 **Random Fourier Features** 对 RBF 核的无偏估计：

$$ \mathbb{E}\big[\phi(q)^\top\phi(k)\big] = e^{q^\top k} \cdot e^{-\lVert q\rVert^2/2 - \lVert k\rVert^2/2} $$

`ch06` 第 4 节实测（归一化输入后，温度 $T{=}2$）：

| 特征维 $D$ | 相对误差 |
|---|---|
| 64 | 1.1513 |
| 256 | 0.3874 |
| 1024 | 0.2013 |
| 4096 | 0.0975 |

**误差随 $D$ 单调下降但收敛慢**（$O(1/\sqrt D)$），这就是 Performer 类方法在实践中
不如 softmax attention 的主要原因之一。

> 写代码时踩过的坑：公式里 $e^{\lVert x\rVert^2/2}$ 和 $\sqrt{2/D}$ 的**位置和符号**很容易搞错，
> 一旦错，误差会是几百倍而不是 1 倍量级。

## 5. 线性注意力的失败模式：秩饱和 / 记忆冲突

状态 $S_t = \sum_{j\le t} v_j k_j^\top$ 的秩最多是 $t$（每步 +1），但**维度只有 $d\times d$**。

$$ \mathrm{rank}(S_t) \le \min(t,\ d) $$

当 $t > d$ 后，新写入的 $v_t k_t^\top$ **无法再张成新的方向**，只能与已有记忆"挤在一起"：

$$ S k_t \approx \underbrace{v_t}_{\text{想要的}} + \underbrace{\sum_{j<t}(v_j k_j^\top k_t)}_{\text{干扰（记忆冲突）}} $$

读出的值混入了其它 key 的记忆 → **召回精度崩塌**。这就是纯线性注意力在长上下文上
不如 softmax attention 的根本原因（复制/联想任务上尤其明显）。

### Δ rule 预览

解决办法：写入前**先擦除**与该 key 关联的旧值（Widrow-Hoff / delta rule）：

$$ S_t = S_{t-1} + \beta_t\big(v_t - S_{t-1}k_t\big) k_t^\top $$

`ch06` 第 5 节用一个"联想记忆召回"任务演示（$n_{pairs}{=}32$）：

| 方法 | 召回误差 |
|---|---|
| 纯线性（外积累加） | 0.9728 |
| Δ rule（1 遍） | 0.7136 |
| Δ rule（8 遍） | 0.3911 |

Δ rule 明显更好 → 引出 CH07 的完整版 **Gated DeltaNet**。

## 6. 运行

```bash
cd code && python ch06_linear_attention.py
```

5 条 `[PASS]`。

## 练习

1. 推导 $S_t = \sum_{j\le t}\lambda^{t-j} v_j k_j^\top$，并说明 $\lambda$ 对奇异值的缩放。
2. 为什么 chunkwise 里块内系数是 $\lambda^{c-1-\mathrm{idx}}$ 而不是 $\lambda^{\mathrm{idx}}$？
3. 用 `torch.outer` 手写 $S_t$ 递推，与 `einsum("bnk,bnv->bkv")` 对比，验证一致。
4. 设计一个"联想召回"任务：随机 $K$ 对 $(k_i, v_i)$，测随 $t$ 增大线性注意力的召回误差曲线，
   观察在 $t \approx d$ 处的拐点。

---

**下一章**：CH07 Gated DeltaNet —— 把"擦除"写进递推，并用 WY 表示 + 三角求解做到可并行。
