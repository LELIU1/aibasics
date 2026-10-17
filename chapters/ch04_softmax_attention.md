# CH04 Softmax 注意力：双线性形式、数值稳定、掩码、GQA 与 KV Cache

> 代码：`code/ch04_softmax_attention.py`
> 前置：CH01–CH03

## 1. 注意力 = 数据依赖的线性算子

把一层的注意力写成矩阵形式：

$$ O = A(X)\cdot (XW_V),\qquad A(X) = \mathrm{softmax}\!\left(\frac{(XW_Q)(XW_K)^\top}{\sqrt d} + M\right) $$

**$A$ 是一个行随机矩阵**（每行非负、和为 1），所以输出的每一行是 $V$ 各行的**凸组合**：

$$ o_n = \sum_{m} a_{nm} v_m,\qquad a_{nm}\ge 0,\ \sum_m a_{nm} = 1 $$

这个视角很有用：

- 注意力 = 用输入**自己**生成的一个线性算子去混合 value；
- 因为 $A$ 依赖数据，它不是"固定的卷积核"，而是**动态的、内容寻址的**；
- 线性注意力（CH06）就是把这个 $A$ 换成能分解的形式。

## 2. 为什么要除以 $\sqrt d$

设 $q, k\in\mathbb{R}^d$ 各分量 iid $\mathcal N(0,1)$，则

$$ q^\top k = \sum_{i=1}^d q_i k_i,\qquad \mathbb{E}[q_i k_i] = 0,\quad \mathrm{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 $$

各项独立 ⇒

$$ \boxed{\mathrm{Var}(q^\top k) = d,\qquad \mathrm{std}(q^\top k) = \sqrt d} $$

`ch04` 第 1 节实测 $d = 16,64,128,256,1024,4096$ 全部吻合（$\mathrm{Var}\approx d$）。

后果：$d$ 越大，logits 的分布越宽，softmax 越接近 one-hot，梯度 $\partial p_i/\partial s_j = p_i(\delta_{ij}-p_j)$ 趋于 0 → **梯度饱和**。

**度量：稳定秩（stable rank）**

$$ \mathrm{sr}(A) = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2} $$

它衡量"这个算子实际用了多少维"。实测（`ch04` 第 6 节：H=8、N=256 的随机 logits 乘不同温度）：

| 缩放系数 | 稳定秩 | 现象 |
|---|---|---|
| $\times 0.2$ | 1.04 | 几乎均匀平均，输出被"抹平" |
| $\times 1.0$ | 2.60 | —— |
| $\times 3.0$ | 32.13 | 开始尖锐 |
| $\times 8.0$ | 64.45 | 接近 one-hot，梯度消失 |

所以 $1/\sqrt d$ 是"让 logits 的尺度与 $d$ 无关"的归一化——不是玄学，是方差控制。

## 3. 数值稳定：平移不变性与 log-sum-exp

$$ \mathrm{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}} = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}},\quad m = \max_j s_j $$

减去最大值保证 $e^{s_i-m} \le 1$，不会溢出（$e^{89}$ 就爆 fp32，而 logits 常常到几百）。

反向传播时只需要 $\log$ 分母，用 **log-sum-exp**：

$$ \mathrm{LSE}(s) = m + \log\sum_j e^{s_j - m} $$

$\log\mathrm{softmax}$ 直接是 $s_i - \mathrm{LSE}(s)$，交叉熵损失也用 LSE 算，全程不产生大数。

`ch04` 第 2 节用 `logits` 很大的情形（`randn(4,100)*20`）验证了手写 LSE 与
`torch.logsumexp` / `torch.log_softmax` 一致（两条 `[PASS]`）。

## 4. 因果掩码 = 加性 $0/-\infty$ 矩阵

$$ M_{nm} = \begin{cases}0, & n \ge m\\ -\infty, & n < m\end{cases} $$

加上 $M$ 后 softmax 把 $n<m$ 的项变成 0 ⇒ $A$ 变成**下三角的行随机矩阵**。

写成矩阵语言：

$$ A = \mathrm{softmax}(S + M) = \text{行归一化}\big(\mathrm{tril}(e^{S})\big),\qquad (e^S)_{nm} = e^{s_{nm}} $$

也就是"把矩阵乘限制到下三角"。这直接给出 FLOPs 减半（$2N^2d \to N^2d$ 每部分）。

> ⚠️ 被 mask 掉的位置要填 **$-\infty$ 而不是 0**：只有 $-\infty$ 才有 $e^{-\infty}=0$。
> 写成 $\mathrm{softmax}(\mathrm{tril}(S))$ 是错的——那是在 **logits** 上做 `tril`，
> 上三角被填成 $0$，于是 $e^{0}=1\neq 0$，掩码完全失效：
> 实测 $N{=}6$ 时上三角最大值高达 **0.206**（正确做法应恒为 0）。
> 正确写法是先取指数、再 `tril`（等价地加 $-\infty$ 掩码），然后做行归一化。

> 实现细节：`-inf` 在 fp16/bf16 里参与 `exp` 会得到 `nan`（`-inf - (-inf)`）。
> 标准做法是用一个**有限的大负数**（如 `-1e9` 或 `torch.finfo(dtype).min`），
> 或者先 `masked_fill` 再减 max。CH05 的 FlashAttention 里用 `-inf` + `nan_to_num` 处理初始块。

## 5. GQA / MQA：K/V 头的重复结构

| 名称 | $H_q$ | $H_{kv}$ | KV Cache（相对 MHA） |
|---|---|---|---|
| MHA | $H$ | $H$ | $1\times$ |
| GQA | $H$ | $H/g$ | $1/g$ |
| MQA | $H$ | $1$ | $1/H$ |

矩阵视角：K/V 头的重复 = 把 $(H_{kv}, D)$ 的张量沿 head 维**广播** $g$ 次：

```python
# k: (N, Hkv, D)  ->  (N, H, D)
# expand 是零拷贝（stride=0 视图），但随后的 reshape 会物化出一份真实拷贝
k = k.unsqueeze(2).expand(N, Hkv, g, D).reshape(N, H, D)
```

注意 `expand` 是 **stride=0 的视图**（CH01），不占内存；但随后的 `reshape` 会物化。
如果想真的零拷贝进 attention kernel，需要 kernel 自己支持 `g` 的步长——
FlashDecoding / PagedAttention 就是这么做的。

`ch04` 第 4 节验证 GQA 与把 K/V 显式复制 $g$ 份的 MHA 数值一致。

## 6. KV Cache = 对 $K, V$ 做行追加

自回归解码第 $t$ 步：

$$ K_{1:t} = \begin{pmatrix}K_{1:t-1}\\ k_t\end{pmatrix},\quad V_{1:t} = \begin{pmatrix}V_{1:t-1}\\ v_t\end{pmatrix},\quad o_t = A_t\, V_{1:t} $$

即**沿序列维（行）追加一行**。这一步的代价：

- 计算：$O(t\,d)$ per step（一个向量 × 矩阵），是 GEMV 而非 GEMM → **带宽受限**；
- 显存：$2\cdot H_{kv}\cdot D\cdot t\cdot (\text{layers})\cdot(\text{bytes})$

这就是为什么解码阶段瓶颈是**带宽**（要把整个 KV Cache 读一遍），
而 prefill 阶段瓶颈是**算力**。这也是 MLA / 量化 KV / GQA 的全部动机（CH10）。

`ch04` 第 5 节验证"增量 decode"与"全量重算"逐步一致。

## 7. 运行

```bash
cd code && python ch04_softmax_attention.py
```

8 条 `[PASS]`。

## 练习

1. 推导 $\partial\,\mathrm{softmax}(s)_i / \partial s_j = p_i(\delta_{ij} - p_j)$，并说明为何 $p$ 接近 one-hot 时梯度消失。
2. 证明行随机矩阵的乘积仍是行随机矩阵（多层注意力堆叠的"保凸组合"性质）。
3. 为什么 MQA 有时掉点？从"不同 head 想 attend 到不同位置"的秩的角度解释。
4. 计算 Llama-3-8B（32 层、$H_{kv}{=}8$、$D{=}128$）在 $t{=}8192$、bf16 下的 KV Cache 大小。

---

**下一章**：CH05 FlashAttention——用在线 softmax + 分块把 $O(N^2)$ 显存压成 $O(N)$。
