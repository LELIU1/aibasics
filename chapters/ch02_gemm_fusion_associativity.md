# CH02 GEMM 经济学：结合律、矩阵吸收、算子融合

> 代码：`code/ch02_gemm_fusion_associativity.py`
> 前置：CH01（stride / einsum / 算术强度）

## 1. Transformer 的 GEMM 普查

一个标准解码器层（$d{=}4096$, $H{=}32$, $D{=}128$, $f_{ff}{=}14336$, $N{=}2048$, $B{=}8$）里，
所有大矩阵运算的 FLOPs 分布：

| 部分 | 表达式 | GFLOP | 占比 |
|---|---|---|---|
| QKV 投影 | $X W_{qkv}$ | 824.6 | 18.2% |
| Attention 分数 | $QK^\top$ | 274.9 | 6.1% |
| Attention 输出 | $A V$ | 274.9 | 6.1% |
| O 投影 | $O W_o$ | 274.9 | 6.1% |
| MLP up + gate | $h W_{up}, h W_{gate}$ | 1924.2 | 42.4% |
| MLP down | $a W_{down}$ | 962.1 | 21.2% |
| 合计 | | 4535.5 | 100% |

（表中数值即 `ch02.py` 第 1 节的打印值。注意**计数口径并不统一**：attention 两行含 MAC 系数 2
（$2HBN^2D$），QKV / O / MLP 各行按"矩阵乘法次数"计数未乘 2。所以绝对值可直接用，但
attention 的占比被高估约 2×：若统一按 $2MNK$ 计，合计约 $8521$ GFLOP，attention 两行合计
占比降到约 6.5%（每行 3.2%）。）

**注意 attention 占比随 $N$ 的剧烈变化**（这是长上下文一切优化的根源）：

| $N$ | attention | 其它 | 占比 |
|---|---|---|---|
| 2048 | 549.8 GFLOP | 3985.7 | **12.1%** |
| 8192 | 8796.1 GFLOP | 15942.9 | **35.6%** |
| 32768 | 140737.5 GFLOP | 63771.7 | **68.8%** |
| 131072 | 2251799.8 GFLOP | 255086.7 | **89.8%** |

>$N$ 翻 4 倍，attention 的 FLOPs 翻 16 倍，其它项只翻 4 倍。
>所以"长上下文优化"本质上就是跟 $N^2$ 打架（CH05–CH07）。

## 2. 结合律：换乘法顺序能省一个数量级

矩阵乘法满足结合律 $(AB)C = A(BC)$，但**不满足交换律**。
FLOPs 却强烈依赖于括号位置：

$$ (M\times K)\cdot(K\times N)\cdot(N\times P) $$

分两步算，每步 = $2 \times \text{行} \times \text{列} \times \text{缩并维}$：

| 顺序 | 第一步 | 第二步 | 合计 FLOPs |
|---|---|---|---|
| $(AB)C$ | $AB\in\mathbb{R}^{M\times N}$：$2MKN$ | $(AB)C\in\mathbb{R}^{M\times P}$：$2MNP$ | $2MKN + 2MNP$ |
| $A(BC)$ | $BC\in\mathbb{R}^{K\times P}$：$2KNP$ | $A(BC)\in\mathbb{R}^{M\times P}$：$2MKP$ | $2KNP + 2MKP$ |

⚠️ 两者**输出形状都是 $M\times P$**，数值完全一样（结合律），但 FLOPs 差很多。

设 $M=K=P=64$、$N=4096$（长序列场景）：

| 顺序 | 算式 | 合计 |
|---|---|---|
| $(AB)C$ | $2\!\cdot\!64\!\cdot\!64\!\cdot\!4096 + 2\!\cdot\!64\!\cdot\!4096\!\cdot\!64$ | $33.5\text{M} + 33.5\text{M} = \mathbf{67.1\text{M}}$ |
| $A(BC)$ | $2\!\cdot\!64\!\cdot\!4096\!\cdot\!64 + 2\!\cdot\!64\!\cdot\!64\!\cdot\!64$ | $33.5\text{M} + 0.52\text{M} = \mathbf{34.0\text{M}}$ |

$A(BC)$ 之所以省，是**把大的 $K\times N$ 先收缩成小的 $K\times P$**，后面那步工作量直接掉两个量级。

若 $N \gg M,P$（长序列），两者相差巨大。

### 案例 A：LoRA 的权重合并

$$ y = x W + x\cdot \frac{\alpha}{r} B A = x\left(W + \frac{\alpha}{r}BA\right) $$

- 不合并：$2\,d_{in}d_{out} + 2\,r(d_{in}+d_{out})$ FLOPs，**两次 kernel launch**，多一次激活读写；
- 合并：$W' = W + \frac{\alpha}{r} BA \in \mathbb{R}^{d_{in}\times d_{out}}$，一次 GEMM。

推理阶段**必须合并**（省掉整条旁路 + 一次逐元素加）；训练阶段**不能合并**（要保持 $A,B$ 可训、
$W$ 冻结，且合并后反向传播会破坏低秩结构）。

`ch02` 第 2 节实测 FLOPs：

| 写法 | GFLOP |
|---|---|
| 合并 $W' = W + \frac\alpha r BA$ | 8.59 |
| 链式 $x \to A \to B$ | 8.66 |
| **先算 $\Delta W$ 再加**（最蠢的写法） | 17.72 |

FLOPs 差别不大（因为 $r$ 很小），真正的收益是**少一次 kernel launch、
少一份中间激活的读写、少一次逐元素加**。

### 案例 B：$(QK^\top)V$ vs $Q(K^\top V)$

- $(QK^\top)V$：$2N^2d + 2N^2d$
- $Q(K^\top V)$：$2Nd^2 + 2Nd^2$

实测（$N{=}4096, d{=}64$，见 `ch02.py` 第 2 节）：$(QK^\top)V$ = 4295.0 MFLOP，
$Q(K^\top V)$ = 67.1 MFLOP，**省 64×**。
**但两者并不等价**（实测直接换序的最大偏差是 2479.7）—— 只有去掉 softmax 之后才能这么换
（这正是 CH06 线性注意力的全部秘密）：

$$ \mathrm{softmax}(QK^\top)V \ne Q(K^\top V) $$

因为 softmax 是**逐行非线性**，它卡在中间，把结合律打断。
`ch02` 明确验证了这个不等价（这是理解"为什么要发明线性注意力"的关键）。

## 3. 矩阵吸收：把对角/标量"吸"进邻居

这是最划算的优化：**零 FLOPs、零访存，纯代数恒等式**。

### 3.1 标量吸收

$$ \frac{1}{\sqrt d} \text{ 可以吸进 } W_Q \text{ 或 } W_K:\quad \frac{q^\top k}{\sqrt d} = \left(\tfrac{1}{d^{1/4}}q\right)^\top\left(\tfrac{1}{d^{1/4}}k\right) $$

### 3.2 对角吸收（RMSNorm 的 $\gamma$）

RMSNorm：

$$ \mathrm{RMSNorm}(x) = \underbrace{\frac{x}{\mathrm{rms}(x)}}_{\text{归一化}} \odot \gamma $$

$\gamma$ 作用在**输入维度**上（$x$ 的第 $j$ 个分量乘 $\gamma_j$）。记 $\hat x = x/\mathrm{rms}(x)$；
下面统一用**代码里的写法** $y = h\,W^\top$（$W\in\mathbb R^{d_{out}\times d_{in}}$，$h$ 是归一化后的激活；
它等价于教科书写法 $y=\hat x\,W$ 里的 $W\in\mathbb R^{d_{in}\times d_{out}}$ 取转置）。若 RMSNorm 后面
紧跟这个线性层，则

$$ (\hat x \odot \gamma)\, W^\top = \hat x\,\big(\mathrm{diag}(\gamma)W^\top\big)
= \hat x\,\big(W\,\mathrm{diag}(\gamma)\big)^\top = \hat x\,W'^\top
\quad\Longrightarrow\quad W'_{ij} = \gamma_j\, W_{ij} $$

即 **$\gamma$ 按列吸收进 $W$**：$W$ 的第 $j$ **列**（也就是输入维 $j$）整体乘 $\gamma_j$，
代码写 `W * gamma[None, :]`（代码里 $W$ 形状 $(d_{out}, d_{in})$，$\gamma$ 长度 $d_{in}$）。
（若坚持 $y=\hat x\,W$ 的记法，$W\in\mathbb R^{d_{in}\times d_{out}}$，同一件事要写
`W * gamma[:, None]`，即按**行**缩放。**代码是对的，别照这句去改代码。**）

> ⚠️ 这是本教程踩过的坑：广播方向搞反会得到 `max|Δ| ~ 1.7e+02`。
> 记住口诀：**$\gamma$ 乘在被求和的那一维上**。

### 3.3 softmax 的平移不变性

$$ \mathrm{softmax}(s + c\mathbf 1) = \mathrm{softmax}(s) $$

所以 causal mask 的 $-\infty$ 可以加在任意一侧；而在线 softmax（CH05）正是靠这一点做重缩放。

## 4. 算子融合的账

`ch02` 第 4 节拿一个 $137.4$ GFLOP 的 GEMM（输出 67 MB）做账：

| 方案 | 访存量 | 算术强度 |
|---|---|---|
| 不融合（`matmul → add → gelu → mask` 各读写一遍） | 604 MB | 227.6 |
| 融合成一个 kernel | 201 MB | 682.7 |

**访存减少 3.0×，FLOPs 一点没变。**

实测（CPU）：`gelu(add(x))` 分两个 pass 约 9~10 ms，融合成单 pass 约 5 ms（**约 2×**）。
**计时数值随机型/机器而异，看数量级和比值即可。**

> 注意这里"不融合"的版本**已经是 GEMM 主导**（AI 仍有 227），所以只差 3×。
> 真实的 norm / RoPE / 掩码这些纯逐元素算子，AI 只有 ~1，融合收益会大得多。

## 5. 头合并：把小 GEMM 变成大 GEMM

多头注意力里，每个头一次 $N\times D \times N$ 的小 GEMM。$H$ 个头就有 $H$ 个小 GEMM，
每个的算术强度都不高（因为 $D=128$ 很小）。

**做法**：把 head 维挪到 batch 维，用一次 `bmm`：

```python
# q: (B, N, H, D) -> (H, B*N, D)
q_ = q.transpose(1, 2).reshape(H, B * N, D)
k_ = k.transpose(1, 2).reshape(H, B * N, D)
S = torch.bmm(q_, k_.transpose(1, 2))        # (H, B*N, B*N)
```

`ch02` 第 5 节实测（CPU）：32 次小 GEMM 约 105 ms vs 一次 `bmm` 约 100 ms ——
**只快 1.05~1.10×**（**数值随机型/机器而异，同机多次跑也会抖**）。
这里的 `bmm` 是 **batch $= H = 32$**，每批 $(B{\cdot}N)\times D = 8192\times128$ 乘 $128\times128$。
这个数字有点"打脸"，但它恰好说明两件事：

1. CPU 上 BLAS 本身对单 GEMM 已经很高效，收益主要来自**省掉 32 次 kernel launch**；
2. **GPU 上差别会大得多**：除了 launch 开销，还有"每个小 GEMM 都要把权重重新读一遍"
   —— 权重访存被重复了 $H$ 次，这才是大头。

所以这个优化的主要价值在 GPU / 小 head_dim 场景，别指望在 CPU 上看到数量级提升。

> 这也是为什么 GQA（CH04）在 $H_{kv} \ll H$ 时依然要保持"头"这个 batch 维——
> 它换来的是 KV Cache 的成倍缩小，而计算侧用 `expand` 补回来。

## 6. 运行

```bash
cd code && python ch02_gemm_fusion_associativity.py
```

7 条 `[PASS]`。

## 练习

1. 证明 $W' = W + \frac{\alpha}{r}BA$ 后，前向等价但**反向不等价**（提示：看 $\partial L/\partial A$ 的表达式）。
2. RMSNorm 的 $\gamma$ 吸收方向：若 $\gamma$ 作用在**输出**维（比如某些实现的 gated norm），
   应该改 $W$ 的行还是列？
3. 估算：把 $H{=}32$ 个 $(8192,128)\times(128,8192)$ 的小 GEMM 合成一个 batch=32 的 `bmm`，
   访存量变化多少？

---

**下一章**：CH03 RoPE —— 全书第一个"漂亮的代数结构"：旋转群 $SO(2)$ 与群同态。
