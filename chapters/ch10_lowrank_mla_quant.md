# CH10 低秩近似、MLA 与权重量化：Eckart–Young、矩阵吸收、GPTQ/AWQ

> 代码：`code/ch10_lowrank_mla_quant.py`
> 前置：CH02（矩阵吸收）、CH03（RoPE）、CH04（GQA / KV Cache）、CH09（量化）

## 1. Eckart–Young 定理：低秩近似的理论基石

**定理**（Eckart–Young–Mirsky）：设 $A$ 的奇异值 $\sigma_1\ge\sigma_2\ge\cdots$，则

$$ \min_{\mathrm{rank}(B)\le r}\ \lVert A - B\rVert_F = \sqrt{\sum_{i>r}\sigma_i^2} $$

且最优解就是**截断 SVD**：

$$ A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top = U_r \Sigma_r V_r^\top $$

**推论**：低秩近似的效果完全由**奇异值的尾部能量**决定。
定义"保留能量比"：

$$ \eta(r) = \frac{\sum_{i\le r}\sigma_i^2}{\sum_i \sigma_i^2} $$

`ch10` 第 1 节构造一个谱按 $\exp(-i/6)$ 衰减的矩阵，实测截断 SVD 的误差
与 $\sqrt{\sum_{i>r}\sigma_i^2}$ 吻合（`max|Δ| ≈ 1.2e-07`）。

### LoRA 的秩怎么选

LoRA 假设微调的更新 $\Delta W$ 是低秩的：$\Delta W = BA$，$B\in\mathbb{R}^{d_{out}\times r}, A\in\mathbb{R}^{r\times d_{in}}$。

- $r$ 太小 → 表达力不足（$\eta(r)$ 太低）；
- $r$ 太大 → 参数量与全量微调接近，失去意义。

经验法则：**看 $\Delta W$ 的谱衰减速度**。如果前 $r$ 个奇异值就吃掉 90% 能量，
$r$ 就够了。`ch10` 第 2 节用真实谱衰减（指数衰减 vs 均匀谱）对比了不同 $r$ 的逼近误差。

> 注意：要评估的是 **$\Delta W$ 的谱**，不是 $W$ 的谱！
> 预训练权重 $W$ 的谱通常很"平"（信息均匀分布），而 $\Delta W$ 的谱往往很"尖"。
> `ch10` 里特意对比了"重建整个 $W$"和"重建 $\Delta W$"两种误差。

## 2. MLA：多层注意力 = 把 KV 压进一个低秩潜向量

**MLA（Multi-head Latent Attention，DeepSeek-V2/V3）** 的核心：
不缓存每个头的 $K, V$，而是缓存一个**共享的低秩潜向量** $c_{KV}$。

### 展开版

$$ c_{KV} = x W_{DKV}\quad (\text{下投影到 } d_c),\qquad
\begin{cases} k^{(h)} = c_{KV} W_{UK}^{(h)}\\ v^{(h)} = c_{KV} W_{UV}^{(h)}\end{cases} $$

### 吸收版（推理时的关键技巧）

QK 方向：

$$ q^{(h)\top} k^{(h)} = \big(W_{UQ}^{(h)} c_q\big)^\top \big(W_{UK}^{(h)} c_{KV}\big)
= c_q^\top \underbrace{\big(W_{UQ}^{(h)\top} W_{UK}^{(h)}\big)}_{\text{预先合并成一个 } d_c\times d_c} c_{KV} $$

把两个矩阵**预先乘好**，推理时直接用合并矩阵 —— 省掉一次投影。

输出方向（更妙）：

$$ \sum_h W_O^{(h)}\, A^{(h)}\, \big(W_{UV}^{(h)} c_{KV}\big)
= \sum_h \big(A^{(h)} c_{KV}\big)\underbrace{\big(W_{UV}^{(h)} W_O^{(h)}\big)}_{\text{合并}} $$

注意 $A^{(h)}$ 是 $N\times N$ 的注意力矩阵，这一步把"先投影 V 再乘 $W_O$"
换成了"先做 attention 加权再一次性投影" —— 顺序换了，**乘的矩阵变少了**。

`ch10` 验证：展开版 == 吸收版，`max|Δ| = 2.235e-08`。

### 解耦 RoPE：为什么必须留 $d_r$ 维

问题：RoPE 是**位置相关的旋转**，它和"低秩潜向量"不兼容。

$$ \big\langle \rho(m)W_{UQ}c_q,\ \rho(n)W_{UK}c_{KV}\big\rangle
\ne c_q^\top (\text{某个固定矩阵})\, c_{KV} $$

因为 $\rho(m)$ 插在中间，且依赖 $m$ —— **无法吸收进一个与位置无关的矩阵**。

**解决办法**：把 $q, k$ 拆成两部分：

$$ q = [\underbrace{q^{C}}_{\text{可压缩部分}},\ \underbrace{q^{R}}_{\text{RoPE 部分}}],\qquad
k = [k^{C},\ k^{R}] $$

- $q^C, k^C$ 走低秩潜向量（可吸收、可压缩）；
- $q^R, k^R$（$d_r = 64$ 维）单独做 RoPE，**每个头都要缓存**。

所以 MLA 的 KV Cache = $d_c + d_r$ per token，而不是 $2\cdot H\cdot D$。

### KV Cache 对比

`ch10` 实测（$H{=}8, D{=}64, d_c{=}32, d_r{=}16$，教学用小模型）：

| 方法 | 每 token 缓存元素数 |
|---|---|
| MHA | 512 |
| GQA ($H_{kv}{=}2$) | 128 |
| MQA | 64 |
| **MLA** | **96** |

真实规模（DeepSeek-V3：$H{=}128, D{=}128, d_c{=}512, d_r{=}64$）：

| 方法 | 每 token | 相对 |
|---|---|---|
| MHA | 32768 | $1\times$ |
| **MLA** | **576** | **$1/56.9$** |

这就是 MLA 能在 128K 上下文下把 KV Cache 压到可接受范围的原因。

## 3. 权重量化：从 INT8 到 INT2

### 仿射量化

$$ \hat w = s\cdot\big(\mathrm{clamp}(\mathrm{round}(w/s) + z,\ -2^{b-1},\ 2^{b-1}-1) - z\big) $$

- per-tensor：整个张量一个 $(s,z)$；
- per-channel（按输出通道）：每个输出通道一个 —— 通常用这个；
- per-group（每 $g$ 个元素）：$g=32/64/128$，精度与开销的折中。

`ch10` 第 4 节实测 $y = x\hat W^\top$ 的相对误差：

| 位宽 | per-tensor | group=128 | group=32 |
|---|---|---|---|
| INT8 | 1.080% | 0.646% | 0.533% |
| INT4 | **19.586%** | 11.709% | 9.688% |
| INT3 | 45.715% | 27.335% | 22.584% |
| INT2 | 97.181% | 76.568% | 65.526% |

两个观察：

1. **均匀量化的理论误差** $\approx \dfrac{\mathrm{step}/\sqrt{12}}{\sigma}$，$\mathrm{step} = \dfrac{2\,\mathrm{amax}}{2^b-1}$。
   对高斯权重，INT8 ≈ 1%，INT4 ≈ 10% —— 和实测吻合。
2. **光靠均匀量化，INT4 的 19.6% 是完全不能用的**。
   真实模型能跑 INT4 甚至 INT2，靠的是把目标从 $\lVert W-\hat W\rVert_F$ 换掉（见下节）。
3. group 越小误差越低，但 scale 的存储和访存开销越大 —— 实践中 group=32/64/128 是常见折中。

### 为什么需要 Hessian 加权（GPTQ / AWQ）

**关键洞察**：不是所有权重的误差都同等重要。

考虑损失函数对权重的二阶近似：$\Delta L \approx \frac12 \Delta w^\top H\, \Delta w$，
其中 $H = \mathbb{E}[xx^\top]$ 是**输入特征的二阶矩**（Hessian 的近似）。

所以量化目标不是 $\min\lVert W - \hat W\rVert_F$，而是

$$ \min_{\hat W}\ \big(W - \hat W\big)^\top H \big(W - \hat W\big) $$

**含义**：

- 若某个权重对应的输入特征方差很大（$H$ 的对角元大）→ 它的误差被放大 → 要保留高精度；
- **AWQ** 据此对重要通道乘一个放大系数 $s$（再在别处除回来），等效于给它们更多"量化级数"；
- **GPTQ** 则是逐个量化权重，并把当前权重的量化误差**按 $H^{-1}$ 分配到后面的未量化权重**上（OBQ 的加速版）。

`ch10` 用对角 Hessian 的加权目标演示了这一点：同样位宽下，Hessian 加权的目标
在"模型输出误差"上明显优于普通 Frobenius 目标。

## 4. 三个技巧的关系

| 技巧 | 数学工具 | 作用 |
|---|---|---|
| LoRA | 低秩分解 $BA$ | 减少**可训练参数** |
| MLA | 低秩潜向量 + **矩阵吸收** | 减少 **KV Cache** |
| 量化 | 仿射映射 + Hessian 加权目标 | 减少**权重/激活的位宽** |

共同点：**它们都在问"这个矩阵真正需要多少信息量"**，
而答案分别由奇异值谱（LoRA/MLA）和 Hessian 谱（量化）给出。

## 5. 运行

```bash
cd code && python ch10_lowrank_mla_quant.py
```

## 练习

1. 用 `torch.linalg.svd` 验证 Eckart–Young：对随机矩阵，比较 $\lVert A - A_r\rVert_F$ 与 $\sqrt{\sum_{i>r}\sigma_i^2}$。
2. MLA 吸收：写出 $W_{UQ}^{(h)\top}W_{UK}^{(h)}$ 的形状，说明推理时省了哪次矩阵乘。
3. 为什么 RoPE 部分不能进潜向量？写出含 $\rho(m)$ 的式子并指出它在哪里"卡住"。
4. 实现 per-group (g=128) 的 INT4 量化，与 per-channel 对比误差。
5. 用对角 Hessian $H = \mathrm{diag}(h)$ 实现一个简单的"重要性加权量化"，与普通量化对比输出误差。

---

**下一章**：CH11 分布式线性代数 + 综合实战。
