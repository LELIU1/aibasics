# CH03 RoPE：旋转矩阵、正交群与相对位置编码

> 代码：`code/ch03_rope_rotation.py`
> 前置：CH01（einsum）、CH02（矩阵吸收）

## 1. 核心思想：把位置编码做成"旋转"

传统位置编码是把位置向量**加**到词向量上：$x_t \to x_t + p_t$。
RoPE 不这样做——它把位置 $m$ 映射成一个**旋转矩阵** $\rho(m)$，然后

$$ q_m = \rho(m)\, W_q x_m,\qquad k_n = \rho(n)\, W_k x_n $$

于是注意力分数变成

$$ \langle q_m, k_n\rangle = (\rho(m)W_q x_m)^\top (\rho(n) W_k x_n) = x_m^\top W_q^\top \underbrace{\rho(m)^\top\rho(n)}_{=\ \rho(n-m)} W_k x_n $$

**关键恒等式**：$\rho(m)^\top \rho(n) = \rho(n-m)$。
分数只依赖相对距离 $n-m$ —— 这就是"相对位置编码"，而且**不需要改注意力公式**，
只是把 $q,k$ 旋转一下。

## 2. 二维旋转矩阵 $R(\theta)$ 的代数性质

$$ R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix} $$

| 性质 | 公式 | 含义 |
|---|---|---|
| 正交 | $R^\top R = I$ | 保范数、保内积（不拉伸不剪切） |
| 行列式 | $\det R = +1$ | 属于**特殊正交群** $SO(2)$，纯旋转无镜像 |
| 群同态 | $R(\theta_1)R(\theta_2) = R(\theta_1+\theta_2)$ | 旋转可加 |
| 逆 | $R(\theta)^{-1} = R(-\theta) = R(\theta)^\top$ | **反向传播就是反向旋转** |

最后一条在代码里很实用：RoPE 的梯度不需要额外推导，把 $q$ 旋转 $-m$ 即可。

`ch03` 第 1 节验证 $0^\circ/30^\circ/90^\circ/180^\circ/270^\circ$ 下的 $\lVert R^\top R - I\rVert \approx 10^{-8}$、$\det = +1$。

## 3. 高维：分块对角 = $d/2$ 个独立旋转平面

$d$ 维空间被切成 $d/2$ 个 2D 平面，每个平面用不同角速度 $\theta_i$ 旋转：

$$ \rho(m) = R(m\theta_0) \oplus R(m\theta_1) \oplus \cdots \oplus R(m\theta_{d/2-1}) = \mathrm{blkdiag}\big(R(m\theta_i)\big)_{i} $$

其中频率（这就是 LLaMA/Mistral 里的 `inv_freq`）：

$$ \theta_i = \mathrm{base}^{-2i/d},\qquad i = 0,\dots,d/2-1,\quad \mathrm{base}=10000 $$

**为什么是几何级数？**
$\theta_i$ 从 $1$ 衰减到 $\mathrm{base}^{-1}\approx 10^{-4}$，跨度 4 个数量级：

- 高频平面（$i$ 小，$\theta\approx 1$）：转得快 → 编码**近距离**精细差异；
- 低频平面（$i$ 大，$\theta\approx 10^{-4}$）：转得慢 → 编码**远距离**粗粒度信息。

这跟傅里叶基/小波基的思路一致：多尺度分解。

## 4. 两种内存布局：交错 vs 半分

同一个 $\rho(m)$ 有两种摆法：

**交错布局（interleaved，GPT-J / RoFormer 原始）**：相邻两个分量组成一对

$$ (x_0,x_1,x_2,x_3,\dots) \Rightarrow (x_0,x_1),(x_2,x_3),\dots $$

**半分布局（half-split，LLaMA / HF 主流）**：前后两半配对

$$ (x_0,\dots,x_{d/2-1})\ \|\ (x_{d/2},\dots,x_{d-1}) \Rightarrow (x_0,x_{d/2}),(x_1,x_{d/2+1}),\dots $$

两者由**置换矩阵 $P$** 共轭关联：

$$ \rho_{\text{half}}(m) = P^\top\, \rho_{\text{int}}(m)\, P,\qquad P^\top P = I $$

代码上，半分布局只要一次 `chunk(2, -1)`：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, -1)                 # 前半 / 后半
    return torch.cat((-x2, x1), -1)

def apply_rope(x, cos, sin):                # cos/sin 形状 (..., d/2)
    return x * cos.cat((cos, cos)) + rotate_half(x) * sin.cat((sin,sin))
```

或者更省：直接

```python
t1, t2 = t.chunk(2, -1)
out = torch.cat((t1*cos - t2*sin, t2*cos + t1*sin), -1)     # cos/sin: (..., d/2)
```

`ch03` 第 2 节用显式 $\mathrm{blkdiag}$ 矩阵（交错）、半分实现、以及 $P^\top R_{\text{int}} P$ 三者互验，全部 PASS。

## 5. 复数视角

半分布局天然是复平面上的相位乘法。把 $(x_{i}, x_{i+d/2})$ 视为 $z_i = x_i + \mathrm{i}\,x_{i+d/2}$：

$$ z_i \mapsto z_i \cdot e^{\mathrm{i} m\theta_i} $$

| 实数形式 | 复数形式 |
|---|---|
| $2\times2$ 旋转矩阵 | 单位模复数 $e^{\mathrm{i}m\theta}$ |
| $R^\top R = I$ | $\lvert e^{\mathrm{i}\phi}\rvert = 1$ |
| $R(\theta_1)R(\theta_2)=R(\theta_1+\theta_2)$ | $e^{\mathrm{i}\phi_1}e^{\mathrm{i}\phi_2}=e^{\mathrm{i}(\phi_1+\phi_2)}$ |
| $\rho(m)^\top\rho(n)=\rho(n-m)$ | $\overline{e^{\mathrm{i}m\theta}}\,e^{\mathrm{i}n\theta} = e^{\mathrm{i}(n-m)\theta}$ |

复数形式在实现上更省：`torch.view_as_complex` 后一次复数乘法即可（很多推理框架这么做）。

## 6. 相对性的实测：解析 vs 经验

从 $\rho(m)^\top\rho(n)=\rho(n-m)$ 可得（对随机 $q,k$）：

$$ \mathbb{E}\big[\langle q,k\rangle \cdot \langle q, \rho(\Delta)k\rangle\big] \big/ (\lVert q\rVert^2\lVert k\rVert^2) = \frac{2}{d}\sum_{i=0}^{d/2-1}\cos(\Delta\,\theta_i) $$

`ch03` 第 3 节把这个**解析曲线**和**蒙特卡洛估计**并列打印：

| $\Delta$ | 解析 | 经验 |
|---|---|---|
| 0 | 1.000 | 0.978 |
| … | … | … |
| 4096 | 0.0035 | 0.0147 |

两条曲线吻合 ⇒ 相对性确实成立；且随 $\Delta$ 增大相关性衰减（远程衰减，long-context 的常见观察）。

## 7. 长上下文外推：NTK-aware 与 YaRN

训练长度 $L_{train}$，想外推到 $L_{test} = s\, L_{train}$。

### NTK-aware scaling

直接改**基底**（高频拉伸得多、低频拉伸得少，等价于"非均匀插值"）：

$$ \mathrm{base}' = \mathrm{base}\cdot s^{\,d/(d-2)} $$

### YaRN（ramp 插值）

对不同频带用不同强度的插值，中间用斜坡过渡。设波长 $\lambda_i = 2\pi/\theta_i$，比值 $r = L_{train}/\lambda_i$：

$$ \gamma_i = \mathrm{clip}\!\left(\frac{r - \alpha}{\beta - \alpha},\ 0,\ 1\right),\qquad
\theta_i' = \begin{cases}\theta_i, & \gamma_i = 0\ (r\le\alpha)\\ \dfrac{\theta_i}{(1-\gamma_i)/s + \gamma_i}, & \text{otherwise}\end{cases} $$

直觉：

- $r \le \alpha$（波长**短**于训练长度 → 已见过足够多周期）→ 不插值，直接外推；
- $r \ge \beta$（波长**长**于训练长度 → 从未见过一个完整周期）→ 完全插值；
- 中间 → 斜坡过渡。

再加一个 attention 温度的修正（$t = 0.1\ln s + 1$）补偿熵的变化。

## 8. 一个反例：不正交会发生什么

`ch03` 第 6 节故意用一个**非正交**的"伪位置矩阵" $A(m)$（比如带上缩放的对角矩阵）替换 $\rho(m)$：

- $\langle A(m)q, A(n)k\rangle$ **不再**只依赖 $n-m$；
- $\lVert A(m) q\rVert \ne \lVert q\rVert$，点积尺度随位置漂移 → 等价于随位置变化的温度。

这解释了为什么 RoPE 一定要**正交**：正交是"不改变向量几何、只改变坐标系"的充要条件。

## 9. 运行

```bash
cd code && python ch03_rope_rotation.py
```

9 条 `[PASS]`。

## 练习

1. 推导 $\rho(m)^\top\rho(n) = \rho(n-m)$，注意 $\rho(m)^\top = \rho(-m)$。
2. 画出 $\theta_i = 10000^{-2i/d}$ 在 $d{=}128$ 时的 64 个频率，指出哪些平面在 $L{=}4096$ 内转够一圈。
3. 写交错布局的 `apply_rope`，并验证 `P^T R_int P == R_half`。
4. 复数形式实现 RoPE，比较与实数版的速度（提示：`view_as_complex` 要求最后一维 size=2 且连续）。

---

**下一章**：CH04 softmax 注意力——为什么除以 $\sqrt d$，以及掩码/GQA/KV Cache 的矩阵视角。
