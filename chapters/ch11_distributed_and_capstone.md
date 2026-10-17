# CH11 分布式线性代数 + 综合实战：张量并行、Ring Attention 与组装一个小模型

> 代码：`code/ch11_distributed_and_capstone.py`
> 前置：CH01–CH10 全部

## 1. 张量并行：把权重矩阵按行/列切开

一切分布式并行的数学本质都是**分块矩阵**。

设 $X\in\mathbb{R}^{T\times d_{in}}$，$W\in\mathbb{R}^{d_{in}\times d_{out}}$，$Y = XW$。

### 列切分（Column Parallel）

把 $W$ 按**输出维**切成 $W = [W_1\ \cdots\ W_n]$，每块 $(d_{in},\ d_{out}/n)$：

$$ Y = X\,[W_1\ \cdots\ W_n] = [XW_1\ \cdots\ XW_n] $$

- 每卡需要**完整的 $X$**；
- 输出天然是拼接 ⇒ **零通信**（输入若已复制）。

### 行切分（Row Parallel）

把 $W$ 按**输入维**切成 $W = [W_1; \cdots; W_n]$，每块 $(d_{in}/n,\ d_{out})$：

$$ Y = [X_1\ \cdots\ X_n]\begin{bmatrix}W_1\\ \vdots\\ W_n\end{bmatrix} = \sum_i X_i W_i $$

- 每卡只需要**对应的 $X_i$**（$X$ 按特征维切好）；
- 输出是部分和 ⇒ 需要一次 **all-reduce**。

> ⚠️ 本教程踩过的坑：行切分时如果**只切 $W$ 不切 $X$**，`x @ w` 会直接报
> `mat1 and mat2 shapes cannot be multiplied (64x256 and 64x512)`。
> 记住：行切要求输入同步按特征维切分 —— 这正是它能**直接接在列切后面**的原因。

### Megatron 的 MLP 并行

MLP：$Y = \mathrm{GeLU}(XA)\,B$，$A:(d,\ 4d)$，$B:(4d,\ d)$。

$$ \begin{aligned} A &\xrightarrow{\text{列切}} [A_1\ \cdots\ A_n] &&\Rightarrow XA = [XA_1\ \cdots\ XA_n]\ \text{（无需通信）}\\ B &\xrightarrow{\text{行切}} [B_1; \cdots; B_n] &&\Rightarrow \text{每卡算 } \mathrm{GeLU}(XA_i)B_i\\ Y &= \sum_i \mathrm{GeLU}(XA_i)B_i &&\Rightarrow \text{最后 all-reduce 一次} \end{aligned} $$

**为什么 GeLU 不用通信？** 因为 GeLU 是**逐元素**非线性，它要求"被切的那一维是完整的"。
$A$ 按列切 ⇒ $XA$ 按列分成 $n$ 块，**每块内部完整** ⇒ GeLU 可以就地算。

反过来（$A$ 行切、$B$ 列切）：$XA$ 是部分和，要先 all-reduce 才能过 GeLU，
之后 $B$ 列切的输出天然拼接 —— **通信次数同样是 1 次**，但 all-reduce 的是中间激活
$T\times 4d$，量从 $T\times d$ 变成 **4 倍**（见下表：128.0 KB vs 32.0 KB）。

`ch11` 实测通信量（$T{=}64, d{=}256$，bf16）：

| 方案 | 通信量 |
|---|---|
| 一次 all-reduce（输出 $T\times d$） | 32.0 KB |
| 一次 all-reduce（中间激活 $T\times 4d$） | 128.0 KB |

**差 4 倍** —— 这就是 Megatron 选"列切 + 行切"的原因。

## 2. Ring Attention：在线 softmax 的分布式版本

CH05 的在线 softmax 用在**块**上；Ring Attention 把它用在**设备**上。

设定：$W$ 个设备，设备 $i$ 持有 $Q_i, K_i, V_i$（各自的 $\frac{N}{W}$ 行）。

```
for step in range(W):
    src = (i - step) % W          # 设备 i 本轮收到来自 src 的 K/V
    K_j, V_j = K[src], V[src]
    S = Q_i K_j^T / sqrt(d)       # (N/W, N/W) —— 永远是这么大
    if causal: mask 掉 k 的全局位置 > q 的全局位置
    在线 softmax 更新 (m, l, O)
O = O / l
```

**关键**：每个设备上的注意力矩阵永远只有 $(N/W)^2$ 大小 —— 对固定 $W$ 它是全局
$N^2$ 的 $1/W^2$（**仍随 $N$ 二次增长**）。只有当保持 $N/W$ 不变、即设备数随 $N$
同比增加时，单卡显存才与总长无关。

`ch11` 验证：

| 检查 | `max|Δ|` |
|---|---|
| 非因果 Ring == 全局注意力 | 2.98e-07 |
| 因果 Ring == 全局因果注意力 | 3.58e-07 |

### 因果时的负载不均

设备 $i$ 需要来自设备 $0..i$ 的 K/V ⇒ 设备 0 只读 1 块，设备 $W-1$ 要读全部 $W$ 块。
平均利用率只有约 50%。

**解决：ZigZag / striped 分区** —— 把序列块按 "$i$ 与 $W-1-i$ 配对"重排
（设备 $i$ 分到第 $i$ 块和第 $W-1-i$ 块），使每个设备的有效步数均衡。

### 通信量

每步传 $\frac{N}{W}\cdot d$ 的 K/V，共 $W$ 步 ⇒ 总 $\approx N\cdot d$，
与单卡 KV 总量同阶 —— **不随 $N^2$ 增长**，这是它能做超长上下文的原因。

## 3. 综合实战：MiniBlock

`ch11` 的 `MiniBlock` 把前 10 章的组件全用上了：

```
x ─→ RMSNorm(γ) ─→ QKV 投影 ─→ RoPE ─→ GQA 广播 ─→ 分块因果注意力 ─→ W_O ─→ + ─→ RMSNorm ─→ SwiGLU ─→ +
```

| 组件 | 章节 | 线性代数要点 |
|---|---|---|
| RMSNorm 的 $\gamma$ | CH02 | 对角吸收进 $W_Q/W_K/W_V$（列缩放） |
| RoPE | CH03 | $d_h/2$ 个旋转平面，半分布局 |
| GQA 广播 | CH04 | `expand` 零拷贝，KV Cache 省 $H/H_{kv}$ 倍 |
| 分块因果注意力 | CH05 | 在线 softmax + 两级分块 |
| SwiGLU MLP | CH02 | 三个 GEMM |
| 参数/FLOPs 账单 | CH01 | 算术强度记账 |

**验证**：`MiniBlock 的分块注意力 == 朴素实现`，`max|Δ| = 5.96e-08`。

### 单序列账单（$N{=}256, d_{model}{=}128$）

| 项 | 数值 |
|---|---|
| 参数量 | 139.5 K |
| Attention FLOPs | 33.6 MFLOP（$O(N^2)$ 部分） |
| 投影 FLOPs | 21.0 MFLOP |
| MLP FLOPs | 50.3 MFLOP |
| KV Cache (GQA) | 32.0 KB/序列 (bf16) |

**最要紧的一句话**：把 $N$ 从 256 换成 128K，attention 项涨 $(128\text{K}/256)^2 = 512^2 = 26.2$ 万倍，
其它项只涨 500 倍。**长上下文的一切问题都是 attention 的 $O(N^2)$ 问题。**

## 4. 全书回顾：一条主线

```
CH01 布局/强度  ──→  知道瓶颈在访存
CH02 GEMM 经济  ──→  结合律、吸收、融合
CH03 RoPE       ──→  正交群 = 相对位置
CH04 注意力     ──→  行随机算子、√d、掩码、GQA
CH05 Flash      ──→  在线 softmax + 分块 ⇒ O(N) 显存
CH06 线性注意力 ──→  外积状态 ⇒ O(Nd²)，代价：秩饱和
CH07 DeltaNet   ──→  擦除 + WY 表示 + 三角求解 ⇒ 可并行
CH08 GroupGEMM  ──→  MoE = 置换 + 块对角
CH09 DeepEP     ──→  跨设备置换 = all-to-all；FP8 block-wise
CH10 低秩/MLA   ──→  Eckart–Young + 矩阵吸收 ⇒ 压 KV Cache
CH11 分布式     ──→  分块矩阵放到多设备；Ring = 跨设备在线 softmax
```

反复出现的三个"万能技巧"：

1. **重缩放对齐**（CH05 在线 softmax / CH07 chunkwise / CH11 Ring）：
   不同块/设备的局部结果，靠一个标量因子 $e^{m_{old}-m_{new}}$ 对齐到统一基准后累加。
2. **矩阵吸收**（CH02 γ、CH02 LoRA、CH10 MLA、CH05 的 $\log_2 e$）：
   把邻近的对角/标量/小矩阵预先乘进权重，运行时代价为零。
3. **分块**（CH01 tiling、CH05、CH06、CH07、CH08、CH11）：
   把 $O(N^2)$ 的内存切成 $O(b^2)$ 的块，把串行递推变成"块内并行 + 块间传递"。

## 5. 运行

```bash
cd code && python ch11_distributed_and_capstone.py
```

6 条 `[PASS]`。

## 练习

1. 证明列切 + concat 与 $XW$ 恒等；行切 + sum 与 $XW$ 恒等（写出分块矩阵）。
2. 若 world=4，$N=4096$，$d=128$，bf16，Ring Attention 每步传多少字节？总通信量？
3. 把 `ring_attention` 改造成 ZigZag 分区，统计每个设备的有效计算步数，验证均衡性。
4. 把 `MiniBlock` 的 RMSNorm $\gamma$ 真正吸收进权重，验证输出不变（CH02）。
5. 给 `MiniBlock` 加一个 MoE 层（CH08），并计算它的负载不均衡损失。

---

**恭喜完成全部 11 章。** 建议下一步：挑一个你工作中真正用到的算子，
用 CH01 的三件套（stride / einsum / 算术强度）自己重新分析一遍。
