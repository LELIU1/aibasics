# 大模型线性代数 · 从 FlashAttention 到 DeepEP

一份**用 PyTorch 代码驱动**的线性代数教程：把 FlashAttention、GatedDeltaNet、RoPE、GroupGEMM、DeepEP
这些大模型核心技术背后的**线性代数结构**拆开讲清楚，每一章都配一份**可以直接跑、自带数值断言**的用例。

```
aibasics/
├── README.md              本文件（章节地图 + 运行方式）
├── index.html             单文件交互式教程站（打开即用，含 MathJax 公式渲染）
├── code/
│   ├── _utils.py          公共工具：check/timer/gemm_flops/bytes_of
│   ├── run_all.py         一键跑通全部 11 章并统计 PASS/FAIL
│   ├── ch01_tensors_strides_einsum.py
│   ├── ch02_gemm_fusion_associativity.py
│   ├── ch03_rope_rotation.py
│   ├── ch04_softmax_attention.py
│   ├── ch05_flash_attention.py
│   ├── ch06_linear_attention.py
│   ├── ch07_gated_deltanet.py
│   ├── ch08_groupgemm_moe.py
│   ├── ch09_deepep_communication.py
│   ├── ch10_lowrank_mla_quant.py
│   └── ch11_distributed_and_capstone.py
└── chapters/
    ├── ch01_tensors_strides_einsum.md   ...  ch11_distributed_and_capstone.md
```

## 快速开始

```bash
cd code
python ch05_flash_attention.py          # 跑单章
python run_all.py                       # 跑全部 11 章（约 40 秒）
python run_all.py 3 5 7                 # 只跑第 3/5/7 章
```

依赖只有 `torch`（CPU 即可，无需 GPU）。当前验证环境：Python 3.11 + PyTorch 2.10。

回归结果：**11 章 / 91 条断言 / 0 失败**。

## 章节地图

| 章 | 主题 | 核心数学 | 你会写出什么 |
|---|---|---|---|
| **01** | 布局 · stride · einsum · 算术强度 | stride 定义、einsum 维度记账、$\mathrm{AI}=\text{FLOPs}/\text{Bytes}$ | 用 einsum 表达 attention/外积/双线性型；算清 GEMM 与逐元素算子的 2800× 差距 |
| **02** | GEMM 经济学 | 结合律 $(AB)C=A(BC)$、对角/标量吸收、算子融合 | LoRA 权重合并；RMSNorm $\gamma$ 吸收；证明 $\mathrm{softmax}(QK^\top)V \ne Q(K^\top V)$ |
| **03** | RoPE | $SO(2)$ 旋转群、$\rho(m)^\top\rho(n)=\rho(n-m)$、置换共轭、NTK/YaRN | 两种布局的 RoPE；复数版；验证相对性解析曲线 |
| **04** | Softmax 注意力 | $\mathrm{Var}(q^\top k)=d$、LSE 稳定性、行随机矩阵、GQA 广播 | $\sqrt d$ 缩放实验；因果掩码；KV Cache 增量解码 |
| **05** | **FlashAttention** | 在线 softmax（$m/\ell/O$ 递推）、两级分块、IO 复杂度 $\Theta(N^2d^2/M)$ | 纯 PyTorch 版 FlashAttention（含梯度验证）；实测 813 MB → 43 MB |
| **06** | 线性注意力 | 状态矩阵 $S_t=\sum v_jk_j^\top$（秩1更新）、chunkwise、核特征映射 | 外积递推；带衰减的 chunkwise；FAVOR+；观察秩饱和 |
| **07** | **Gated DeltaNet** | $I-\beta kk^\top$ 的谱、**WY 表示**、三角求解、累积衰减 | 完整 chunkwise Gated DeltaNet，对比逐 token 循环，CPU 上 24.5× 加速 |
| **08** | **GroupGEMM / MoE** | $Y = P^\top\mathrm{blkdiag}(E_e)PX$、置换正交性、padding 分析 | 三种 MoE 实现互验；负载不均辅助损失 $f\cdot P\cdot E$ |
| **09** | **DeepEP** | All-to-All = 跨设备块置换、FP8 e4m3（次正规地板）、block-wise scaling | 量化的 per-tensor 失效 vs per-128 恢复；计算/通信比 7.17 |
| **10** | 低秩 · MLA · 量化 | **Eckart–Young 定理**、MLA 矩阵吸收、解耦 RoPE、Hessian 加权 | 截断 SVD 最优性验证；MLA 展开版 == 吸收版；INT8/4/3/2 误差对比 |
| **11** | 分布式 + 实战 | 列切/行切分块矩阵、Megatron MLP、**Ring Attention** | TP 通信量对比；跨设备在线 softmax；组装一个 MiniBlock 并出账单 |

## 建议的学习路线

```
入门路线（先建立直觉）
  01 → 02 → 04 → 05
进阶路线（吃透长上下文）
  01 → 02 → 04 → 05 → 06 → 07
系统路线（覆盖 MoE / 部署 / 并行）
  01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11
```

**只有半天时间？** 跑 `03 → 05 → 07`：这三个分别是"漂亮的代数结构"、
"最著名的工程优化"、"最硬核的现代递推"，覆盖了本教程的三种典型思维方式。

## 贯穿全书的三个万能技巧

1. **重缩放对齐** —— 不同块/设备的局部结果，靠一个标量因子 $e^{m_{old}-m_{new}}$ 对齐后累加。
   出现在 CH05（在线 softmax）、CH07（chunkwise 状态传递）、CH11（Ring Attention）。
2. **矩阵吸收** —— 把邻近的对角/标量/小矩阵预先乘进权重，运行时代价为零。
   出现在 CH02（$\gamma$、LoRA）、CH05（$\log_2 e$ 吸收进 $1/\sqrt d$）、CH10（MLA）。
3. **分块** —— 把 $O(N^2)$ 内存切成块，把串行递推变成"块内并行 + 块间传递"。
   出现在 CH01、CH05、CH06、CH07、CH08、CH11。

## 代码约定

- 每章都是**独立可执行**的脚本（`python chXX_*.py`），不依赖其它章。
- `check(name, a, b)` 打印 `[PASS]` 或 `[FAIL]` 以及 `max|Δ|`，所有数值结论都由它保证。
- `run_all.py` 统计每章的 `[PASS]/[FAIL]` 条数，出现 `[FAIL]` 或非 0 退出码即判失败。
- 全部代码 **CPU 可跑**，不需要 GPU / Triton / 分布式环境（CH05 附了 Triton kernel 伪代码只打印不执行；
  CH09/CH11 的通信用单进程模拟）。

## 一句话总结

> 大模型里所有的"魔法"，拆到底都是**矩阵怎么切、怎么乘、怎么搬**。
> 这本书就是把这句话对每个具体算子展开一遍。
