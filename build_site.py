#!/usr/bin/env python3
"""
把 chapters/*.md + code/ch*.py 打包成一个单文件交互教程站 index.html。

用法:  python build_site.py        （在 llm-linalg/ 目录下运行）
依赖:  pip install markdown
产出:  index.html（自包含，除 MathJax 走 CDN 外无外部依赖）
"""
import html
import os
import re

import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = os.path.join(ROOT, "chapters")
CODE = os.path.join(ROOT, "code")
OUT = os.path.join(ROOT, "index.html")

META = [
    ("01", "布局 · stride · einsum · 算术强度", "物理层：同一个矩阵在内存里怎么摆"),
    ("02", "GEMM 经济学", "结合律 / 矩阵吸收 / 算子融合"),
    ("03", "RoPE 旋转位置编码", "SO(2) 旋转群与相对位置"),
    ("04", "Softmax 注意力", "√d、LSE、掩码、GQA、KV Cache"),
    ("05", "FlashAttention", "在线 softmax + 分块 → O(N) 显存"),
    ("06", "线性注意力", "外积状态 / chunkwise / 秩饱和"),
    ("07", "Gated DeltaNet", "Δ rule / WY 表示 / 三角求解"),
    ("08", "GroupGEMM 与 MoE", "置换路由 + 块对角矩阵"),
    ("09", "DeepEP 通信", "All-to-All / FP8 / 计算通信重叠"),
    ("10", "低秩 · MLA · 量化", "Eckart–Young / 矩阵吸收 / GPTQ"),
    ("11", "分布式与综合实战", "张量并行 / Ring Attention / MiniBlock"),
]

MD = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])


def render_md(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # 去掉首行的 "# CHxx ..." 大标题（侧栏已提供导航，正文里保留 h2 层级更清晰）
    text = re.sub(r"^#\s+CH\d+[^\n]*\n", "", text, count=1)
    MD.reset()
    return MD.convert(text)


def esc_py(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def build():
    files = sorted(os.listdir(CHAPTERS))
    pairs = []
    for m in META:
        num = m[0]
        md_file = [f for f in files if f.startswith("ch" + num)]
        py_file = [f for f in os.listdir(CODE) if f.startswith("ch" + num) and f.endswith(".py")]
        if not md_file or not py_file:
            print(f"  [skip] CH{num}: 缺少 md 或 py")
            continue
        pairs.append((m, os.path.join(CHAPTERS, md_file[0]), os.path.join(CODE, py_file[0])))

    nav_items, sections = [], []
    for (num, short, sub), md_path, py_path in pairs:
        nav_items.append(
            f'<a class="nav-item" href="#ch{num}" data-ch="{num}">'
            f'<span class="nav-num">{num}</span>'
            f'<span class="nav-txt"><b>{html.escape(short)}</b><i>{html.escape(sub)}</i></span></a>'
        )
        sections.append(
            f'<section class="chapter" id="ch{num}">'
            f'<header class="ch-head"><span class="ch-num">CH{num}</span>'
            f'<h2>{html.escape(short)}</h2>'
            f'<div class="tabs">'
            f'<button class="tab active" data-tab="doc">讲义</button>'
            f'<button class="tab" data-tab="code">代码 · {html.escape(os.path.basename(py_path))}</button>'
            f"</div></header>"
            f'<div class="pane pane-doc active">{render_md(md_path)}</div>'
            f'<div class="pane pane-code"><pre><code class="language-python">{html.escape(esc_py(py_path))}</code></pre></div>'
            f"</section>"
        )

    readme = ""
    rp = os.path.join(ROOT, "README.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            readme = f.read()
        # 只取「快速开始」之后、章节地图之前的部分作为首页说明
        readme = re.sub(r"^#\s+[^\n]*\n", "", readme, count=1)
        MD.reset()
        readme = MD.convert(readme)

    doc = HTML_TEMPLATE.replace("<!--NAV-->", "\n".join(nav_items))
    doc = doc.replace("<!--SECTIONS-->", "\n".join(sections))
    doc = doc.replace("<!--README-->", readme)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  已生成 {OUT}  ({len(doc)/1024:.0f} KB, {len(pairs)} 章)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>大模型线性代数 · 从 FlashAttention 到 DeepEP</title>
<script>
  window.MathJax = {
    tex: { inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']],
           packages: {'[+]': ['ams','boldsymbol']} },
    options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code','annotation'] },
    startup: { typeset: true }
  };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root{
  --bg:#0d1117; --bg2:#161b22; --bg3:#1c2431; --line:#2a323d;
  --fg:#e6edf3; --fg2:#9aa7b4; --fg3:#6e7d8c;
  --accent:#58a6ff; --accent2:#7ee787; --warn:#f0883e; --pink:#ff7b72; --purple:#d2a8ff;
  --mono:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;
  font-size:15px;line-height:1.75}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

/* ---- layout ---- */
#sidebar{position:fixed;left:0;top:0;bottom:0;width:290px;background:var(--bg2);
  border-right:1px solid var(--line);overflow-y:auto;padding:18px 0;z-index:20}
#sidebar::-webkit-scrollbar{width:8px}
#sidebar::-webkit-scrollbar-thumb{background:#30363d;border-radius:4px}
.brand{padding:6px 20px 16px;border-bottom:1px solid var(--line);margin-bottom:10px}
.brand h1{margin:0;font-size:16px;letter-spacing:.3px}
.brand p{margin:4px 0 0;font-size:12px;color:var(--fg3);line-height:1.5}
.nav-item{display:flex;gap:11px;align-items:flex-start;padding:8px 20px;border-left:3px solid transparent}
.nav-item:hover{background:var(--bg3);text-decoration:none}
.nav-item.active{background:var(--bg3);border-left-color:var(--accent)}
.nav-num{font-family:var(--mono);font-size:12px;color:var(--fg3);padding-top:2px;min-width:20px}
.nav-item.active .nav-num{color:var(--accent)}
.nav-txt{display:flex;flex-direction:column;min-width:0}
.nav-txt b{font-size:13.5px;font-weight:600;color:var(--fg)}
.nav-txt i{font-size:11.5px;font-style:normal;color:var(--fg3);line-height:1.45;margin-top:1px}
.side-foot{padding:14px 20px;margin-top:10px;border-top:1px solid var(--line);font-size:11.5px;color:var(--fg3)}

#main{margin-left:290px;padding:0 0 120px}
.wrap{max-width:900px;margin:0 auto;padding:0 32px}

/* ---- hero ---- */
.hero{padding:52px 0 30px;border-bottom:1px solid var(--line);margin-bottom:8px}
.hero h2{margin:0 0 10px;font-size:30px;line-height:1.3;letter-spacing:-.3px}
.hero .lead{font-size:15.5px;color:var(--fg2);max-width:720px;margin:0 0 18px}
.badges{display:flex;flex-wrap:wrap;gap:8px}
.badge{font-size:12px;padding:4px 11px;border-radius:20px;background:var(--bg3);
  border:1px solid var(--line);color:var(--fg2);font-family:var(--mono)}
.badge.ok{border-color:#238636;color:var(--accent2)}
.badge.hot{border-color:#8957e5;color:var(--purple)}

/* ---- chapter ---- */
.chapter{scroll-margin-top:16px;padding:34px 0 10px;border-bottom:1px solid var(--line)}
.ch-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.ch-num{font-family:var(--mono);font-size:12px;color:var(--accent);
  border:1px solid var(--accent);border-radius:5px;padding:2px 8px}
.ch-head h2{margin:0;font-size:23px;letter-spacing:-.2px}
.tabs{margin-left:auto;display:flex;gap:6px}
.tab{background:var(--bg3);border:1px solid var(--line);color:var(--fg2);
  padding:5px 13px;border-radius:6px;font-size:12.5px;cursor:pointer;font-family:inherit}
.tab:hover{color:var(--fg)}
.tab.active{background:var(--accent);border-color:var(--accent);color:#06121f;font-weight:600}
.pane{display:none}
.pane.active{display:block}

/* ---- markdown body ---- */
.pane-doc h3{margin:30px 0 10px;font-size:18px;color:var(--fg);
  border-left:3px solid var(--accent);padding-left:10px}
.pane-doc h4{margin:22px 0 8px;font-size:15.5px;color:var(--accent2)}
.pane-doc p{margin:11px 0}
.pane-doc ul,.pane-doc ol{padding-left:24px;margin:11px 0}
.pane-doc li{margin:5px 0}
.pane-doc blockquote{margin:16px 0;padding:10px 16px;background:var(--bg2);
  border-left:3px solid var(--warn);color:var(--fg2);border-radius:0 6px 6px 0}
.pane-doc blockquote p{margin:4px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left}
th{background:var(--bg3);font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:#11161d}
code{font-family:var(--mono);font-size:12.8px;background:var(--bg3);
  padding:2px 6px;border-radius:4px;color:var(--purple)}
pre{background:#0b0f14;border:1px solid var(--line);border-radius:9px;
  padding:14px 16px;overflow-x:auto;margin:14px 0;line-height:1.6}
pre code{background:none;padding:0;color:var(--fg);font-size:12.6px}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
.pane-doc strong{color:#fff}

/* ---- python highlight ---- */
.k{color:#ff7b72}.s{color:#a5d6ff}.c{color:#6e7d8c;font-style:italic}
.n{color:#79c0ff}.d{color:#d2a8ff}.f{color:#d2a8ff}.b{color:#ffa657}

/* ---- code pane ---- */
.pane-code pre{max-height:78vh;overflow:auto}
.pane-code .ln{color:var(--fg3);user-select:none;display:inline-block;width:2.4em;text-align:right;margin-right:1em}

/* ---- misc ---- */
.to-top{position:fixed;right:22px;bottom:22px;background:var(--bg3);border:1px solid var(--line);
  color:var(--fg2);width:38px;height:38px;border-radius:9px;cursor:pointer;font-size:16px;display:none}
.to-top.show{display:block}
@media(max-width:900px){
  #sidebar{display:none}
  #main{margin-left:0}
  .wrap{padding:0 18px}
}
</style>
</head>
<body>

<nav id="sidebar">
  <div class="brand">
    <h1>大模型线性代数</h1>
    <p>从 FlashAttention 到 DeepEP<br>11 章 · 91 条数值断言 · 全部可跑</p>
  </div>
  <!--NAV-->
  <div class="side-foot">
    <div>依赖：仅 torch（CPU 即可）</div>
    <div style="margin-top:4px"><code>python run_all.py</code></div>
  </div>
</nav>

<main id="main"><div class="wrap">
  <div class="hero">
    <h2>大模型里的线性代数，一次拆到底</h2>
    <p class="lead">把 FlashAttention、GatedDeltaNet、RoPE、GroupGEMM、DeepEP 背后真正的数学结构
    —— 旋转群、外积状态、WY 表示、块对角矩阵、低秩吸收 —— 逐个展开，
    每一章配一份<strong>自带数值断言</strong>的 PyTorch 用例。</p>
    <div class="badges">
      <span class="badge ok">run_all: 11/11 PASS · 91 断言</span>
      <span class="badge">PyTorch · CPU 可跑</span>
      <span class="badge hot">CH05 Flash · CH07 DeltaNet</span>
      <span class="badge">约 40 秒跑完全部</span>
    </div>
  </div>

  <!--SECTIONS-->

  <section class="chapter" id="readme">
    <header class="ch-head"><span class="ch-num">README</span><h2>目录、运行方式与学习路线</h2></header>
    <div class="pane pane-doc active"><!--README--></div>
  </section>
</div></main>

<button class="to-top" id="toTop">↑</button>

<script>
/* ---------- 极简 Python 高亮 ---------- */
const PY_RE = /(#[^\n]*)|((?:'{3}|"{3})[\s\S]*?(?:'{3}|"{3}))|('(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*")|(@[A-Za-z_]\w*)|\b(def|class|return|yield|if|elif|else|for|while|break|continue|pass|import|from|as|with|try|except|finally|raise|assert|lambda|global|del|not|and|or|in|is|None|True|False|self|print)\b|\b(0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)\b/g;
function hlPy(src){
  // textContent 拿到的是已解码的原文，写回 innerHTML 前必须先转义
  src = src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return src.replace(PY_RE, (m, com, tri, str, dec, kw, num) => {
    if (com !== undefined) return '<span class="c">' + com + '</span>';
    if (tri !== undefined) return '<span class="s">' + tri + '</span>';
    if (str !== undefined) return '<span class="s">' + str + '</span>';
    if (dec !== undefined) return '<span class="d">' + dec + '</span>';
    if (kw  !== undefined) return '<span class="k">' + kw + '</span>';
    if (num !== undefined) return '<span class="n">' + num + '</span>';
    return m;
  });
}
document.querySelectorAll('pre > code').forEach(el => {
  const raw = el.textContent;
  if (el.className.includes('language-python')) el.innerHTML = hlPy(raw);
});

/* ---------- 代码行号 ---------- */
document.querySelectorAll('.pane-code pre code').forEach(el => {
  const lines = el.innerHTML.split('\n');
  el.innerHTML = lines.map((l,i) => '<span class="ln">' + (i+1) + '</span>' + l).join('\n');
});

/* ---------- 讲义 / 代码 切换 ---------- */
document.querySelectorAll('.tabs').forEach(tabs => {
  const sec = tabs.closest('.chapter');
  tabs.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sec.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
      sec.querySelector('.pane-' + btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'code' && window.MathJax) window.MathJax.typesetPromise();
    });
  });
});

/* ---------- 侧栏高亮 ---------- */
const items = [...document.querySelectorAll('.nav-item')];
const secs  = items.map(a => document.getElementById('ch' + a.dataset.ch)).filter(Boolean);
function onScroll(){
  let idx = 0, y = window.scrollY + 120;
  secs.forEach((s,i) => { if (s && s.offsetTop <= y) idx = i; });
  items.forEach(a => a.classList.remove('active'));
  if (items[idx]) {
    items[idx].classList.add('active');
    const nav = document.getElementById('sidebar');
    const r1 = items[idx].getBoundingClientRect(), r2 = nav.getBoundingClientRect();
    if (r1.top < r2.top || r1.bottom > r2.bottom)
      nav.scrollTop += r1.top - r2.top - 80;
  }
  document.getElementById('toTop').classList.toggle('show', window.scrollY > 500);
}
window.addEventListener('scroll', onScroll, {passive:true});
onScroll();
document.getElementById('toTop').onclick = () => window.scrollTo({top:0,behavior:'smooth'});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
