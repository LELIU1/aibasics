"""
一次性跑通全部 11 章用例，并统计 [PASS]/[FAIL]。

用法:
    python run_all.py              # 跑全部
    python run_all.py 3 5 7        # 只跑第 3、5、7 章
    python run_all.py --quiet      # 只打印汇总

判定标准：任一章的 stdout 里出现 "[FAIL]" 或进程非 0 退出码，即判为该章失败。
"""
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(HERE, "ch*_*.py")


def discover():
    files = sorted(glob.glob(PATTERN))
    return [f for f in files if not os.path.basename(f).startswith("_")]


def run_one(path, quiet=False, timeout=1200):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "-u", path],
            cwd=HERE, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 1, "", f"TIMEOUT after {timeout}s", time.perf_counter() - t0
    return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0


def summarize(stdout):
    n_pass = len(re.findall(r"\[PASS\]", stdout))
    n_fail = len(re.findall(r"\[FAIL\]", stdout))
    fails = [l.strip() for l in stdout.splitlines() if "[FAIL]" in l]
    return n_pass, n_fail, fails


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv[1:]

    files = discover()
    if args:
        wanted = {int(a) for a in args}
        files = [f for f in files
                 if int(re.search(r"ch(\d+)_", os.path.basename(f)).group(1)) in wanted]
    if not files:
        print("没有匹配到任何章节文件")
        return 1

    results, all_fails = [], []
    for f in files:
        m = re.search(r"ch(\d+)_(.*)\.py", os.path.basename(f))
        ch, name = m.group(1), m.group(2)
        rc, out, err, dt = run_one(f, quiet)
        n_pass, n_fail, fails = summarize(out)
        ok = (rc == 0 and n_fail == 0)
        results.append((ch, name, ok, n_pass, n_fail, dt))
        if not ok:
            all_fails.append((ch, name, rc, fails, err.strip()[-1500:]))
        if not quiet:
            flag = "OK  " if ok else "FAIL"
            print(f"  [{flag}] CH{ch} {name:<32} PASS={n_pass:<3} FAIL={n_fail:<3} {dt:6.1f}s")
            for line in fails:
                print(f"         {line}")
            if rc != 0 and not err.endswith("\n"):
                pass
            if rc != 0:
                print(f"         exit={rc}\n{err.strip()[-1200:]}")

    print("\n" + "=" * 78)
    tot_p = sum(r[3] for r in results)
    tot_f = sum(r[4] for r in results)
    bad = [r for r in results if not r[2]]
    print(f"  章节 {len(results)} 个 | 断言 PASS {tot_p} / FAIL {tot_f} | "
          f"用时 {sum(r[5] for r in results):.1f}s")
    if bad:
        print(f"  未通过: {', '.join('CH' + r[0] + ' ' + r[1] for r in bad)}")
    else:
        print("  全部章节通过 ✅")
    print("=" * 78)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
