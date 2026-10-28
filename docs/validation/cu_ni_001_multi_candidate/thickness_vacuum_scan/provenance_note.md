# 归档 provenance 注记（厚度/真空收敛扫描）

本目录的 `README.md`、`thickness_vacuum_scan.csv`、`aggregate_scan.py`、
六个组目录与六个 `*.log` 均为 `F:/tmp/d1_thickness` 的**原样复制**，
内容未做任何改写。

## 运行环境注记

这批运行于 worktree **`F:/tmp/wt-thick`**（detached HEAD `e12ed58`，
origin/main，PR #16/#17/#18 合并后），通过
`PYTHONPATH=F:/tmp/wt-thick/src` 显式使用该 worktree 的源码——
当时 venv 里 atominterface 的 editable 安装路径指向主仓库的旧位置，
因此必须用 PYTHONPATH 指向 wt-thick，**不是从安装版跑的**。
各组完整命令行见 `README.md` §1。

## 数据校验

六个组的 `candidates.csv` SHA-256 均与各自 `metadata.yaml` 内记录的值
一致（入库时逐组复核通过）。机器可读汇总
`thickness_vacuum_scan.csv` 由 `aggregate_scan.py` 从各组
`candidates.csv` 聚合生成（28 行 = 7 组 × 4 候选，仅 status=ok 行；
厚度与真空扫描共用同一 14.4 Å + 15 Å 基准组，故为 6 个目录）。
