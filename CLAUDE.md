# AtomInterface 开发纪律（Claude Code 每次会话自动读取）

Python 库：相干界面构建 · uMLIP 带护栏弛豫 · 界面热力学 · CZM 导出。
源码 `src/atominterface/`，测试 `tests/`，文档 `docs/`。GitHub: aaminute-hash/atominterface，提交身份 `aaminute-hash <aaminute@gmail.com>`。
当前版本/发布状态看 `git log` 与 `CHANGELOG.md`，不在此维护快照。

## 铁律（违反即返工）
1. 不改 LICENSE、不改 pyproject.toml 依赖范围、不动 `src/atominterface/relax/`（守卫层）
2. 改仓库任何文本一律 `open(..., newline="")` 读写，禁止 `Path.read_text()/write_text()`（否则整文件被重写成 CRLF）；归档 CSV 保持 CRLF + UTF-8 无 BOM
3. Conventional Commits，每项修复单独提交；改动前 `git branch` 确认分支，新任务先开新分支
4. pytest 必带 `--basetemp=F:/tmp/pt-xxx`（沙箱 safe-delete 钩子会中断批量 tmp 清理造成假失败）
5. 等价性结论必须带"仅同环境成立"限定；坐标基线不要硬编码（pymatgen/numpy 版本敏感），用同进程 A/B + `array_equal` 证明
6. VASP 能量差值统一用 `energy(sigma->0)`，不混用 `free energy TOTEN`
7. 分数坐标换算：先取分数坐标 → 回绕 → 再乘回笛卡尔；判据是实测落点不是能量
8. 旋转约定已统一（0.5.2 起）：`rotation_matrix` 返回与 ASE 同手性的标准右手矩阵；`rotate_positions`/`transform_forces` 用行向量 `X @ R.T`。ASE 旋转可安全喂入，勿回滚
9. 禁止用 extxyz 文件哈希当回归闸门（弛豫产物跨运行 ~5e-6 Å 抖动）；比未弛豫 built 结构或能量/γ 数值差
10. git 传输可能被代理挡（502/schannel）→ 推送/建 PR 走 GitHub REST API
11. 不臆造哈希/数值：MACE 权重锚点等只能取自真实下载文件的计算值，取不到就置 None 并注明
12. 密钥不入库：含 API Key 的本地脚本一律进 `.gitignore`；提交前 `git diff --cached` 自查无 `sk-`
13. `__version__` 与 pyproject 版本号同步 bump；发版前 `grep -n version pyproject.toml src/atominterface/__init__.py` 双点确认

## 常用命令（照抄 CI 口径，勿用 `.`，否则 docs 代码块会被纳入报无关失败）
```bash
# 测试
python -m pytest --basetemp=F:/tmp/pt-xxx
# lint / type
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src/atominterface
# CLI smoke
atominterface build examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp --backend dummy --outdir outputs/smoke
```

## 开发约定优先级
`claude_code_续修指令.md`（仓库上级目录）> `software-atominterface-docs/PRD-*.md` > `docs/roadmap.md`
