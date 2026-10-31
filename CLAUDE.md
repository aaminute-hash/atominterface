# AtomInterface 开发纪律（供 Claude Code 每次会话自动读取）

## 项目概况
- Python 库：相干界面构建 · uMLIP 带护栏弛豫 · 界面热力学 · CZM 导出
- 源码 `src/atominterface/`，测试 `tests/`，文档 `docs/`
- GitHub: aaminute-hash/atominterface；提交身份 `aaminute-hash <aaminute@gmail.com>`
- 开发约定优先级：`claude_code_续修指令.md`（仓库上级目录）> `software-atominterface-docs/PRD-*.md` > `docs/roadmap.md`

## 铁律（违反即返工）
1. **不改 LICENSE、不改 pyproject.toml 依赖范围、不动 `src/atominterface/relax/`（守卫层）**
2. **换行纪律**：修改仓库任何文本一律 `open(..., newline="")` 读写（读写都要），禁止 `Path.read_text()/write_text()`——否则整文件被重写成 CRLF
3. **提交纪律**：Conventional Commits，每项修复单独提交；改动前 `git branch` 确认分支，不要在 `feat/v0.5.1-increment`（待推送）上直接改，新任务先开新分支
4. **pytest 必带** `--basetemp=F:/tmp/pt-xxx`（沙箱 safe-delete 钩子会中断批量 tmp 清理造成假失败）
5. 归档 CSV 是 UTF-8 无 BOM + CRLF，sha256 按字节计算，新 CSV 必须照抄风格
6. 等价性结论必须带"仅同环境成立"限定；坐标基线不要硬编码（pymatgen/numpy 版本敏感），用同进程 A/B + `array_equal` 证明
7. VASP 能量差值统一用 `energy(sigma->0)`，不混用 `free energy TOTEN`
8. 分数坐标换算：先取分数坐标 → 回绕 → 再乘回笛卡尔；判据是实测落点不是能量
9. **旋转约定（0.5.2 起已统一）**：`rotation_matrix(axis,θ)` 现在返回与 ASE 同手性的标准右手矩阵（`R @ v` 列向量语义）；`rotate_positions`/`transform_forces`/`transform_stress_voigt` 用行向量 `X @ R.T`。**ASE 的旋转（如 ZSL `match.rotation`）现在可以安全喂入这些 helper**。旧版（≤0.5.1）因四元数向量负号返回 R(−θ)，与 ASE 手性相反——若看到旧代码里手动转置 `R.T` 补偿，那是为绕开该 bug，应予删除
10. **禁止用 extxyz 文件哈希当回归闸门**：弛豫产物跨运行不可逐位复现（float32+多线程，~5e-6 Å 抖动）；要证"改动 X 不影响 Y"比未弛豫 built 结构（逐位 0.000）或能量/γ 数值差
11. git 传输可能被代理挡（502/schannel）→ 推送/建 PR 走 GitHub REST API（`git credential fill` 取 token）
12. 不臆造哈希/数值：MACE 权重锚点等只能取自真实下载文件的计算值，取不到就置 None 并注明
13. **密钥不入库**：`claude-local.sh` 等含 API Key 的本地脚本一律写进 `.gitignore`；提交前 `git diff --cached` 自查，任何 `sk-` 前缀字符串都不得进入暂存区
14. **`__version__` 与 `pyproject.toml` 版本号必须同步 bump**：CHANGELOG 出现新版本标题 ≠ 版本号已改，发版前 `grep -n version pyproject.toml src/atominterface/__init__.py` 双点确认

## 常用命令
```bash
# 测试（PYTHONPATH=src 或 pip install -e .）
python -m pytest --basetemp=F:/tmp/pt-xxx
# lint / type
python -m ruff check .
python -m mypy src
# CLI smoke
atominterface build examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp --backend dummy --outdir outputs/smoke
```

## 当前状态（2026-10-31）
- main @ c8ca185 = **v0.5.0 已发布**（tag 在位、Release 附件 2 个；`__version__="0.5.0"`）；PR #36（quickstart 0.5.0 修正）远端在途
- 分支 `feat/v0.5.1-increment` **8 commits** 待推送（R1-R9 修复，`b4e96b1..1301206`）；**CHANGELOG 写了 0.5.1 但 `__version__` 与 pyproject 仍是 0.5.0，发版前必须补 chore(release) 提交**
- MACE 权重 SHA-256 锚点待填（见 `backends/mace_.py` 常量旁注释）
