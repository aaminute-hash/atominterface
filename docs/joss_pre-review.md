# JOSS Pre-review 自检说明

本文件逐条对照 JOSS（Journal of Open Source Software）审稿 checklist，给出本仓库内的
对应证据，供 editor / reviewer 快速核验。所有路径相对于仓库根目录
`aaminute-hash/atominterface`（当前发布版 **v0.6.0**）。

---

## 1. Repository

**Checklist**: 软件源码可在仓库 URL 获取。

- ✅ 仓库：<https://github.com/aaminute-hash/atominterface>
- ✅ 发布版：<https://github.com/aaminute-hash/atominterface/releases/tag/v0.6.0>
- ✅ 源码在 `src/atominterface/`，安装元数据在 `pyproject.toml`

## 2. License

**Checklist**: 仓库根有纯文本 LICENSE，内容为 OSI 认可的软件许可证。

- ✅ `LICENSE` — **MIT License**（OSI approved）

## 3. Installation

**Checklist**: 按文档安装能成功。

- ✅ `README.md`「Installation」一节：`pip install -e .`（Python ≥ 3.12）
- ✅ `pyproject.toml` 声明依赖：`numpy`、`ase`、`typer`（核心）；`pymatgen`、
  `mace`、`chgnet` 为可选 extra（依赖范围冻结，见 `docs/roadmap.md` 依赖事实）

## 4. Functionality

**Checklist**: 软件的功能声明已被证实。

- ✅ `README.md` + `examples/` 五个可运行示例（`01_build_and_relax.py`、
  `02_md_stress_strain.py`、`03_interface_energy.py`、`05_float64_vs_float32.py`，
  以及本文档完成时新增的 `04_fenicsx_unit_cell.py`）
- ✅ 真实数据对拍：`docs/validation_md_virial_cauchy.md`、
  `docs/validation_md_elastic_constant.md`（Cu `C11 = 170.01 GPa`，与实验 168 GPa
  偏差 1.2%）
- ✅ 界面能交付值：Cu/Ni(001) `γ_int = 0.059 ± 0.003 J/m²`（VASP 平面波交叉验证）

## 5. Statement of need

**Checklist**: 作者清楚说明软件要解决的问题与目标用户。

- ✅ `paper.md` → `# Statement of need`：界面稳定性/粘附预测的「可靠性层」，
  目标用户为构建/评估界面的材料计算研究者

## 6. State of the field

**Checklist**: 作者说明本软件与同类软件的比较。

- ✅ `paper.md` → `# State of the field`：与 pymatgen / ASE / AiiDA / MACE / CHGNet
  的边界划分（「build vs. contribute」判断）

## 7. Quality of writing

**Checklist**: 论文结构、语言、写作质量无需编辑。

- ✅ `paper.md` 六段齐全（Summary / Statement of need / State of the field /
  Software design / Research impact statement / AI usage disclosure），正文 1106 词，
  落在 JOSS 2026 规范 750–1750 词区间

## 8. References

**Checklist**: 参考文献完整，应引尽引。

- ✅ `paper.bib` 6 条，DOI 均已核实，与正文 `@` 引用 6/6 一一对应：
  ASE [Larsen2017]、pymatgen [Ong2013]、MACE [Batatia2022]、CHGNet [Deng2023]、
  NumPy [Harris2020]、Zur–McGill [Zur1984]

## 9. Version

**Checklist**: 论文声明的版本与 GitHub Release 一致。

- ✅ `v0.6.0` tag + Release 已建（`paper.md` 与 `CITATION.cff` 均为 `0.6.0`）

## 10. Authors

**Checklist**: 作者均有 ORCID。

- ✅ 唯一作者 **Feng Qi**，ORCID `0009-0001-3074-8592`（`paper.md` YAML front
  matter + `CITATION.cff`）

## 11. Community guidelines

**Checklist**: 第三方有明确的贡献 / 提 issue / 求助渠道。

- ✅ `CONTRIBUTING.md`（贡献指南）
- ✅ `CODE_OF_CONDUCT.md`（Contributor Covenant 2.1）
- ✅ GitHub Issues / Discussions 开放

## 12. Automated tests

**Checklist**: 有自动化测试或手写步骤验证功能。

- ✅ 全量 pytest：**821 passed, 2 skipped**（`tests/`，含物理不变性、API surface、
  13 软件 reader 合成样例）
- ✅ 静态检查：`ruff check` / `ruff format --check` / `mypy` 全绿
- ✅ 持续集成：`.github/workflows/ci.yml`（Python 3.12 + 3.13）+ `release.yml`

## 13. AI usage disclosure（JOSS 2026 新规）

**Checklist**: 生成式 AI 的使用须如实披露。

- ✅ `paper.md` → `# AI usage disclosure`：LLM 起草 + 人工审核 + 821 测试 +
  逐位对拍验证，无未经独立计算核验的 AI 结论

---

## 提交后动作备忘

1. 在 <https://joss.theoj.org> 填投稿表单（需作者本人 GitHub 授权）。
2. editorialbot 在 `openjournals/joss-reviews` 开 pre-review issue 后，把本文件
   关键结论贴到 issue 里可加速 pre-review。
3. reviewer 意见返回后，逐条处理并 `@editorialbot` 更新。
