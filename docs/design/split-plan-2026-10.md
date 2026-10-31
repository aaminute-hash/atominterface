# 设计：三个巨石文件的模块拆分边界（R10 预研）

**状态**：**仅预研，不落地**。本文是 `PRD-increment-2026-10-31` §4 R10 的拆分边界记录，
供下周期立项评审。本次不修改任何现有源码，只产出本设计文档。

**范围**：`src/atominterface/builders/coherent.py`、`src/atominterface/builders/charge.py`、
`src/atominterface/cli_build.py`。实测行数与本周期立项时的估计不同（立项说
1509/932/865，`wc -l` 实测 **2753/1027/970**），但拆分结论不受影响，只会让理由更充分。

**铁律约束**（`CLAUDE.md`，拆分落地时必须逐条遵守）：

1. 不改 `LICENSE`、不改 `pyproject.toml` 依赖范围、不动 `src/atominterface/relax/`。
2. 换行纪律：改任何文本用 `open(..., newline="")` 读写。本仓库 `.md` 文档已全部为 LF
   （实测 `docs/design/registry_field.md`、`docs/roadmap.md` 等 CRLF=0），新建 `.md` 保持 LF；
   归档 CSV 仍是 UTF-8 无 BOM + CRLF，拆分不得触碰。
   〔补充核实 2026-10-31（按 git blob 二进制统计，非 `grep`——本机 `grep -c $'\r'` 会匹配空模式
   给出全命中假象）：22 个 `.md` 中 19 个纯 LF、3 个归档验证目录 README 纯 CRLF；连同
   `.py`/`.toml`/`.yml`/`.sh`/`.csv` 共 87 个文本文件，71 个纯 LF、16 个纯 CRLF 且 16 个全部是
   归档 CSV，**无一混合换行**。即 `.py`/`.md`/`.toml` 一律 LF，只有归档 CSV 是 CRLF。〕
3. 每步单独提交（Conventional Commits），先确认分支。`feat/v0.5.1-increment` 已于 2026-10-31
   作为 v0.5.1 合并进 `main`（v0.5.2 随后发布），落地时从当前 `main` 开 `refactor/split-*` 分支。

---

## 1. 三个文件的现状与职责边界

### 1.1 `coherent.py`（2753 行）

一块石头装了五件互不相干的事。按行号切成下面的职责块（行号为当前 HEAD 实测）：

| 行号区间 | 职责 | 关键符号 |
|---|---|---|
| 1–169 | 模块 docstring、导入、`GeometryGateError`、模块级 pymatgen 绑定、一堆容差常量 | `_PYMTG_COHERENT_BUILDER`、`_PYMTG_STRUCTURE`、`ON_TOP_LATERAL_TOL`、`REGISTRY_PLACEMENT_TOL`、`PUBLISHED_MILLER`、`AREA_CONSERVATION_RTOL` 等 |
| 172–272 | 面向用户的告警文案 | `_describe_registry`、`_warn_unscanned_orientation` |
| 275–436 | 配置数据类 | `CoherentInterfaceConfig`（`validate` 依赖 `validate_charge_options` + `_registry_errors` + `AREA_POLICIES`） |
| 439–589 | 面积账本 | `_MatchInfo`、`_lattice_area`、`AREA_POLICIES`、`InterfaceAreas`、`interface_areas`、`_in_plane_vectors` |
| 592–609 | 晶格归一化 | `_as_pymatgen_lattice` |
| 612–766 | 电荷预处理 + 切板 | `_prepare_charge_structure`、`_get_terminations`、`_flatten_terminations`、`_make_slabs`、`_symmetric_terminations` |
| 769–1028 | 超胞/应变/晶格匹配 | `_supercell_transform`、`_multiplicity`、`_format_matrix`、`_match_lattices`、`_area_ratio_ok`、`_strain_of`、`_fallback_matches`、`_dense_transform`、`_scale_in_plane` |
| 1064–1112 | 膜应变 | `_strain_film_onto` |
| 1115–1235 | 几何闸门 B/C | `_shortest_inplane_period`、`_pair_label`、`_check_area_conservation`、`_check_inplane_period` |
| 1238–1434 | 跨界面距离 + 堆叠 | `_RegistryPair`、`_closest_cross_pair`、`_warn_interface_proximity`、`_prepare_stack`、`_atomic_planes` |
| 1437–1899 | 注册位点放置（v0.5.0 全部新逻辑） | `_off_on_top_shift`、`_wrap_frac`、`_in_plane_frac`、`_join_lateral`、`_stacking_step`、`_stacking_period`、`_sites_of_step`、`_registry_target_frac`、`_registry_shift`、`_check_registry_placement`、`_registry_errors`、`_registry_as_json`、`_frac_or_none`、`_registry_lateral_ang`、`_place_join`、`_record_registry_properties` |
| 1902–2050 | 组装 | `_assemble_interface`、`_attach_charge_geometry` |
| 2053–2113 | 转换 + 电荷诊断文案 | `_reduced_formula`、`_to_ase`、`_warn_charge_diagnostics` |
| 2116–2396 | Candidate 组装 + 参考板 | `_to_candidate`、`measure_registry`、`_select_termination`、`reference_slabs` |
| 2399–2753 | 构建报告 + 构建器类 + 注册 | `_note_assembly_reason`、`CoherentInterfaceBuilder`、`@register_builder("coherent")`、末尾 eager 注册 |

**职责边界结论**：coherent.py 是"构建器本体 + 面积账本 + 晶格匹配 + 几何闸门 +
注册位点 + 组装/参考板"六个域，外加贯穿全局的 `_to_ase`/`_reduced_formula`/`_pair_label` 工具。
其中**注册位点域（~460 行，1437–1899）是最该先切出来的**——它是 v0.5.0 新增、语义自洽
（"先定点位再解位移"）、且 roadmap 已预留 `registry="scan"`（R11）要在这上面扩展；切出来
能让 R11 的改动落点可预期。

### 1.2 `charge.py`（1027 行）

| 行号区间 | 职责 | 关键符号 |
|---|---|---|
| 1–55 | docstring、常量、`_GeometryDefault` | `ANALYSIS_CHARGE`、`CHARGE_KNOWN`、`CHARGE_SOURCE`、`SLAB_ID`、`_POLICIES`、`_SOURCES`、`_GROUPS`、`_MAX_THICKNESS_PASSES` |
| 58–192 | 电荷数据解析（单一来源、不补齐） | `_real_number`、`_vector`、`_numeric_data`、`_validate_symbols`、`_mapping_data`、`_stored_data`、`resolve_charge_data`、`validate_charge_options` |
| 195–249 | 结构适配（ASE/pymatgen） | `_structure_data`、`_resolve_structure_charges` |
| 252–492 | 偶极/极性分析 | `_geometry_options`、`_vacuum_interval`、`_normal_geometry`、`_unwrap_periodic`、`_prepare_geometry`、`_group_masks`、`_finite_sum`、`_group_analysis`、`analyze_charge_polarity` |
| 495–580 | 策略解释 helper | `charge_rejection_reasons`、`is_potentially_polar`、`check_neutral_slabs`、`suggest_dipole_correction`、`tasker_classification` |
| 583–629 | 对称板数据类 + 有限板几何 | `SymmetricSlab`、`finite_slab_geometry` |
| 632–798 | 对称板机械（切/镜像/截断） | `_select_sites`、`mirror_slab`、`_truncate_layers`、`_as_structure_copy`、`_normal_axis`、`_normal_extent`、`_mark_generated`、`_site_dependent_charge`、`termination_cuts`、`_oriented_repeat_height` |
| 801–856 | 对称板选项/择优 | `_prefer_stoichiometric_termination`、`_symmetric_options`、`_select_termination_index`、`_symmetric_board_notes` |
| 859–1027 | 定向 + 对称板入口 | `_ORIENT_TOL`、`_orient_slab`、`make_symmetric_slab` |

**职责边界结论**：charge.py 天然分成两半——(A) **电荷数据解析 + 偶极/极性分析**（纯数值、
无 pymatgen 结构生成，只读结构）；(B) **对称板/终止面构造**（重度依赖 pymatgen
`SlabGenerator`，产生新结构）。这正对应 roadmap 已记的"termination sampling vs.
builder/polarity analysis"那条切线。`_orient_slab` 是 (B) 的产物，但被 coherent.py 的
`_get_terminations` 复用，拆分时归属要单列（见 §4）。

### 1.3 `cli_build.py`（970 行）

| 行号区间 | 职责 | 关键符号 |
|---|---|---|
| 1–118 | docstring、导入、typer app、常量表 | `app`、`_DEFAULT_OUTDIR`、`_COHERENT_DEFAULTS`、`_RUN_CONFIG_KEYS`、`_CHARGE_POLICIES`、`_CHARGE_OPTIONS`、`_DEVICES`、`_CSV_FIELDS`、`_RELAX_FIELDS` |
| 120–199 | 结构/后端/构建器解析 + 配置构造 | `_load_structure`、`_resolve_builder`、`_resolve_backend`、`_validate_device`、`_make_config` |
| 202–390 | 命令行值解析/校验 | `_charge_policy`、`_finite_nonnegative`、`_finite_positive`、`_parse_backend_list`、`_optional_finite_positive`、`_registry`、`_merged_registry`、`_parse_oxidation`、`_check_charge_support`、`_load_config_overrides` |
| 393–463 | 无候选/预览报告 | `_report_no_candidates`、`_ood_columns` |
| 466–566 | 弛豫编排 | `_relax_candidates` |
| 569–656 | 输出写盘/终端打印 | `_write_outputs`、`_print_candidates` |
| 659–709 | 参数合并/回放 | `_resolve_run_params` |
| 712–970 | `build` 命令体 | `build`（`@app.command("build")`） |

**职责边界结论**：cli_build.py 是"命令定义 + 参数合并回放 + 弛豫编排 + 输出报告"四件
互不相干的事。它其实是三个文件里**风险最低、最该先拆的**：它的公共 API 面最小（只有
`app` 被 `cli.py` 引用），内部函数被测试以 `cli_build._xxx` 直接访问（见 §4 的 monkeypatch
面），拆成纯函数子模块几乎不动任何物理路径。

---

## 2. 目标结构

统一采用 **facade（门面）模式**：三个公开模块保留原路径、原导入面、原注册点，实现移入
`_` 前缀的私有子包；facade 只做 `from ... import ...` 的 re-export。理由：

- 公开导入 `from atominterface.builders.coherent import X`、`from atominterface.builders.charge import Y`、
  `from atominterface import cli_build` 是**库 API + 测试 + CLI 的三重契约**，facade 让拆分
  **零 API 变更**；
- monkeypatch 的 dotted-string 目标（§4）继续解析到 facade 上的名字；
- `builders/registry.py::autodiscover` 和 `builders/__init__.py` 硬编码
  `"atominterface.builders.coherent"` 这个模块路径，facade 不换名就不用动它。

### 2.1 `builders/` 目标树

```
src/atominterface/builders/
  coherent.py                       # facade：re-export + 底部 eager 注册（保留现状）
  _coherent/
    _constants.py                   # 容差、REGISTRY_MODES、AREA_POLICIES、PUBLISHED_MILLER 等
    _util.py                        # _to_ase、_reduced_formula、_pair_label（无内部依赖）
    cells.py                        # 面积账本 + 超胞/应变机械
    match.py                        # 晶格匹配 + 兜底
    gates.py                        # 几何闸门 B/C + GeometryGateError
    placement.py                    # 注册位点放置/测量（命名避开 builders/registry.py）
    config.py                       # CoherentInterfaceConfig
    assembly.py                     # 切板 + 组装 + reference_slabs
    builder.py                      # CoherentInterfaceBuilder + _to_candidate + 报告 helper
  charge.py                         # facade（保留现状导入面）
  _charge/
    _constants.py                   # ANALYSIS_CHARGE / CHARGE_KNOWN / CHARGE_SOURCE / SLAB_ID 等
    resolve.py                      # 电荷数据解析 + validate_charge_options
    analyze.py                      # analyze_charge_polarity + 几何/分组辅助
    policy.py                       # charge_rejection_reasons / tasker_classification 等解释 helper
    symmetric.py                    # SymmetricSlab / make_symmetric_slab / termination_cuts / mirror_slab
    orient.py                       # _orient_slab
```

### 2.2 `cli_build.py` 目标

```
src/atominterface/
  cli_build.py                      # facade + app + build 命令体（命令体留在 facade，保证 _xxx 仍是模块全局）
  _cli_build/
    params.py                       # 值解析 + 参数合并回放 + _make_config + _check_charge_support
    relax.py                        # _relax_candidates + _ood_columns
    report.py                       # _write_outputs + _print_candidates + _report_no_candidates
```

> `_load_structure` / `_resolve_builder` / `_resolve_backend` / `_validate_device` 放
> `params.py`，再 re-export 到 `cli_build.py` 模块全局——它们被测试 `monkeypatch.setattr(cli_build, ...)`
> 直接换掉，而 `build` 命令体按模块全局名调用，所以只要 re-export 到位，patch 语义不变（见 §4.3）。

---

## 3. 各子模块归属（符号 → 目标文件）

### 3.1 `_coherent/`

| 目标文件 | 迁入符号 |
|---|---|
| `_constants.py` | `GeometryGateError`（或入 `gates.py`）、`AREA_CONSERVATION_RTOL`、`INPLANE_PERIOD_RANGE`、`INPLANE_PERIOD_RTOL`、`ON_TOP_LATERAL_TOL`、`REGISTRY_PLACEMENT_TOL`、`_STACKING_STEP_TOL`、`PUBLISHED_MILLER`、`_STRAIN_TIE_DIGITS`、`_PLANE_Z_FLOOR`、`_OFF_ON_TOP_MIN_SHIFT`、`_JOIN_ROUND_OFF`、`REGISTRY_MODES`、`AREA_POLICIES` |
| `_util.py` | `_to_ase`、`_reduced_formula`、`_pair_label` |
| `cells.py` | `_lattice_area`、`_in_plane_vectors`、`_as_pymatgen_lattice`、`InterfaceAreas`、`interface_areas`、`_supercell_transform`、`_multiplicity`、`_format_matrix`、`_dense_transform`、`_scale_in_plane`、`_strain_film_onto` |
| `match.py` | `_MatchInfo`、`_match_lattices`、`_area_ratio_ok`、`_strain_of`、`_fallback_matches` |
| `gates.py` | `_shortest_inplane_period`、`_check_area_conservation`、`_check_inplane_period` |
| `placement.py` | `_describe_registry`、`_warn_unscanned_orientation`、`_RegistryPair`、`_closest_cross_pair`、`_warn_interface_proximity`、`_atomic_planes`、`_off_on_top_shift`、`_wrap_frac`、`_in_plane_frac`、`_join_lateral`、`_stacking_step`、`_stacking_period`、`_sites_of_step`、`_registry_target_frac`、`_registry_shift`、`_check_registry_placement`、`_registry_errors`、`_registry_as_json`、`_frac_or_none`、`_registry_lateral_ang`、`_place_join`、`_record_registry_properties`、`measure_registry` |
| `config.py` | `CoherentInterfaceConfig` |
| `assembly.py` | `_prepare_charge_structure`、`_get_terminations`、`_flatten_terminations`、`_make_slabs`、`_symmetric_terminations`、`_prepare_stack`、`_assemble_interface`、`_attach_charge_geometry`、`_select_termination`、`reference_slabs` |
| `builder.py` | `_warn_charge_diagnostics`、`_to_candidate`、`_ASSEMBLY_DETAIL_LIMIT/_WIDTH`、`_note_assembly_reason`、`CoherentInterfaceBuilder`、`@register_builder("coherent")` 装饰 |

### 3.2 `_charge/`

| 目标文件 | 迁入符号 |
|---|---|
| `_constants.py` | `ANALYSIS_CHARGE`、`CHARGE_KNOWN`、`CHARGE_SOURCE`、`SLAB_ID`、`_POLICIES`、`_SOURCES`、`_GROUPS`、`_MAX_THICKNESS_PASSES` |
| `resolve.py` | `_GeometryDefault`、`_DEFAULT_*`、`_real_number`、`_vector`、`_numeric_data`、`_validate_symbols`、`_mapping_data`、`_stored_data`、`resolve_charge_data`、`validate_charge_options` |
| `analyze.py` | `_structure_data`、`_resolve_structure_charges`、`_geometry_options`、`_vacuum_interval`、`_normal_geometry`、`_unwrap_periodic`、`_prepare_geometry`、`_group_masks`、`_finite_sum`、`_group_analysis`、`analyze_charge_polarity` |
| `policy.py` | `charge_rejection_reasons`、`is_potentially_polar`、`check_neutral_slabs`、`suggest_dipole_correction`、`tasker_classification` |
| `symmetric.py` | `SymmetricSlab`、`finite_slab_geometry`、`_select_sites`、`mirror_slab`、`_truncate_layers`、`_as_structure_copy`、`_normal_axis`、`_normal_extent`、`_mark_generated`、`_site_dependent_charge`、`termination_cuts`、`_oriented_repeat_height`、`_prefer_stoichiometric_termination`、`_symmetric_options`、`_select_termination_index`、`_symmetric_board_notes`、`make_symmetric_slab` |
| `orient.py` | `_ORIENT_TOL`、`_orient_slab` |

### 3.3 `_cli_build/`

| 目标文件 | 迁入符号 |
|---|---|
| `params.py` | `_load_structure`、`_resolve_builder`、`_resolve_backend`、`_validate_device`、`_make_config`、`_charge_policy`、`_finite_nonnegative`、`_finite_positive`、`_parse_backend_list`、`_optional_finite_positive`、`_registry`、`_merged_registry`、`_parse_oxidation`、`_check_charge_support`、`_load_config_overrides`、`_resolve_run_params` 及所有常量表 |
| `relax.py` | `_relax_candidates`、`_ood_columns` |
| `report.py` | `_report_no_candidates`、`_write_outputs`、`_print_candidates` |

---

## 4. 模块间接口与依赖分层

### 4.1 分层规则（无环）

唯一规则：**子模块只向下引用，facade 只向上 re-export，永不反向**。依赖分层（叶子 → 根）：

```
_coherent/_constants.py            （零内部依赖）
_coherent/_util.py                 （仅标准库/ase/pymatgen）
_coherent/cells.py                 → _constants、pymatgen
_coherent/match.py                 → cells、_constants、pymatgen
_coherent/gates.py                 → cells、_util、_constants、relax.proximity
_coherent/placement.py             → cells、_util、_constants、relax.proximity
_coherent/config.py                → charge facade、placement、_constants、base
_coherent/assembly.py              → config、match、cells、gates、placement、charge facade
_coherent/builder.py               → config、match、cells、assembly、placement、charge facade、candidates
builders/coherent.py (facade)      → 上述全部 + charge facade + registry
```

charge 侧同理：

```
_charge/_constants.py              （零依赖）
_charge/resolve.py                 → _constants、ase
_charge/analyze.py                 → resolve、_constants、ase
_charge/policy.py                  → analyze、resolve、_constants
_charge/orient.py                  → _constants、pymatgen
_charge/symmetric.py               → analyze、orient、policy、resolve、_constants、thermo.surface、pymatgen
charge.py (facade)                 → 上述全部
```

关键事实：**coherent → charge 是单向边**（coherent 大量 import charge，charge 从不 import
coherent），所以 charge 必须先拆、coherent 后拆，顺序不能反。

### 4.2 跨模块的稳定接口（拆分后对外不变、对内是 seam）

这些签名今天已存在，拆分只是把它们从"同一文件内部调用"变成"跨文件调用"，语义与参数不变：

| seam | 签名 | 所属目标 |
|---|---|---|
| 晶格匹配 | `_match_lattices(sub_lattice, film_lattice, config, *, strain_limit=None, label="") -> list[_MatchInfo]` | `match.py` |
| 兜底匹配 | `_fallback_matches(sub_lattice, film_lattice, config, *, strain_limit=None)` | `match.py` |
| 面积账本 | `interface_areas(substrate_lattice, film_lattice, *, tolerance=1e-6, policy="strict") -> InterfaceAreas` | `cells.py` |
| 膜应变 | `_strain_film_onto(film_slab, substrate_slab_2d) -> Structure` | `cells.py` |
| 面内超胞 | `_scale_in_plane(structure, transform)` | `cells.py` |
| 闸门 B | `_check_area_conservation(side, before, after, transform)` | `gates.py` |
| 闸门 C | `_check_inplane_period(side, slab, max_strain) -> d_ref` | `gates.py` |
| 位点放置 | `_place_join(sub_coords, film_coords, sub_2d, mode) -> (target_frac, as_built)` | `placement.py` |
| 位点测量 | `measure_registry(atoms) -> dict | None` | `placement.py` |
| 组装 | `_assemble_interface(substrate_slab, film_slab, match=None, config=None)` | `assembly.py` |
| 切板 | `_make_slabs(substrate, film, config) -> (sub_terms, film_terms)` | `assembly.py` |
| 参考板 | `reference_slabs(substrate, film, config, match=None, *, substrate_termination=0, film_termination=0)` | `assembly.py` |
| candidate | `_to_candidate(interface, substrate_slab, film_slab, match, sub_term_idx, film_term_idx, config) -> Candidate` | `builder.py` |
| 电荷解析 | `resolve_charge_data(symbols, *, charges, oxidation_states, stored, initial_charges, oxi_states)` | `_charge/resolve.py` |
| 极性分析 | `analyze_charge_polarity(atoms, slab_tags=None, *, ...) -> dict` | `_charge/analyze.py` |
| 对称板 | `make_symmetric_slab(structure, miller, *, termination, min_thickness, vacuum, max_layers) -> SymmetricSlab` | `_charge/symmetric.py` |
| 定向 | `_orient_slab(slab) -> Structure` | `_charge/orient.py` |

### 4.3 monkeypatch 面（拆分最易踩的坑，逐条列）

测试用 dotted-string `patch` / `monkeypatch.setattr` 钉死了 6 处，拆分必须让它们**照旧解析**：

| patch 目标 | 测试位置 | 现状是否真的生效 | 拆分处理 |
|---|---|---|---|
| `atominterface.builders.coherent._PYMTG_COHERENT_BUILDER` | `test_coherent_builder.py:111` | 该绑定在 coherent.py 里**只定义不使用**（grep 仅 62–67 行），是死绑定；`_assemble_interface` 实际走 `from pymatgen.core import Structure`。patch 它不改变任何执行路径 | facade 继续 re-export `_PYMTG_COHERENT_BUILDER`/`_PYMTG_STRUCTURE` 即可，属零风险兼容 |
| `atominterface.builders.coherent.ZSLGenerator`（`create=True`） | `test_coherent_builder.py:456` | **不生效**：`_match_lattices` 在函数内 `from pymatgen.analysis.interfaces.zsl import ZSLGenerator`，patch 的是 `coherent.ZSLGenerator` 属性，本地 import 不读它。测试恰好两路径都产出 match 所以不报错 | 属**潜在测试 bug**，不在本次修复范围；落地时在 facade 加 `ZSLGenerator` re-export 或顺带修该测试目标为 `monkeypatch.setattr(ZSLGenerator, "__call__", ...)`（后者才是真正打断 ZSL 的写法，见 `test_coherent_builder.py:1130` 的正确示范） |
| `atominterface.builders.coherent._orient_slab` | `test_slab_geometry.py:798` | **生效**：`_orient_slab` 定义在 charge.py，被 coherent.py 顶部 import 成模块全局；`_get_terminations` 按 coherent 模块全局名调用，patch `coherent._orient_slab` 就换掉了它 | **必须保留"经 facade 全局解析"的调用方式**。若 `_get_terminations` 迁到 `assembly.py` 后改成 `from atominterface.builders.charge import _orient_slab`，patch 就落空。二选一：① `assembly.py` 通过 `import atominterface.builders.coherent as _coh` 后调 `_coh._orient_slab`（保 patch 点，但引入 coherent→assembly→coherent 回环，需用运行时 import 或改 facade 结构）；② **推荐**：patch 目标改成 `atominterface.builders.charge._orient_slab`（1 行测试改动，语义等价，不引入环） |
| `cli_build._resolve_builder` | `test_cli.py:126`、`test_run_config.py:70`、`test_registry_field.py:738` | **生效**：`build` 命令按 cli_build 模块全局名调用 | facade re-export 到位即可（`build` 命令体留在 `cli_build.py`，`_resolve_builder` 从 `params.py` re-export 成模块全局） |
| `cli_build._load_structure` | 同上三处 | 生效 | 同上 |
| `pymatgen.core.surface.SlabGenerator` | `test_coherent_builder.py:790` | 生效（patch pymatgen 本身，与 coherent 无关） | 拆分无关，照旧 |

**一句话规则**：凡被 patch 的名字，拆分后必须仍"从被 patch 的模块全局名解析得到"。导入 re-export
解决"能不能 import 到"，调用点解析解决"换掉后生不生效"——后者才是 `_orient_slab` 这类
"定义在别处、被这里调用"的名字会踩的坑。

---

## 5. 迁移顺序（每步全绿、单独提交）

依赖顺序决定顺序：charge 是 coherent 的下游，先拆；cli_build 是纯前端，随时可拆。每步
验收底线沿用 PRD §7：全量 pytest ≥ 现有数量全绿 + `ruff check` + `mypy src` + CLI smoke。

| 步 | 内容 | 提交类型 | 验证重点 |
|---|---|---|---|
| 0 | `git branch` 确认在独立分支（非 `feat/v0.5.1-increment`），新建 `refactor/split-...` | — | — |
| 1 | 抽 `_charge/_constants.py`（常量 + `SLAB_ID` 等），`charge.py` 改 re-export | `refactor(charge): extract charge constants module` | `test_charge.py` 全绿，`SLAB_ID` 等名字 import 不变 |
| 2 | 拆 `_charge/resolve.py` + `analyze.py` + `policy.py` | `refactor(charge): split charge resolution, analysis, policy` | `test_charge.py` 全绿（这是电荷模块最大测试面，823 行） |
| 3 | 拆 `_charge/orient.py` + `symmetric.py` | `refactor(charge): split slab orientation and symmetric-slab construction` | `test_slab_geometry.py`、`test_reference_slabs.py` 全绿；**同步决定 `_orient_slab` 的 patch 目标（§4.3）** |
| 4 | 抽 `_coherent/_constants.py` + `_util.py` + `cells.py` | `refactor(coherent): extract constants, util, cell helpers` | `test_interface_area.py`（455 行，只 import `AREA_POLICIES`/`InterfaceAreas`/`interface_areas`）全绿 |
| 5 | 拆 `_coherent/match.py` | `refactor(coherent): extract lattice matching and fallback` | `test_coherent_builder.py` 匹配用例 + `test_strain.py`；确认 `ZSLGenerator` patch 语义（§4.3） |
| 6 | 拆 `_coherent/gates.py` + `placement.py` | `refactor(coherent): extract geometry gates and registry placement` | `test_slab_geometry.py`（`_check_area_conservation` 等）、`test_registry_field.py`、`test_registry_stability.py` 全绿 |
| 7 | 拆 `_coherent/config.py` + `assembly.py` + `builder.py`，`coherent.py` 收尾成 facade + eager 注册 | `refactor(coherent): split config, assembly, builder` | `test_coherent_builder.py` 全绿（1167 行，最大面）+ `builders/__init__.py` 的 `autodiscover` 仍注册到 coherent |
| 8 | 拆 `_cli_build/` 三个子模块，`cli_build.py` 留命令体 + re-export | `refactor(cli): split build command into params, relax, report` | `test_cli.py`、`test_cli_energy.py`、`test_candidate_reports.py`、`test_run_config.py` 全绿；CLI smoke `atominterface build examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp --outdir /tmp/x` 仍产出候选 |

**关键顺序约束**：步 4–7 必须在步 1–3 之后（coherent 依赖 charge facade）；步 8 可提前，
与步 1–7 无先后依赖，但建议放最后——它依赖 `measure_registry`（在 `_coherent/placement.py`）
保持从 `coherent` 可 import。

---

## 6. 测试影响面

**目标：facade 阶段做到"零测试文件改动"**（除 §4.3 的 `_orient_slab` patch 目标这一个可选
1 行改动外）。逐测试文件的受影响面：

| 测试文件 | 依赖模块 | 拆分后状态 |
|---|---|---|
| `test_charge.py`（823） | `builders.charge` | 步 1–3 后仍从 facade import，零改动 |
| `test_slab_geometry.py`（840） | `builders.charge` + `builders.coherent`（`_check_area_conservation`、`_check_inplane_period`、`_shortest_inplane_period`、`_make_slabs`、`_get_terminations`、`_flatten_terminations` 等）+ `patch("...coherent._orient_slab")` | facade re-export 覆盖 import；唯一要动的是 `_orient_slab` patch 目标（§4.3） |
| `test_coherent_builder.py`（1167） | `builders.coherent`（11 个下划线名 + 2 个 dotted patch） | facade re-export 覆盖 import；`_PYMTG_COHERENT_BUILDER`/`ZSLGenerator` 两个 patch 见 §4.3 |
| `test_registry_stability.py`（265） | `builders.coherent` | facade，零改动 |
| `test_registry_field.py`（850） | `builders.coherent` + `cli_build`（`_relax_candidates` 直调 + `monkeypatch` `_resolve_builder`/`_load_structure`） | facade，零改动 |
| `test_interface_area.py`（455） | `builders.coherent`（面积账本） | facade，零改动 |
| `test_reference_slabs.py`（246） | `builders.charge.SLAB_ID` + `builders.coherent` | facade，零改动 |
| `test_candidate_reports.py`（248） | `cli_build._print_candidates`/`_write_outputs` | `cli_build` re-export，零改动 |
| `test_cli.py`（674）、`test_run_config.py`（295）、`test_cli_energy.py`（856） | `atominterface.cli_build` 模块 + monkeypatch | 步 8 后 `cli_build` 模块全局仍在，零改动 |
| `test_validation_cu_ni.py`（474） | `atominterface.builders` 包（`CoherentInterfaceBuilder`/`CoherentInterfaceConfig`） | 包 `__init__` 不变，零改动 |

**等价性纪律**：拆分是纯搬移（不改变任何数值路径），验收用"同进程 A/B + `array_equal`"证明
搬移前后候选逐位一致（`CLAUDE.md` 第 6 条），结论带"仅同环境成立"限定；**禁止用 extxyz
文件哈希当回归闸门**（第 10 条，弛豫产物本身跨运行不可逐位复现）。

---

## 7. 风险与注意事项

1. **回环是头号风险**。`CoherentInterfaceConfig` 同时被 `assembly`/`builder`/facade/测试引用，
   `_MatchInfo` 被 `match`/`assembly`/`builder`/测试引用（细胞的账本函数不引用它，已逐行核对）
   ——两者必须落在叶子层（`config.py`、`match.py`），不能让 `builder` 反向 import `coherent`
   facade 拿 config。注意 `match.py` 单向依赖 `cells.py`，所以 `_MatchInfo` 只能放 `match.py`
   或更靠下的叶子，放进 `cells.py` 会立即成环。
   落地时用 `python -c "import atominterface.builders.coherent"` 冷启动冒烟即可发现环。
2. **`_orient_slab` 的调用点解析**（§4.3）是唯一会"看似绿实则 patch 失效"的点，拆分评审必须
   单独盯。
3. **eager 注册必须留在 `coherent.py` facade 底部**（现 2746–2753 行），`builders/registry.py:75`
   硬编码模块名 `"atominterface.builders.coherent"`，`autodiscover` 会 `importlib.reload` 它——
   facade 若把注册逻辑搬进子模块，reload 语义会变。
4. **`_to_ase`/`_reduced_formula`/`_pair_label` 是共享工具**，单独放 `_util.py`，避免三个子模块
   各自复制一份（复制会制造"同一逻辑两处"的漂移，正是本项目铁律要防的）。
5. 换行纪律：`.md`/`.py` 保持 LF（`Path.write_text()` 在 Windows 会把 LF 翻成 CRLF，
   所以只能用 `open(..., newline="")`）；归档 CSV 保 CRLF，不碰。
6. 打包不构成风险，但结论要写对：`src/atominterface.egg-info/` 是本地构建产物，`.gitignore:23`
   已忽略、未跟踪，不参与版本库；`pyproject.toml` 用
   `[tool.setuptools.packages.find] include = ["atominterface*"]` 自动发现子包，所以新私有子包
   **只需带 `__init__.py`** 就会被打包。落地第 0 步用 `pip install -e .` 后
   `python -c "import atominterface._coherent, atominterface._charge, atominterface._cli_build"`
   各验一次即可。

---

## 8. 待拍板的问题（留给下周期立项）

1. **拆的粒度**：§2 是"功能域全拆"（`_coherent/` 9 个文件）。更保守的替代是按 roadmap 已记的
   粗切线只切 3 块（匹配族 / 兜底 / 构建器类 + misc），`coherent.py` 保留 ~700 行 facade+类。
   推荐先做 charge + cli_build 全拆（风险低、收益立现），coherent 走保守 3 块 + 单独切出
   `placement.py`（为 R11 `registry="scan"` 铺路）。
2. **`_orient_slab` patch 目标**：改测试目标为 `charge._orient_slab`（推荐），还是在 facade 造
   `_orient_slab` 的运行时间接层。
3. **`ZSLGenerator`（`create=True`）无效 patch**（§4.3）：建议顺带修测试为
   `monkeypatch.setattr(ZSLGenerator, "__call__", _boom)`（`test_coherent_builder.py:1130` 已是这么写的）。
   **不要选「只 re-export `ZSLGenerator` 做兼容」**——那是假修复：`_match_lattices` 在函数体内
   `from pymatgen...zsl import ZSLGenerator` 本地导入，永远读 pymatgen 的类，facade 上 re-export
   只会让 patch 不再报错、行为依旧不受影响，把「删了会红」变成「留着也没用」的静默无效。
   `__call__` 那条路确实咬得住：`coherent.py:878` 是 `generator(film_vecs, sub_vecs)` 实例调用。
4. 私有子包是否加 `__init__.py` 的 `__all__`（建议：`_coherent/__init__.py` 留空，只作命名空间，
   避免再引入一层 import 顺序敏感）。

---

## 9. 与本周期其余项的关系

- 本文只**记录边界，不落地**（PRD R8 明确"完整拆分不在本周期"；R10 是 P2 延后项）。
- 与 `roadmap.md` "Monolith split candidates" 一节互补：roadmap 给了粗切线，本文给行号级
  边界、模块间接口、迁移顺序和 monkeypatch 面。落地时以本文为准，roadmap 那节回写为
  "已立项/已拆分"状态。
- 初稿要求落地前确认的两项**已于 2026-10-31 完成**，不再是前置条件：MACE-MP-0 medium 权重
  SHA-256 锚点已填入 `mace_.py`（`01bfe221…5845`，四路互证），版本号已 bump 至 0.5.2 并发布
  （v0.5.1 → PR #37、v0.5.2 → PR #38）。拆分应从 `main`（`80eac59` 起）开分支，不存在交叠提交。
