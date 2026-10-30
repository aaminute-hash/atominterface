# 复现指南：Cu/Ni(111) 与 Cu/Ni(110) 界面形成能验证

本目录归档的是 Cu/Ni 界面在两个**非 (001)** 取向上的验证数据。原始产物在
`F:/tmp/d1_111_fixed/`，本目录的三个 CSV 由脚本从那里的机器可读文件直接导出
（见 §4），**未做任何筛选、剔除或后处理**。

实验分五批，每批都是**单变量**：

| 批次 | 唯一变量 | 产物 |
|---|---|---|
| 厚度扫描 | `--min-thickness` ∈ {10.8, 14.4, 18.0, 21.6} | `thickscan_fixed/t*/`（(111)）、`orient_110/t*/`（(110)） |
| 注册扫描 | 界面注册（4 个位点） | `registry_scan_111/`、`registry_scan_110/` |
| 入口无关性 | 初始 gap（2.0 → 2.4 Å） | `gap_probe/` |
| 初态注册映射 | `--min-thickness`（dummy 后端，无能量） | `asbuilt_map/asbuilt_111.json` |
| fmax 对照 | `--fmax`（0.002 → 0.01） | `fmax_scan/` |

---

## 0. 前置条件

- Python 3.13.14；torch 2.14.1+cpu、mace-torch 0.3.17、ase 3.29.0、
  numpy 2.5.4、pymatgen 2026.10.24。
- MACE-MP-0 medium 权重（44.42 MB）。首次运行自动下载；也可预先放置：
  `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model`
  权重 SHA-256：`01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845`。
- atominterface **0.3.0** 源码（须包含 `c440b81`
  `fix(examples): strain the coherent bulk reference on the (hkl) plane`，
  否则 (111)/(110) 的 γ_Ni 会随层数单调漂移、不收敛）。
- **`PYTHONPATH` 必须显式指定**：venv 里的 editable 安装指向旧位置，
  不指定会静默加载错版本。
  `PYTHONPATH=<repo>/src`
- 本目录所有数据在 `F:/tmp/wt-111b`（分支 `fix/non-001-slab-geometry`，
  HEAD `c440b81`）上测得；入口无关性那批在钉死的 `F:/tmp/wt-111c`
  （detached HEAD @ `c440b81`）上测得。

---

## 1. 完整命令

### 1.1 厚度扫描（两个取向，同一协议，只改 miller）

```bash
cd F:/tmp/wt-111b && PYTHONPATH=F:/tmp/wt-111b/src F:/tmp/venv-ledger/Scripts/python.exe -u \
  examples/03_interface_energy.py --backend mace --device cpu \
  --sub-miller 1 1 1 --film-miller 1 1 1 \
  --min-thickness 18.0 --vacuum 15.0 --max-candidates 1 \
  --fmax 0.002 --max-steps 400 \
  --outdir F:/tmp/d1_111_fixed/thickscan_fixed/t18_0
```

- **(110)**：把 `--sub-miller 1 1 1 --film-miller 1 1 1` 改为 `1 1 0` / `1 1 0`，
  `--outdir` 改为 `F:/tmp/d1_111_fixed/orient_110/t18_0`。
- 另外三档：`--min-thickness 10.8 / 14.4 / 21.6`，`--outdir` 末段相应改为
  `t10_8 / t14_4 / t21_6`。

### 1.2 注册扫描（决定性实验）

脚本复用 example 03 的 `_parse_args`，所以 CLI 与上面同形；只是输出目录写死在
脚本里（`OUT` 常量），CSV 文件名由 `--min-thickness` 决定。

```bash
# (111)：4 个注册各独立松弛，参考板只松弛一次共享
cd F:/tmp/wt-111b && PYTHONPATH=F:/tmp/wt-111b/src F:/tmp/venv-ledger/Scripts/python.exe -u \
  F:/tmp/d1_111_fixed/registry_scan_111.py --backend mace --device cpu \
  --sub-miller 1 1 1 --film-miller 1 1 1 \
  --min-thickness 18.0 --vacuum 15.0 --max-candidates 1 \
  --fmax 0.002 --max-steps 900

# (110)：同一脚本形态，miller 改 1 1 0；脚本为 registry_scan_110.py，
#        位点表在文件头（on-top / short-bridge / ridge / trough）
cd F:/tmp/wt-111b && PYTHONPATH=F:/tmp/wt-111b/src F:/tmp/venv-ledger/Scripts/python.exe -u \
  F:/tmp/d1_111_fixed/registry_scan_110.py --backend mace --device cpu \
  --sub-miller 1 1 0 --film-miller 1 1 0 \
  --min-thickness 14.4 --vacuum 15.0 --max-candidates 1 \
  --fmax 0.002 --max-steps 900
```

- **`--max-steps 900` 不是浪费**：注册扫描必须给足步数，否则"卡在对称点"会被
  误报成"收敛"。
- **脚本里写死了两个绝对路径**，换机器/换 worktree 时必须先改：
  `EX03 = Path("F:/tmp/wt-111b/examples/03_interface_energy.py")`（入口脚本）与
  `OUT = Path("F:/tmp/d1_111_fixed/registry_scan_xxx")`（输出目录）。
  除这两处外脚本不改任何源码，只 import 并复用 example 03 的函数。
- **无预松弛**：这是 (110) 的教训——预松弛会掩盖注册锁定。
- 位点偏移**不靠推算**：(111) 的三个空位由初态板自身读出，(110) 的四个位点写在
  脚本文件头；每个偏移施加后都**实测回来并打印**，标签是被验证的，不是假设的。
- 断言：若出现低于 0.081 J/m²（(111)）或低于 0.088（(110)）的注册，**停下报告**，
  说明还有未扫到的更低位点。

### 1.3 入口无关性（gap 2.0 → 2.4）

```bash
# gap = 2.0 与 2.4 各跑一遍；只覆盖模块级 GAP 常量，不改源码
cd F:/tmp/wt-111c && PYTHONPATH=F:/tmp/wt-111c/src F:/tmp/venv-ledger/Scripts/python.exe -u \
  F:/tmp/d1_111_fixed/gap_probe.py --orient 111 --registry fcc-hollow \
  --gap 2.4 --min-thickness 18.0
```

`--fmax` 默认 0.002、`--max-steps` 默认 900，无需显式给。三个构型各两档，共六次：

| `--orient` | `--registry` | `--min-thickness` |
|---|---|---|
| 110 | trough | 14.4 |
| 111 | fcc-hollow | 18.0 |
| 111 | on-top | 18.0 |

### 1.4 builder 初态注册映射（不耗能，几秒）

```bash
cd F:/tmp/wt-111b && PYTHONPATH=F:/tmp/wt-111b/src F:/tmp/venv-ledger/Scripts/python.exe -u \
  F:/tmp/d1_111_fixed/asbuilt_map.py
```

无参数；内部用 dummy 后端、直接读 `_build_candidates`，不涉及能量、不松弛。
输出 `asbuilt_map/asbuilt_111.json`（四档厚度的初态注册与堆垛序列）。

### 1.5 fmax 对照

```bash
# (111) @18.0 Å：只把 --fmax 从 0.002 改成 0.01，其余与 §1.1 相同
... examples/03_interface_energy.py --backend mace --device cpu \
  --sub-miller 1 1 1 --film-miller 1 1 1 \
  --min-thickness 18.0 --vacuum 15.0 --max-candidates 1 \
  --fmax 0.01 --max-steps 400 \
  --outdir F:/tmp/d1_111_fixed/fmax_scan/111_18_0_fmax0.01

# (001) @14.4 Å：published 协议（10 候选、max-steps 200）
... examples/03_interface_energy.py --backend mace --device cpu \
  --min-thickness 14.4 --vacuum 15.0 --max-candidates 10 \
  --fmax 0.01 --max-steps 200 \
  --outdir F:/tmp/d1_111_fixed/fmax_scan/001_fmax0.01
```

(001) 的 fmax 0.002 侧为 `orient_001/t001/`（同协议，只把 `--fmax` 改回 0.002）。
**(111) 的 14.4 Å 档不做 fmax 对照**——它被 on-top 锁死，fmax 效应会被 11 倍的
注册误差淹没。

---

## 2. 预期产出与耗时

- **耗时量级**：单候选 (111) 约 **58 秒**（含模型加载，15 原子）。注册扫描一档
  = **4 次独立松弛** + 一次共享参考板，量级是单候选的数倍；厚度扫描四档同理。
  (110) 超胞更大（24–36 原子），单点更贵。
- 产物结构（每批次一个目录）：

```
candidates.csv        每候选一行（本批次 --max-candidates 1 → 1 行）
metadata.yaml         脚本产出的 provenance（cli_args / run_id / 权重 hash / 环境）
*.extxyz              弛豫后的界面与两侧参考板结构（rank 前缀）
registry_scan_*/*.csv 注册扫描：4 注册 × 1 厚度 → 4 行
gap_probe/*.json      入口无关性：构型 × gap → 每个一份 JSON
```

---

## 3. 怎么验证自己复现对了

按优先级：

1. **几何先对，再谈能量。** 1×1 面积必须逐位等于解析值：
   (111) **5.658707 Å²**、**(110) 9.240631 Å²**、(001) 6.5341 Å²。
   面积不对意味着几何或取向错了，后面所有数字都无意义。
2. **状态列全绿**：`valid=True, status=ok, converged=True,
   area_consistent=True, energies_finite=True`。
   ⚠ 全绿**不等于**注册对——见 §5 第 1 条。
3. **表面能收敛**（容差按写入精度）：
   - γ_Ni(111) = 1.8258 → 1.8237 → 1.8237 → 1.8237（总漂移 0.0021）
   - γ_Cu(111) = 1.2994 / 1.2994 / 1.2994 / 1.2993
   - γ_Ni(110) = 2.1209 → 2.1191 → 2.1192 → 2.1192（0.0018）
   - γ_Cu(110) = 1.5684 四档逐位相同
   若 γ_Ni 随层数**线性单调下降**而不平台化，说明共格参考的缺陷还在
   （即源码不含 `c440b81`）。
4. **界面能（注册扫描定稿值）**：
   - (111) fcc 空位 **0.0812** @18.0 与 @21.6；hcp 空位 **0.0810 / 0.0811**
   - (110) 槽位 **0.0884 / 0.0881 / 0.0880 / 0.0881**（四档厚度）
   - 若只做单次运行而不扫注册，会得到 0.898（(111)@14.4）或 0.5953（(110)
     薄板）——**那是驻点假象，不是复现失败**，见 §5。
5. **入口无关性**：三构型的 Δγ_int ≤ 1×10⁻⁴、Δdz ≤ 5×10⁻⁴ Å、
   `site_shift = 0.0000`。
6. **文件级校验**：见 `metadata.yaml` 的 `artifacts` 段（三个 CSV 的 SHA-256）。

---

## 4. 本目录 CSV 是怎么从产物导出来的

三个 CSV 由一段一次性导出脚本生成（脚本不随本目录归档，逻辑如下，可照做）：

**`registry_scan.csv`（24 行）**
- 源：`registry_scan_111/registry_scan_111_t{18_0,21_6}.csv`（8 行）+
  `registry_scan_110/registry_scan_110_t{10_8,14_4,18_0,21_6}.csv`（16 行）。
- **取两者列集合的并集**，新增 `orientation` 列（`(111)` / `(110)`）；
  某取向源文件没有的列留空，**不填充、不推算**。
- 逐行照抄，不改任何数值。

**`registry_vs_thickness.csv`（8 行）**
- (111) 四行：`asbuilt_map/asbuilt_111.json`（层数 / 原子数 / 初态偏移 /
  初态注册）+ `thickscan_fixed/t*/candidates.csv`（γ_surf → J/m²、γ_int）。
- (110) 四行：`orient_110/t*/candidates.csv`（层数 / 原子数 / γ）+
  `gap_probe/110_trough_gap2.0_t14_4.json`（`as_built_offset` 与
  `as_built_lateral_A` = 脊位 1.8075 Å，四档相同）。
- `ratio_pct` = 100 × γ_int / (γ_Cu + γ_Ni)，**由本文件内 4 位小数的 γ 值重算**
  （源 candidates.csv 无该列），末位可能与报告表格相差 1。

**`gap_probe.csv`（6 行）**
- 源：`gap_probe/*.json` 六个文件。
- 字段直取，**只有两处改名**（因为源命名会让人混用口径）：
  `dz_post_A` → **`layer_dz_post_A`**（层平均 z 差）、
  `CuNi_pair_dz_A` → **`pair_dz_post_A`**（最近对 z 差）。

γ 的单位换算统一用 `1 eV/Å² = 16.02176634 J/m²`（与产生脚本同一个常数）。

---

## 5. 已知噪声与口径（引用前必读）

1. **全绿 ≠ 注册对。** 驻点假象的四项检查全绿（收敛、fmax 达标、`valid=True`、
   OOD 无告警），γ 是个像样的正数，`NeighborWatcher` 也抓不到（最近接触 2.00 /
   2.27 Å 远高于阈值）。**唯一可靠判据是横向偏移与堆垛序列**，必须显式扫注册。
2. **加大步数不是解药。** (110) 三档薄板给到 900 步仍逐位停在脊位；
   (111) 14.4 Å 900 步与 400 步逐位相同。**扫注册才是解药。**
3. **dz 两个口径不得混用**：`layer_dz` 与 `pair_dz` 在 (110) 槽位是 1.2306 与
   2.4460 Å。源文件里 `registry_scan_110.csv` 的 `dz_post_A` 是 pair_dz，
   而 `gap_probe.json` 的 `dz_post_A` 是 layer_dz——同名不同义。
4. **`registry_unchanged` 列有假阴性**：脚本比较分数坐标时未做整格回绕，
   (111)@21.6 的 fcc/hcp 两行 post = pre ± 1.0，被误判为 False。
   判据请用 `lateral_pre_A == lateral_post_A` 与 Δsite。
5. **末位噪声**：跨进程重跑能量末位漂移约 1×10⁻⁶ eV/Å²（torch CPU 线程归约序）。
   不要对 extxyz 字节或能量末位做相等断言。
6. **无 DFT 交叉验证**：(111)/(110) 只有 MACE-MP-0 一个数，**不得与文献值并列
   成对照**，只能标"待独立复核"。
7. **(001) 不受影响**：`c440b81` 对 (001) 是恒等变换，复跑的共格块体能量 4 列
   与界面能逐位相同，γ_int 0.0567 逐位复现。
8. 本目录数值在 **float32** 下测得。float64 对照只在 (001) 上做过
   （换精度只动 γ_int 0.0001 J/m²，精度不是瓶颈），(111)/(110) 未做。
