# 复现指南：多候选 Cu(001)|Ni(001) 界面形成能基准

本目录（`docs/validation/cu_ni_001_multi_candidate/`）是两次完整多候选运行的
**原始产物**：`examples/03_interface_energy.py`，atominterface `0.2.1`
（commit `92dc284`），MACE-MP-0 medium，float32，CPU，10 个候选
（`--max-candidates 10`），两次运行**唯一变量是 `--fmax`**：

- 根目录（`candidates.csv` 等）：**fmax=0.01 批次**（run_id `e377717853`）。
  其 γ_int 已知系统性偏高，见 README 顶部偏差提示。
- `fmax_0002_converged/`：**fmax=0.002 批次**（run_id `8a64f9a363`），
  收敛后基准，γ_int 中位数 0.0567 J/m²、极差 0.0002。

数据未经任何筛选或后处理。

## 0. 前置条件

- Python 3.13.14；torch 2.14.1+cpu、mace-torch 0.3.17、ase 3.29.0、
  numpy 2.5.4、pymatgen 2026.10.24（完整锁定见 `environment.lock.yml`）。
- MACE-MP-0 medium 权重（44.42 MB）。首次运行 mace-torch 会自动下载；
  也可预先放置：
  `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model`
  权重 SHA-256：`01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845`。
- atominterface 0.2.1 源码（commit `92dc284` 或包含同等能量账本与
  多候选主循环的版本）。本仓库内 `pip install -e .` 后可直接运行；
  未安装时用 `PYTHONPATH=<repo>/src`。

## 1. 完整命令行（原始运行，逐字）

```bash
cd F:/tmp/wt-d1 && PYTHONPATH=F:/tmp/wt-d1/src F:/tmp/venv-ledger/Scripts/python.exe \
  examples/03_interface_energy.py --backend mace --device cpu \
  --max-candidates 10 --min-thickness 14.4 --vacuum 15.0 \
  --fmax 0.01 --max-steps 200 --outdir F:/tmp/d1_out
```

在任意等效环境（本仓库 worktree + 锁定环境）中的通用形式：

```bash
python examples/03_interface_energy.py --backend mace --device cpu \
    --max-candidates 10 --min-thickness 14.4 --vacuum 15.0 \
    --fmax 0.01 --max-steps 200 --outdir <你的输出目录>
```

各参数含义（全部保持脚本默认口径，未做任何放宽/收紧）：

| 参数 | 值 | 作用 |
|---|---|---|
| `--backend mace` | MACE-MP-0 medium | 真实 uMLIP 驱动（`dummy` 仅链路自检，非物理） |
| `--device cpu` | CPU | 本次为 CPU 运行 |
| `--max-candidates 10` | 10 | ZSL 候选池（`max_area=100 Å²` 内）保留面积最小的 10 个 |
| `--min-thickness 14.4` | 14.4 Å | 两侧共用厚度 → 8 层 Cu / 10 层 Ni（厚度扫描平台点） |
| `--vacuum 15.0` | 15 Å | 真空层（真空扫描已证 10 Å 即足够，15 Å 带余量） |
| `--fmax 0.01` | 0.01 eV/Å | FIRE 力收敛判据（0.05 太松，见验证报告 §5） |
| `--max-steps 200` | 200 | 每个结构弛豫步数上限（本批无一撞上限） |

脚本内固定常量（非 CLI 可调）：`SEED=0`、`OOD_THRESHOLD=0.8`、
`MAX_AREA=100.0 Å²`、`GAP=2.0 Å`、`LAYER_GAP_A=0.75 Å`、
substrate/film 均为 (0,0,1)，体相输入 `examples/data/Cu_fcc.vasp`、
`Ni_fcc.vasp`。

每一步在做什么：枚举 ZSL 界面候选 → 构建共格界面与两侧参考 slab
（参考态 `strained-coherent`，slab 面内应变计入 coherent 体相，
out-of-plane 抛物线松弛）→ 对界面、两个参考 slab 各做一次 FIRE 弛豫 →
按能量账本算 γ_surf_Cu / γ_surf_Ni / γ_int → 写 `candidates.csv`、
`metadata.yaml`、每候选 3 个 extxyz 与完整日志。

### 1b. 收敛批次（fmax = 0.002，基准值）

与 §1 **唯一变量是 `--fmax`**（其余参数、候选、权重、seed 全部相同）：

```bash
python examples/03_interface_energy.py --backend mace --device cpu \
    --max-candidates 10 --min-thickness 14.4 --vacuum 15.0 \
    --fmax 0.002 --max-steps 200 --outdir <你的输出目录>/fmax_0002_converged
```

实测耗时约 21 分钟（CPU；界面弛豫步数 49–53 → 71–75，无一撞
`max_steps=200`）。§1 的 fmax=0.01 批次 γ_int 带有已知系统性偏高
（小超胞约 +0.005 J/m²），**引用基准值以本批次为准**。

## 2. 预期产出与耗时

- **耗时量级**：fmax=0.01 约 25 分钟（原始运行 24m50s），fmax=0.002 约 21
  分钟（原始运行 20m56s，另有一次 5m25s 的四候选计时探针，未入库），
  均为 CPU、10 候选。其中一部分是每个单点重建 `MACECalculator` 重复
  `torch.jit.load` 44 MB 权重的开销，属已知性能问题（README §9.4）。
- 产物清单（与本目录一致）：

```
candidates.csv        10 行 × 44 列
metadata.yaml         脚本产出的 provenance（原样，未后处理）
provenance_extra.yaml 运行后补充的环境/权重/耗时（与脚本产出分离）
run.log               完整运行日志（含 FIRE 轨迹）
README.md             本批数据的索引与逐候选分析（顶部有偏差提示）
structures/           每候选 3 个弛豫后 extxyz（rank 前缀）+ 2 个共享体相
fmax_0002_converged/  fmax=0.002 收敛批次，结构与上相同，另含
                      comparison_fmax.csv/.md（两次运行的逐候选机器可读对照）
```

## 3. 怎么验证自己复现对了

按优先级：

1. **状态列全绿**：`candidates.csv` 中 10 行全部
   `valid=True, status=ok, converged=True, area_consistent=True,
   energies_finite=True`，无失败行。
2. **表面能跨候选一致**（超胞尺寸一致性检验，容差 0 即可按写入精度比对）：
   - fmax=0.01：γ_Cu = **1.4977 J/m²**（0.093476 eV/Å²，10 个候选完全相同）、
     γ_Ni = **1.9183 J/m²**（0.119729 eV/Å²，极差 0.000003 eV/Å²）
   - fmax=0.002：γ_Cu = 1.4976 J/m²（0.093472 eV/Å²）、
     γ_Ni = 1.9180 J/m²（0.119714 eV/Å²），同样近似逐位相同
     （比 fmax=0.01 低约 0.0001–0.0002 J/m²，小两个量级，可作交叉判据）
3. **γ_int 统计量（两个批次都给出）**：
   - fmax=0.01：中位数 0.0573 J/m²，均值 0.0589，极差 0.0049
     （最小 0.0568，最大 0.0617）；按超胞大小分三簇：18/36 原子 → 0.0617，
     54 原子 → 0.0568，72 原子 → 0.0573。**该批次 γ_int 已知系统性偏高**：
     小超胞约 +0.005 J/m²、批次中位数 +0.0006 J/m²（欠收敛所致）。
   - fmax=0.002（**基准**）：中位数 **0.0567 J/m²**，极差 **0.0002**
     （全部候选落在 0.0567–0.0569），标准差 0.0001；三簇合并。
     引用基准时用这一组。
   - 离散归因已由收敛验证**证实**（不再是推断）：收紧 fmax 后 γ_int 极差
     缩小约 25 倍、E_interface/原胞极差缩小约 22 倍（1.983 → 0.089 meV），
     且各簇位移方向与幅度同原先残余力单调对应（A −0.0050 / C −0.0006 /
     B ≈ 0 J/m²）。
4. **参考态口径**：所有行 `reference_state=strained-coherent`，
   `area_source=candidate.area` 且 `area_sub_A2 == area_film_A2 == area_A2`。
5. **文件级校验**：`candidates.csv` 的 SHA-256 为
   `27f77756b595596304573bca9609cd65ee1c32b4ced95327e30c39281876a2fa`
   （fmax=0.01 批次）；fmax=0.002 批次为
   `f63a01b1f0e4bbb3d3d1815229846319844f46af49d50487af66f9e0a624013c`
   （均与各自 `metadata.yaml` 内记录一致）。
   同一进程内重跑 `candidates.csv` 逐字节一致；跨进程时能量按写入精度一致，
   extxyz 坐标末位可能有 ~1e-7 Å 级噪声（torch CPU 线程归约顺序，
   见验证报告 §11），不要对 extxyz 字节做相等断言。

## 4. 已知环境噪声（如实记录，详见 README §9）

- `metadata.yaml` 中 `pymatgen: unknown`：provenance 未抓到版本号，
  实际安装 2026.10.24 且全程在用；真实版本见 `provenance_extra.yaml`
  与 `environment.lock.yml`。
- MACE 权重缓存路径异常（`d:\MSC.Software\Adams\2018\mace\...`），
  换机器复现时缓存路径不同，不影响结果。
- 本批 10 个候选是 `max_area=100 Å²` 下**面积最小的 10 个**，
  且 10 个候选 in-plane 应变完全相同（stretch 2.5823%、shear 0），
  因此这是**同一应变下的超胞一致性检验**，不是低/高应变扫描，
  也不是随机抽样。引用本批数据时须带上该口径。
