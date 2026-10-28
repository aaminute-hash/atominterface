# 复现指南：多候选 Cu(001)|Ni(001) 界面形成能基准

本目录（`docs/validation/cu_ni_001_multi_candidate/`）是一次完整多候选运行的
**原始产物**：`examples/03_interface_energy.py`，atominterface `0.2.1`
（commit `92dc284`），MACE-MP-0 medium，float32，CPU，`fmax=0.01`，
10 个候选（`--max-candidates 10`）。数据未经任何筛选或后处理。

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

## 2. 预期产出与耗时

- **耗时量级：约 25 分钟**（原始运行 24m50s，CPU，10 候选 ≈ 30 次弛豫 +
  若干单点）。其中一部分是每个单点重建 `MACECalculator` 重复
  `torch.jit.load` 44 MB 权重的开销，属已知性能问题（README §9.4）。
- 产物清单（与本目录一致）：

```
candidates.csv        10 行 × 44 列
metadata.yaml         脚本产出的 provenance（原样，未后处理）
provenance_extra.yaml 运行后补充的环境/权重/耗时（与脚本产出分离）
run.log               完整运行日志（含 FIRE 轨迹）
README.md             本批数据的索引与逐候选分析
structures/           每候选 3 个弛豫后 extxyz（rank 前缀）+ 2 个共享体相
```

## 3. 怎么验证自己复现对了

按优先级：

1. **状态列全绿**：`candidates.csv` 中 10 行全部
   `valid=True, status=ok, converged=True, area_consistent=True,
   energies_finite=True`，无失败行。
2. **表面能逐位一致**（超胞尺寸一致性检验，容差 0 即可按写入精度比对）：
   - γ_Cu = **1.4977 J/m²**（0.093476 eV/Å²，10 个候选完全相同）
   - γ_Ni = **1.9183 J/m²**（0.119729 eV/Å²，极差 0.000003 eV/Å²）
3. **γ_int 三簇 + 统计量**：
   - 中位数 **0.0573 J/m²**，均值 0.0589，极差 0.0049（最小 0.0568，最大 0.0617）
   - 按超胞大小分三簇：18/36 原子 → 0.0617，54 原子 → 0.0568，72 原子 → 0.0573。
     该离散已归因为 `fmax=0.01` 的残余收敛误差而非物理（见 README §7 的
     两组对照实验）；**完整 `fmax=0.002` 收敛验证进行中**，完成后以实测数据
     更新本文档与验证报告，暂以 0.0568–0.0573 为最佳估计区间。
4. **参考态口径**：所有行 `reference_state=strained-coherent`，
   `area_source=candidate.area` 且 `area_sub_A2 == area_film_A2 == area_A2`。
5. **文件级校验**：`candidates.csv` 的 SHA-256 为
   `27f77756b595596304573bca9609cd65ee1c32b4ced95327e30c39281876a2fa`。
   同一进程内重跑 `candidates.csv` 逐字节一致；跨进程时能量按写入精度一致，
   extxyz 坐标末位可能有 ~1e-7 Å 级噪声（torch CPU 线程归约顺序，
   见验证报告 §10），不要对 extxyz 字节做相等断言。

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
