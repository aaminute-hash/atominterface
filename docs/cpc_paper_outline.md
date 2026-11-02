# CPC 科学版论文框架（草案）

> 定位：AtomInterface 投完 JOSS（软件论文）后，沉淀**科学创新**投
> *Computer Physics Communications*（CPC）或同类计算物理期刊。本文档只搭框架，
> 待 JOSS 评审反馈 + FEniCSx 示例落地后再填充正文。
>
> 与 JOSS 论文的区别：JOSS 回答「这是什么工具、怎么用」；CPC 回答「方法上有什么
> 新东西、数值上证明了什么」。CPC 需要明确的**方法贡献 + 可复现的数值证据**。

---

## 标题候选

1. *A fail-closed reliability layer for atomistic interface energies: guarding
   universal machine-learning potentials against out-of-distribution silent failure*
2. *AtomInterface: fail-closed assembly of interface energies and a code-agnostic
   normalization of DFT/MD constitutive data*
3. （精简版）*Reliable interface energies from guarded universal potentials*

## 摘要要点（3–4 句）

- **问题**：界面能是小大数之差，U-MLIP 在训练分布外会「平滑但错误」，单个未收敛
  参考计算会静默污染最终数。
- **方法**：三层可靠性设计——(i) 弛豫守卫（OOD 局部环境隔离 + 写 DFT 单点请求）；
  (ii) fail-closed 能量账本（所有参考能有限且收敛才出数）；(iii) 跨 13 代码的
  统一单位/符号约定桥接。
- **证据**：Cu/Ni(001) `γ_int = 0.059 ± 0.003 J/m²`（VASP 交叉验证）；Cu 弹性常数
  `C11 = 170.01 GPa`（与实验 168 GPa 偏差 1.2%）；13 代码 reader 逐位对拍。

## 1. Introduction

- 界面在电池/涂层/异质结中的中心地位。
- 三个静默失效模式（几何应变但化学无意义 / U-MLIP 外推 / 参考计算未收敛）。
- 现有工具（pymatgen / ASE / MACE / CHGNet / AiiDA）各管一段、缺「可靠性层」。
- **贡献声明**（3 条，与摘要对应）。

## 2. Methods

### 2.1 弛豫守卫（relax-and-guard）
- 守卫「默认开启」而非 opt-in；OOD 判据（局部环境的分布外检测）。
- 触发后：暂停 → 切除违规簇 → 写 DFT 单点输入（而非静默信任模型）。

### 2.2 Fail-closed 能量账本（thermo）
- 界面能 / 分离功 / 内聚参数的组装规则。
- 关键不变量：任一参考能非有限或未收敛 → 单元留空 + 标记 invalid，不代填。
- 终止能参考：面内锁定应变的共格块体 + 局部三点抛物线（消除 n×ΔE_strain 漂移）。

### 2.3 统一物理约定与代码桥接（physics）
- 约定：能量 eV、应力对称 3×3 拉正 eV/Å³、力 eV/Å。
- 13 代码 reader（7 DFT + 6 MD）归一化到同一约定；符号/单位坑逐条列出
  （VASP `sigma→0`、QE Ry→eV、ABACUS 能量已是 eV、SIESTA `E_KS(eV)`、应力压正取负）。

### 2.4 本构管线 + CZM 代码生成（export）
- Green-Lagrange 应变 + Cauchy 应力提取；本构表 `.npz`；UMAT/VUMAT/MOOSE 三端
  内聚律代码生成（双线性 T-δ 律）。

## 3. Results

- **3.1 界面能**：Cu/Ni(001) 候选表 + (111)/(110) 取向归档，`γ_int` 与 VASP 对照。
- **3.2 弹性常数**：小应变 `σ_xx vs ε_xx` 斜率 → `C11`，与实验/独立弛豫对照。
- **3.3 代码桥接正确性**：13 软件 reader 逐位对拍表（≤1e-12 相对）。
- **3.4 守卫有效性**（若有）：构造 OOD 案例，展示守卫拦截 vs 无守卫的错误结果。
- **3.5 可复现性**：821 测试、CI、验证笔记归档。

## 4. Discussion

- 可靠性层的适用范围与局限（U-MLIP 外推检测的边界）。
- 与其他工作流引擎（AiiDA）的分工，不做重复。
- 未来：上游 PR（ASE / pymatgen）、更多体系的 β 验证。

## 5. Conclusion

- 一句话方法贡献 + 一句话数值证据 + 一句话生态定位。

## 6. Data availability

- 代码 + 测试 + 验证笔记（`docs/validation_*.md`）+ 版本 tag；SCNET 原始数据位置。

---

## 投稿策略备忘

- **目标期刊**：CPC 首选（算法/数值导向）；备选 *Comput. Mater. Sci.*、
  *npj Comput. Mater.*。
- **方法创新的锚点**：`fail-closed` 能量账本 + 弛豫守卫是区别于 pymatgen/AiiDA 的
  核心，论文重心必须放在这里，而非「又一个 builder」。
- **弱项预警**：目前守卫有效性的「OOD 拦截 vs 无守卫对照」定量案例尚未系统构造，
  若 CPC 审稿要求，需补一个最小可复现案例（用户红线允许的范围内）。
- **与 JOSS 的关系**：JOSS 软件论文先发拿 DOI/引用；CPC 科学论文引用 JOSS 的
  软件 DOI，形成「工具 + 方法」双引用链。

## 待补（启动 CPC 正文前）

- [ ] 守卫 OOD 拦截定量案例
- [ ] FEniCSx 单元胞示例落地（`examples/04_fenicsx_unit_cell.py`）
- [ ] 各向异性弹性张量完整验证（若正文提 C44）
- [ ] 作者名单扩展确认（若引入合作者）
