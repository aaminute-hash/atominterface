# float64 control run — precision is not the bottleneck

A four-candidate control run whose only difference from the parent batch
(`../`) is precision: `--dtype float64`, everything else identical
(`--fmax 0.01 --max-candidates 4`).

## Why this run exists

The parent batch (and the `fmax_0002_converged/` sweep) established that
γ_int's +0.005 J/m² bias and cluster spread are *numerical*. Two numerical
limits were still candidates: relaxation convergence and float32 round-off.
This run excludes the second one.

## Result

| configuration | γ_int (rank 1–4) |
|---|---|
| float32, `fmax = 0.01` | 0.0617 J/m² |
| **float64, `fmax = 0.01`** | **0.0616 J/m²** |
| float32, `fmax = 0.002` | 0.0567 J/m² |

4/4 `status=ok`. All four candidates give 0.0616 J/m² to the fourth
decimal.

**Double precision buys 0.0001 J/m²; tightening the force gate buys
0.0050 J/m².** The bias is convergence, not round-off.

## Reproducing

See `provenance_extra.yaml` for the exact code state (detached HEAD
`c750688` = PR #16 + PR #17 merged, equivalent to `main` after both land)
and the mandatory pre-flight that asserts the model parameters are really
`torch.float64` — without that check this experiment silently degrades
into a repeat of the float32 run.

Command:

```bash
PYTHONPATH=<worktree>/src python examples/03_interface_energy.py \
    --backend mace --dtype float64 --fmax 0.01 --max-candidates 4 \
    --outdir <outdir>
```

Cost: 6 min 19 s versus about 3 min for the same four candidates in
float32. Not worth it for reproducing this benchmark — use
`float32` + `--fmax 0.002` (see `../fmax_0002_converged/`).

## Files

- `candidates.csv`, `metadata.yaml` — script output, unmodified
- `structures/` — 14 extxyz (4 candidates × 3 + 2 bulk single points)
- `provenance_extra.yaml` — code state, command, pre-flight, timing
