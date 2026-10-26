"""Rung 3 inherited share with the fixed harness, and a check of its drift against the earlier cells."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

F, ROOT, E = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])  # out-inherit-fixed, out-rung3, out-eddy
t = pd.DataFrame([r for f in F.glob("cell-*.json") for r in json.loads(f.read_text())["rows"]])
t = t.rename(columns={"distortion": "eddy"})
bvals = {r["index"]: r["bval"] for r in json.loads((E / "cell-AP-d1-m1.0-s0.json").read_text())["rows"]}
t["shell"] = t["index"].map(bvals).round(-2).astype(int)
t = t[t.shell > 100]

old = pd.DataFrame([r | {"count": k} for k in sorted(t["count"].unique())
                    for f in Path(f"{ROOT}-k{k}").glob("cell-*.json") for r in json.loads(f.read_text())["rows"]])
check = t.merge(old, on=["model", "count", "rigid", "eddy", "index"], suffixes=("", "_old"))
print(f"drift vs the earlier rung 3 cells, {len(check)} rows: max |Δ nrmse| = {np.abs(check.nrmse - check.nrmse_old).max():.2e}")

alone = pd.DataFrame([r | {"eddy": e} for e in (0.5, 1.0, 2.0, 4.0)
                      for r in json.loads((E / f"cell-AP-d1-m{e}-s0.json").read_text())["rows"]])
alone["shell"] = alone.bval.round(-2).astype(int)
base = alone[alone.model.isin(t.model.unique()) & (alone.shell > 100)].groupby(["model", "shell"]).inherited.mean()

pd.set_option("display.width", 220)
share = t.groupby(["model", "shell", "count", "rigid"]).inherited.mean().unstack("rigid")
share.insert(0, 0.0, [base[(m, b)] for m, b, _ in share.index])
print("\nInherited share, fixed (eddy 0.5 to 4 mm pooled; rigid 0 is rung 2)")
print(share.round(3).to_string())
print("\nBy eddy magnitude at 4°/4 mm (no longer 1/magnitude)")
print(t[t.rigid == 4.0].groupby(["model", "shell", "count", "eddy"]).inherited.mean().unstack("eddy").round(3).to_string())
print("\nQuantiles over volumes (5/25/50/75/95%), gp-loo, pooled shells and eddy")
q = t[t.model == "gp-loo"].groupby(["count", "rigid"]).inherited.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
print(q.round(2).to_string())
