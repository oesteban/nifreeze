"""Split the combined inherited share into motion (rigid-only fit) and the rest, per volume."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

M, ROOT, E = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])  # out-inherit, out-rung3, out-eddy
motion = pd.DataFrame([r for f in M.glob("motion-*.json") for r in json.loads(f.read_text())["rows"]])
combined = []
for (k, rigid), g in motion.groupby(["count", "rigid"]):
    for e in sorted(g.eddy.unique()):
        combined += [r | {"count": k} for r in json.loads(Path(f"{ROOT}-k{k}/cell-r{rigid}-e{e}.json").read_text())["rows"]]
combined = pd.DataFrame(combined)[["model", "count", "rigid", "eddy", "index", "inherited"]]
alone = pd.DataFrame([r | {"eddy": e} for e in sorted(motion.eddy.unique())
                      for r in json.loads((E / f"cell-AP-d1-m{e}-s0.json").read_text())["rows"]])
alone = alone[["model", "eddy", "index", "inherited"]].rename(columns={"inherited": "eddy_alone"})
t = motion.merge(combined, on=["model", "count", "rigid", "eddy", "index"]).merge(alone, on=["model", "eddy", "index"])
t["shell"] = t.bval.round(-2).astype(int)
t["rest"] = t.inherited - t.motion_inherited

pd.set_option("display.width", 220)
keys = ["model", "count", "rigid", "shell"]
t = t[t.model != "gqi"]
print("Mean over the shell: combined share, motion-only share, the rest, and the share with eddy alone")
print(t.groupby(keys)[["inherited", "motion_inherited", "rest", "eddy_alone"]].mean().round(3).to_string())
print("\nPer volume: r(combined, motion-only) and r(rest, eddy alone), pooled over the eddy magnitudes")
print(t.groupby(keys).apply(lambda g: pd.Series({
    "r(comb, motion)": np.corrcoef(g.inherited, g.motion_inherited)[0, 1],
    "r(rest, eddy alone)": np.corrcoef(g.rest, g.eddy_alone)[0, 1],
}), include_groups=False).round(2).to_string())

print("\nBy eddy magnitude, gp-loo, K=192, 4°/4 mm, b=1000")
g = t[(t.model == "gp-loo") & (t["count"] == 192) & (t.rigid == 4.0) & (t.shell == 1000)]
print(g.groupby("eddy")[["inherited", "motion_inherited", "rest", "eddy_alone"]].mean().round(3).to_string())
