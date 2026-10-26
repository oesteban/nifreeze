"""Rung 3 inherited share with the fixed harness: sensitivity() end to end, every (K, rigid) cell."""
import json, sys, warnings
from pathlib import Path
import attrs
from joblib import Parallel, delayed

S = Path(__file__).parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = int(sys.argv[2])
MODELS = ("gp-loo", "gqi-krr")
MAGNITUDES, COUNTS, SEED, AXIS, DEGREE = (0.5, 1.0, 2.0, 4.0), (16, 64, 96, 192), 0, 1, 1
ORIGINAL = S / "out-fast/sherbrooke.h5"


def run(count, rigid):
    warnings.simplefilter("ignore")
    from nifreeze.analysis import sensitivity as sens
    from nifreeze.data.dmri import DWI

    target = OUT / f"cell-k{count}-r{rigid}.json"
    if target.exists():
        return
    original = DWI.from_filename(ORIGINAL)
    motion = sens.perturbation_plan(len(original), (rigid,), (count,), seeds=(SEED,), center=sens.mask_centroid(original))[0]
    plan = [attrs.evolve(motion, distortion=sens.eddy_distortion(original, e, axis=AXIS, degree=DEGREE, seed=SEED))
            for e in MAGNITUDES]
    rows = []
    for model in MODELS:
        rows += [r | {"model": model, "rigid": rigid} for r in sens.sensitivity(model, original, plan)]
    target.write_text(json.dumps({"rows": rows}))


if __name__ == "__main__":
    cells = [(k, r) for k in COUNTS for r in MAGNITUDES]
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run)(*c) for c in cells)
