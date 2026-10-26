"""Does the inheritance fit pick up rigid motion? Fit the distortion's terms to rigid-only prediction changes."""
import json, sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

S = Path(__file__).parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
EDDY = Path(sys.argv[2])  # rung 2 outputs: aligned3d-*.npy
N_JOBS = int(sys.argv[3])
MODELS = ("gp-loo", "gqi-krr", "gqi")
EDDIES, SEED, AXIS, DEGREE = (0.5, 1.0, 2.0, 4.0), 0, 1, 1
CELLS = ((192, 1.0), (192, 2.0), (192, 4.0), (64, 4.0))
ORIGINAL = S / "out-fast/sherbrooke.h5"
GRID = S / "out-grid/sherbrooke_grid0.25.h5"


def run_cell(count, rigid):
    warnings.simplefilter("ignore")
    from nifreeze.analysis import sensitivity as sens
    from nifreeze.data.dmri import DWI

    target = OUT / f"motion-k{count}-r{rigid}.json"
    if target.exists():
        return
    original, grid = DWI.from_filename(ORIGINAL), DWI.from_filename(GRID)
    mask = grid.brainmask
    distortions = {e: sens.eddy_distortion(original, e, axis=AXIS, degree=DEGREE, seed=SEED) for e in EDDIES}
    first = distortions[EDDIES[0]]
    for d in distortions.values():  # the fitted terms do not depend on the magnitude
        assert np.allclose(d.center, first.center) and np.allclose(d.radius, first.radius)
        assert np.allclose(d.frame, first.frame)
    plan = sens.perturbation_plan(len(original), (rigid,), (count,), seeds=(SEED,), center=sens.mask_centroid(original))
    perturbed = sens.perturb(original, plan[0], grid=grid)
    rows = []
    for model in MODELS:
        predicted = sens.held_out_predictions(model, perturbed)
        aligned = np.load(EDDY / f"aligned3d-{model}.npy", mmap_mode="r")
        for i in range(len(grid)):
            fitted = sens.inherited_distortion(np.asarray(aligned[i], dtype=float), predicted[i], first, grid.affine, mask)
            for e, d in distortions.items():
                own = d.coefficients[i]
                if not own.any():
                    continue
                rows.append({"model": model, "count": count, "rigid": rigid, "eddy": e, "index": i,
                             "bval": float(grid.gradients[i, -1]), "motion_inherited": float(fitted @ own / (own @ own))})
    target.write_text(json.dumps({"rows": rows}))


if __name__ == "__main__":
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_cell)(*c) for c in CELLS)
