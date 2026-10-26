"""Real eddy currents in Sherbrooke: fit each volume against its shell median, keep what flips with the gradient."""
import warnings
from pathlib import Path
import numpy as np
warnings.simplefilter("ignore")
from nifreeze.analysis import sensitivity as sens
from nifreeze.data.dmri import DWI

S = Path(__file__).parent
grid = DWI.from_filename(S / "out-grid/sherbrooke_grid0.25.h5")
mask = grid.brainmask
pred = np.load(S / "out-eddy/aligned3d-avgdwi.npy", mmap_mode="r")
g = np.asarray(grid.gradients, dtype=float)
dw = np.flatnonzero(g[:, 3] > 100)
amp = np.sqrt(g[dw, 3] / g[dw, 3].max())[:, None]
X = np.column_stack([np.ones(len(dw)), g[dw, :3] * amp])  # intercept, then gx, gy, gz scaled by gradient amplitude
names = ["translation", "∝ x (RL)", "∝ y (AP)", "∝ z (IS)"]
print("radius (mm):", None)
for axis, label in ((1, "PE = AP (y)"), (0, "PE = RL (x)")):
    d = sens.eddy_distortion(grid, 1.0, axis=axis)
    C = np.array([sens.inherited_distortion(np.asarray(pred[i], float), np.asarray(grid.dataobj[i], float), d, grid.affine, mask) for i in dw])
    beta, *_ = np.linalg.lstsq(X, C, rcond=None)
    fit = X[:, 1:] @ beta[1:]
    resid = C - X @ beta
    print(f"\n{label}; radius {np.round(d.radius, 1)} mm; coefficients = displacement along PE at the mask's edge (mm)")
    print(f"{'term':<12}{'R² vs g':>9}{'RMS(g part)':>13}{'RMS(rest)':>11}   mm per unit gradient (gx, gy, gz)")
    for t, name in enumerate(names):
        r2 = 1 - resid[:, t].var() / C[:, t].var()
        print(f"{name:<12}{r2:9.2f}{fit[:, t].std():13.3f}{resid[:, t].std():11.3f}   {np.round(beta[1:, t], 3)}")
