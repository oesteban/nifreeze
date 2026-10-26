"""S5: eddy geometry. Ball centroid shifts along j against f*TRT, per gradient, grid frame, b-value."""
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from common import *

N, NZ, VOX = 32, 12, 2.0
CENTERS = [(i, j, k) for i in (8, 16, 24) for j in (8, 16, 24) for k in (3, 6, 9)]
dirs = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "xyz": (1, 1, 1)}
bvecs = [(0, 0, 0)] + [d for d in dirs.values()] + [(1, 0, 0)]
bvals = [0] + [1000] * 4 + [2000]
labels = ["b0"] + [f"{k}@1000" for k in dirs] + ["x@2000"]
g = gradient_table(np.array(bvals, float), bvecs=np.array(bvecs, float) / np.maximum(np.linalg.norm(bvecs, axis=1, keepdims=True), 1e-9))

ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(NZ), indexing="ij")
balls = np.zeros((N, N, NZ), np.float32)
for c in CENTERS:
    balls += np.exp(-((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2) / (2 * 1.2**2)).astype(np.float32)
balls = np.clip(balls, 0, 1)

def affine_for(kind):
    A = np.diag([VOX, VOX, VOX, 1.0])
    if kind == "LPS":
        A = np.diag([-VOX, -VOX, VOX, 1.0])
    if kind == "oblique":
        t = np.deg2rad(30)
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        A[:3, :3] = R @ np.diag([VOX] * 3)
    A[:3, 3] = -A[:3, :3] @ np.array([(N - 1) / 2, (N - 1) / 2, (NZ - 1) / 2])
    return A

P = clean_protocol(VOX, t2=False)
TRT = P.sidecar(N)["TotalReadoutTime"]
EDDY = 1.0 / (12 * 1000 * TRT)  # about one voxel at 12 voxels off-centre, b=1000
res = {"TRT_s": TRT, "eddy": EDDY}
for kind in ("RAS", "LPS", "oblique"):
    A = affine_for(kind)
    obj = ts.objects.from_arrays(balls, None, None, oversample=1, voxel_mm=VOX, affine=A, fibers=[((0.6, 0.0, 0.8), 1.0)])
    data, aff, _, _ = run(obj, g, P, clean_artifacts(eddy=EDDY, eddy_tau_ms=1e12))
    mag = np.abs(data)
    assert np.allclose(aff, A), (aff, A)
    X = np.array([[c[0], c[1], c[2], 1.0] for c in CENTERS])
    out = {}
    for v, lab in enumerate(labels[1:], start=1):
        shifts = []
        for c in CENTERS:
            win = (slice(c[0] - 3, c[0] + 4), slice(c[1] - 5, c[1] + 6), slice(c[2] - 2, c[2] + 3))
            jw = np.arange(c[1] - 5, c[1] + 6)[None, :, None]
            cj = lambda m: float((m[win] * jw).sum() / m[win].sum())
            shifts.append(cj(mag[..., v]) - cj(mag[..., 0]))
        coef, *_ = np.linalg.lstsq(X, np.array(shifts), rcond=None)
        resid = np.array(shifts) - X @ coef
        # Prediction in voxels per voxel of offset: -TRT * eddy * bval * component (component in grid or world frame)
        b = bvals[v]; vec = np.asarray(bvecs[v], float); vec /= np.linalg.norm(vec)
        R = A[:3, :3] / VOX  # grid axes in world
        out[lab] = {
            "slope_ijk": np.round(coef[:3], 4).tolist(), "zero_crossing_ijk": np.round(-coef[3] / np.where(np.abs(coef[:3]) > 1e-6, coef[:3], np.nan), 2).tolist(),
            "pred_if_world_comp": np.round(-TRT * EDDY * b * vec, 4).tolist(),
            "pred_if_grid_comp": np.round(-TRT * EDDY * b * (R.T @ vec), 4).tolist(),
            "fit_rms_resid_vox": float(np.sqrt((resid**2).mean())),
        }
    res[kind] = out
record("s5", res)
