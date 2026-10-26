"""S5c: reproduce TRXScan's eddy volumes from the no-eddy volumes with the analytic field, by the same k-space sum."""
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from common import *

N, NZ, VOX = 32, 12, 2.0
ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(NZ), indexing="ij")
rng = np.random.default_rng(0)
blob = np.zeros((N, N, NZ))
for _ in range(40):  # smooth random texture inside an ellipsoid
    c = rng.uniform([6, 6, 2], [26, 26, 10]); s = rng.uniform(1.0, 3.0)
    blob += np.exp(-((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2) / (2 * s**2))
wm = np.clip(blob / blob.max(), 0, 1).astype(np.float32)
dirs = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.48, -0.6, 0.64)], float)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
bvecs = np.vstack([np.zeros(3), dirs, dirs]); bvals = np.array([0] + [1000] * 4 + [2000] * 4, float)
g = gradient_table(bvals, bvecs=bvecs)
P = clean_protocol(VOX, t2=False)
TRT = P.sidecar(N)["TotalReadoutTime"]
EDDY = 1.0 / (16 * 1000 * TRT)
t_s = np.asarray(P.readout(N, N).t_ms) / 1000.0
results = {}
for kind, A in (("RAS", None), ("LPS", np.diag([-VOX, -VOX, VOX, 1.0]))):
    if A is not None:
        A = A.copy(); A[:3, 3] = -A[:3, :3] @ np.array([(N - 1) / 2, (N - 1) / 2, (NZ - 1) / 2])
    obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, affine=A, fibers=[((0.6, 0.0, 0.8), 1.0)])
    clean, *_ = run(obj, g, P, clean_artifacts())
    eddy, *_ = run(obj, g, P, clean_artifacts(eddy=EDDY, eddy_tau_ms=1e12))
    ky = np.arange(N) - N // 2
    F = np.exp(-2j * np.pi * np.outer(ky, np.arange(N)) / N)  # (ky, y)
    Finv = np.exp(2j * np.pi * np.outer(np.arange(N), ky) / N) / N  # (y', ky)
    for zname, zc in (("k-nz/2", np.arange(NZ) - NZ / 2), ("k", np.arange(NZ).astype(float)), ("k-(nz-1)/2", np.arange(NZ) - (NZ - 1) / 2)):
        errs = []
        for v in range(1, len(bvals)):
            gv = bvecs[v] * bvals[v]
            pred = np.empty((N, N, NZ), complex)
            for i in range(N):
                for k in range(NZ):
                    f = EDDY * (gv[0] * (i - N / 2) + gv[1] * (np.arange(N) - N / 2) + gv[2] * zc[k])  # Hz, per source y
                    S = (F * np.exp(2j * np.pi * np.outer(t_s, f))) @ clean[i, :, k, v]
                    pred[i, :, k] = Finv @ S
            errs.append(float(np.abs(np.abs(pred) - np.abs(eddy[..., v])).max() / np.abs(eddy[..., v]).max()))
        results[(kind, zname)] = errs
        print(kind, zname, "max rel error per volume:", np.round(errs, 6))
    moved = float(np.abs(np.abs(eddy) - np.abs(clean))[..., 1:].max() / np.abs(clean).max())
    print(kind, "for scale: eddy vs no-eddy max rel difference", round(moved, 4))
