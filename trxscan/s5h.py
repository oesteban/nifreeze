import numpy as np
exec(open("s5c.py").read().split("results = {}")[0])
from epi import EpiEddy
obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, fibers=[((0.6, 0.0, 0.8), 1.0)])
clean, aff, *_ = run(obj, g, P, clean_artifacts())
op = EpiEddy(clean.shape[:3], t_s, TRT)
mask = np.abs(clean[..., 0]) > 0.05 * np.abs(clean[..., 0]).max()
for scale in (1 / 16, 1 / 4, 1, 2):
    e = EDDY * scale
    eddy, *_ = run(obj, g, P, clean_artifacts(eddy=e, eddy_tau_ms=1e12))
    lin, gn = [], []
    for v in (1, 2, 3, 4):
        gv = bvecs[v] * bvals[v]
        coef = np.concatenate([[0.0], TRT * e * gv])  # voxels per voxel offset
        assert np.allclose(op.apply(clean[..., v], coef), np.abs(eddy[..., v]), rtol=0, atol=1e-4 * np.abs(eddy).max())
        lin.append(np.linalg.norm(op.fit(np.abs(clean[..., v]), np.abs(eddy[..., v]), mask, iters=1) - coef) / np.linalg.norm(coef))
        gn.append(np.linalg.norm(op.fit(np.abs(clean[..., v]), np.abs(eddy[..., v]), mask, iters=6) - coef) / np.linalg.norm(coef))
    print(f"max shift {scale * 16 * 1000 * TRT * EDDY * 1:.3f} vox at b=1000 edge: linear {np.round(lin, 4)}  Gauss-Newton {np.round(gn, 5)}")
