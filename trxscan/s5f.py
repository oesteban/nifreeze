import attrs
import nibabel as nb
import numpy as np
exec(open("s5c.py").read().split("results = {}")[0])
from nifreeze.analysis.sensitivity import Distortion, inherited_distortion
obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, fibers=[((0.6, 0.0, 0.8), 1.0)])
clean, aff, *_ = run(obj, g, P, clean_artifacts())
frame = aff[:3, :3] / np.linalg.norm(aff[:3, :3], axis=0)
center = aff[:3, :3] @ np.array([N / 2, N / 2, NZ / 2]) + aff[:3, 3]
mask = np.abs(clean[..., 0]) > 0.05 * np.abs(clean[..., 0]).max()
print("max displacement (mm) at the mask edge -> relative error of the fit, per volume (x, y, z, oblique @1000)")
for scale in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2):
    e = EDDY * scale
    eddy, *_ = run(obj, g, P, clean_artifacts(eddy=e, eddy_tau_ms=1e12))
    errs, dmax = [], 0.0
    for v in (1, 2, 3, 4):
        gv = bvecs[v] * bvals[v]
        coef = np.concatenate([[0.0], TRT * e * gv * VOX])
        dist = Distortion(magnitude=0.0, axis=1, seed=None, degree=1, center=center, radius=np.full(3, VOX), frame=frame, coefficients=coef[None, :])
        vals, _ = dist.basis(nb.affines.apply_affine(aff, np.argwhere(mask)).T)
        dmax = max(dmax, float(np.abs(coef @ vals).max()))
        fitted = inherited_distortion(np.abs(clean[..., v]), np.abs(eddy[..., v]), dist, aff, mask)
        errs.append(np.linalg.norm(fitted - coef) / np.linalg.norm(coef))
    print(f"  {dmax:5.2f} mm -> {np.round(errs, 3)}")
