"""S5e: exact reproduction on every volume and affine; NiFreeze's inherited_distortion recovers the analytic coefficients."""
import numpy as np
import nibabel as nb
exec(open("s5c.py").read().split("results = {}")[0])
from nifreeze.analysis.sensitivity import Distortion, inherited_distortion

ky = -(np.arange(N) - N // 2)
F = np.exp(-2j * np.pi * np.outer(ky, np.arange(N)) / N)
Finv = np.exp(2j * np.pi * np.outer(np.arange(N), ky) / N) / N
out = {}
for kind, A in (("RAS", None), ("LPS", np.diag([-VOX, -VOX, VOX, 1.0])), ("oblique", "obl")):
    if isinstance(A, str):
        t = np.deg2rad(25); R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        A = np.eye(4); A[:3, :3] = R * VOX
    if A is not None:
        A = A.copy(); A[:3, 3] = -A[:3, :3] @ np.array([(N - 1) / 2, (N - 1) / 2, (NZ - 1) / 2])
    obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, affine=A, fibers=[((0.6, 0.0, 0.8), 1.0)])
    clean, aff, *_ = run(obj, g, P, clean_artifacts())
    eddy, *_ = run(obj, g, P, clean_artifacts(eddy=EDDY, eddy_tau_ms=1e12))
    errs, fits = [], []
    frame = aff[:3, :3] / np.linalg.norm(aff[:3, :3], axis=0)
    center = aff[:3, :3] @ np.array([N / 2, N / 2, NZ / 2]) + aff[:3, 3]
    mask = np.abs(clean[..., 0]) > 0.05 * np.abs(clean[..., 0]).max()
    for v in range(1, len(bvals)):
        gv = bvecs[v] * bvals[v]
        pred = np.empty((N, N, NZ), complex)
        for i in range(N):
            for k in range(NZ):
                f = EDDY * (gv[0] * (i - N / 2) + gv[1] * (np.arange(N) - N / 2) + gv[2] * (k - NZ / 2))
                pred[i, :, k] = Finv @ ((F * np.exp(2j * np.pi * np.outer(t_s, f))) @ clean[i, :, k, v])
        errs.append(float(np.abs(np.abs(pred) - np.abs(eddy[..., v])).max() / np.abs(eddy[..., v]).max()))
        coef = np.concatenate([[0.0], TRT * EDDY * gv * VOX])  # pull displacement (mm) per voxel offset, NiFreeze's sign
        d = Distortion(magnitude=0.0, axis=1, seed=None, degree=1, center=center, radius=np.full(3, 1.0) * 1.0,
                       frame=frame, coefficients=coef[None, :]) if False else None
        fits.append(coef)
    out[kind] = errs
    print(kind, "reproduction max rel error:", np.format_float_scientific(max(errs), 2))
    # inherited_distortion on volume x@1000 and xyz@1000, with the analytic basis
    for v in (1, 2, 3, 4):
        gv = bvecs[v] * bvals[v]
        coef = np.concatenate([[0.0], TRT * EDDY * gv * VOX])
        import attrs
        dist = Distortion.__new__(Distortion)
        fields = {a.name for a in attrs.fields(Distortion)}
        kw = dict(magnitude=0.0, axis=1, seed=None, degree=1, center=center, radius=np.ones(3), frame=frame, coefficients=coef[None, :])
        dist = Distortion(**{k: v_ for k, v_ in kw.items() if k in fields})
        # radius in mm: one voxel, so the normalized coordinate is the voxel offset
        dist = attrs.evolve(dist, radius=np.full(3, VOX))
        fitted = inherited_distortion(np.abs(clean[..., v]), np.abs(eddy[..., v]), dist, aff, mask)
        print(f"  {kind} vol {v} analytic {np.round(coef, 4)}  fitted {np.round(fitted, 4)}  rel err {np.linalg.norm(fitted - coef) / np.linalg.norm(coef):.3f}")
