import numpy as np
import nibabel as nb
exec(open("s5c.py").read().split("results = {}")[0])
from nifreeze.analysis.sensitivity import Distortion

def fit_spectral(aligned, perturbed, dist, affine, mask):
    """inherited_distortion with the PE derivative taken spectrally (the EPI shift's own kernel)."""
    n = aligned.shape[dist.axis]
    k = np.fft.fftfreq(n)  # cycles per voxel
    shape = [1, 1, 1]; shape[dist.axis] = n
    spec = np.fft.ifft(np.fft.fft(aligned, axis=dist.axis) * (2j * np.pi * k).reshape(shape), axis=dist.axis).real
    slope = spec[mask] / np.linalg.norm(affine[:3, dist.axis])
    pts = nb.affines.apply_affine(affine, np.argwhere(mask)).T
    values, derivatives = dist.basis(pts)
    design = (values * slope + derivatives * aligned[mask]).T
    return np.linalg.lstsq(design, (perturbed - aligned)[mask], rcond=None)[0]

obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, fibers=[((0.6, 0.0, 0.8), 1.0)])
clean, aff, *_ = run(obj, g, P, clean_artifacts())
frame = aff[:3, :3] / np.linalg.norm(aff[:3, :3], axis=0)
center = aff[:3, :3] @ np.array([N / 2, N / 2, NZ / 2]) + aff[:3, 3]
for thr in (0.05, 0.0):
    mask = np.abs(clean[..., 0]) > thr * np.abs(clean[..., 0]).max()
    print(f"mask threshold {thr}: max displacement -> relative error (x, y, z, oblique @1000), spectral derivative")
    for scale in (1 / 16, 1 / 4, 1, 2):
        e = EDDY * scale
        eddy, *_ = run(obj, g, P, clean_artifacts(eddy=e, eddy_tau_ms=1e12))
        errs, dmax = [], 0.0
        for v in (1, 2, 3, 4):
            gv = bvecs[v] * bvals[v]
            coef = np.concatenate([[0.0], TRT * e * gv * VOX])
            dist = Distortion(magnitude=0.0, axis=1, seed=None, degree=1, center=center, radius=np.full(3, VOX), frame=frame, coefficients=coef[None, :])
            vals, _ = dist.basis(nb.affines.apply_affine(aff, np.argwhere(mask)).T)
            dmax = max(dmax, float(np.abs(coef @ vals).max()))
            fitted = fit_spectral(np.abs(clean[..., v]).astype(float), np.abs(eddy[..., v]).astype(float), dist, aff, mask)
            errs.append(np.linalg.norm(fitted - coef) / np.linalg.norm(coef))
        print(f"  {dmax:5.2f} mm -> {np.round(errs, 4)}")
