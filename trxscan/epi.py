"""TRXScan's EPI eddy operator (single coil, o=1, tau=inf), and a fit of its coefficients."""
import numpy as np


class EpiEddy:
    """Displace along axis 1 as TRXScan's k-space model does: shift = -TRT * field, field in Hz."""

    def __init__(self, shape, t_s, trt_s):
        self.nx, self.ny, self.nz = shape
        n = self.ny
        ky = -(np.arange(n) - n // 2)
        self.F = np.exp(-2j * np.pi * np.outer(ky, np.arange(n)) / n)
        self.Finv = np.exp(2j * np.pi * np.outer(np.arange(n), ky) / n) / n
        self.t = np.asarray(t_s)
        self.trt = trt_s
        self.xc = np.arange(self.nx) - self.nx / 2
        self.yc = np.arange(self.ny) - self.ny / 2
        self.zc = np.arange(self.nz) - self.nz / 2

    def apply(self, image, coef_vox):
        """``coef_vox``: pull displacement in voxels per voxel offset (const, x, y, z), NiFreeze's sign."""
        c = np.asarray(coef_vox, float) / self.trt  # Hz per voxel offset, with field = -pull/TRT ... sign below
        out = np.empty(image.shape, complex)
        for i in range(self.nx):
            for k in range(self.nz):
                field = -(c[0] + c[1] * self.xc[i] + c[2] * self.yc + c[3] * self.zc[k])
                # TRXScan: shift = -TRT * f; pull d = -shift = TRT * f  ->  f = d / TRT; here phase uses +f
                field = -field
                out[i, :, k] = self.Finv @ ((self.F * np.exp(2j * np.pi * np.outer(self.t, field))) @ image[i, :, k])
        return np.abs(out)

    def fit(self, aligned, perturbed, mask, iters=5, eps=1e-3):
        """Coefficients (voxels per voxel offset) of the EPI eddy that best maps aligned onto perturbed."""
        aligned = np.asarray(aligned, float)
        c = np.zeros(4)
        for _ in range(iters):
            base = self.apply(aligned, c)
            J = np.column_stack([
                ((self.apply(aligned, c + eps * np.eye(4)[t]) - base) / eps)[mask] for t in range(4)
            ])
            step = np.linalg.lstsq(J, (perturbed - base)[mask], rcond=None)[0]
            c = c + step
            if np.linalg.norm(step) < 1e-6 * max(np.linalg.norm(c), 1e-12):
                break
        return c
