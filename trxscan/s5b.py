"""S5b: eddy geometry from the PE edges of a uniform box, per (i, k) column."""
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from common import *

N, NZ, VOX = 32, 12, 2.0
box = np.zeros((N, N, NZ), np.float32)
box[4:28, 8:24, :] = 1.0
vecs = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
g = gradient_table(np.array([0, 1000, 1000, 1000, 2000, 2000, 2000.0]),
                   bvecs=np.array([(0, 0, 0)] + list(vecs.values()) * 2, float))
labels = ["x@1000", "y@1000", "z@1000", "x@2000", "y@2000", "z@2000"]
P = clean_protocol(VOX, t2=False)
TRT = P.sidecar(N)["TotalReadoutTime"]
EDDY = 1.0 / (16 * 1000 * TRT)
obj = ts.objects.from_arrays(box, None, None, oversample=1, voxel_mm=VOX, fibers=[((0.6, 0.0, 0.8), 1.0)])
data, aff, _, _ = run(obj, g, P, clean_artifacts(eddy=EDDY, eddy_tau_ms=1e12))
mag = np.abs(data)

def edges(profile):
    """Sub-voxel positions where the profile crosses half its plateau (rising, falling)."""
    half = 0.5 * profile[12:20].mean()
    up = np.flatnonzero((profile[:-1] < half) & (profile[1:] >= half))[0]
    dn = np.flatnonzero((profile[:-1] >= half) & (profile[1:] < half))[-1]
    lin = lambda k: k + (half - profile[k]) / (profile[k + 1] - profile[k])
    return lin(up), lin(dn)

unit = TRT * EDDY  # voxels of shift per Hz-unit; field = EDDY * g * c
print(f"TRT {TRT} s, eddy {EDDY:.3g}; predicted shift per voxel of offset at b=1000: {-unit*1000:.4f}")
for v, lab in enumerate(labels, start=1):
    rows = []
    for i in range(6, 26, 2):
        for k in range(NZ):
            e0 = edges(mag[i, :, k, 0]); e1 = edges(mag[i, :, k, v])
            for which in (0, 1):
                rows.append((i, e0[which], k, e1[which] - e0[which]))
    r = np.array(rows)
    X = np.column_stack([r[:, 0], r[:, 1], r[:, 2], np.ones(len(r))])
    coef, *_ = np.linalg.lstsq(X, r[:, 3], rcond=None)
    rms = np.sqrt(((r[:, 3] - X @ coef) ** 2).mean())
    b = 1000 if "1000" in lab else 2000
    print(f"{lab}: slope (i, j, k) {np.round(coef[:3], 4)}  intercept {coef[3]:.3f}  rms {rms:.4f}  "
          f"pred slope {-unit*b:.4f}  zero-crossing {np.round(-coef[3]/np.where(np.abs(coef[:3])>1e-4, coef[:3], np.nan), 2)}")

print("\nper-slice mean shift of the rising/falling edges (column i=16)")
for v, lab in enumerate(labels, start=1):
    if not lab.startswith("z"):
        continue
    out = []
    for k in range(NZ):
        e0 = edges(mag[16, :, k, 0]); e1 = edges(mag[16, :, k, v])
        out.append((round(e1[0] - e0[0], 3), round(e1[1] - e0[1], 3)))
    print(lab, out)
print("\nper-column shift (k=6), x gradients")
for v, lab in enumerate(labels, start=1):
    if not lab.startswith("x"):
        continue
    out = []
    for i in range(6, 26, 2):
        e0 = edges(mag[i, :, 6, 0]); e1 = edges(mag[i, :, 6, v])
        out.append((i, round(e1[0] - e0[0], 3), round(e1[1] - e0[1], 3)))
    print(lab, out)
print("\nprofile i=16 k=11 b0 vs z@1000:\n", np.round(mag[16, 4:12, 11, 0], 2), "\n", np.round(mag[16, 4:12, 11, 3], 2))
