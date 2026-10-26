"""S6: TRXScan's pose convention against candidate world 4x4s, on b=0 volumes of the phantom."""
import itertools
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from common import *

poses = np.array([
    [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0],
    [0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 6, 0], [0, 0, 0, 0, 0, 6], [3, -2, 1, 4, -5, 3],
], float)
n = len(poses)
g = gradient_table(np.zeros(n), bvecs=np.zeros((n, 3)))
obj = phantom(100_000)
P = clean_protocol(3.0)
data, aff, sec, _ = run(obj, g, P, clean_artifacts(motion=ts.Motion.from_arrays(poses, None)), context=12)
mag = np.abs(data)
ref = mag[..., 0]
mask = ref > 0.1 * ref.max()
shape = np.array(ref.shape)
print("grid", ref.shape, "seconds", round(sec, 1))

def rot(r, order):
    return Rotation.from_euler(order, r, degrees=True).as_matrix()

def resample(M):
    """Image of the head moved by M (world 4x4): value at x is ref at M^-1 x."""
    ijk = np.indices(ref.shape).reshape(3, -1)
    world = aff[:3, :3] @ ijk + aff[:3, 3:4]
    Minv = np.linalg.inv(M)
    src = Minv[:3, :3] @ world + Minv[:3, 3:4]
    vox = np.linalg.solve(aff[:3, :3], src - aff[:3, 3:4])
    return map_coordinates(ref, vox, order=1, mode="constant").reshape(ref.shape)

centres = {
    "fov(n-1)/2": aff[:3, :3] @ ((shape - 1) / 2) + aff[:3, 3],
    "fov n/2": aff[:3, :3] @ (shape / 2) + aff[:3, 3],
    "world origin": np.zeros(3),
}
orders = {"Rz.Ry.Rx (extrinsic xyz)": "xyz", "Rx.Ry.Rz (extrinsic zyx)": "zyx"}
scores = {}
for (cname, c), (oname, o), inverse in itertools.product(centres.items(), orders.items(), (False, True)):
    errs = []
    for v in range(1, n):
        t, r = poses[v, :3], poses[v, 3:]
        R = rot(r, o)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c - R @ c + t
        if inverse:
            M = np.linalg.inv(M)
        pred = resample(M)
        errs.append(float(np.sqrt(((pred - mag[..., v])[mask | (mag[..., v] > 0.1 * ref.max())] ** 2).mean()) / ref[mask].std()))
    scores[(cname, oname, "inverse" if inverse else "forward")] = errs
for key, errs in sorted(scores.items(), key=lambda kv: np.mean(kv[1]))[:6]:
    print(f"{np.mean(errs):.4f}  {key}  per pose {np.round(errs, 3)}")
moved_vs_ref = [float(np.sqrt(((mag[..., v] - ref)[mask] ** 2).mean()) / ref[mask].std()) for v in range(1, n)]
print("for scale, unresampled difference per pose", np.round(moved_vs_ref, 3))

from scipy.ndimage import center_of_mass
def com_world(img):
    return aff[:3, :3] @ np.array(center_of_mass(img)) + aff[:3, 3]
c0 = com_world(ref)
print("\ncentre-of-mass displacement (mm): measured vs predicted (forward, Rz.Ry.Rx)")
for cname in ("fov(n-1)/2", "fov n/2"):
    c = centres[cname]; errs = []
    for v in range(1, n):
        R = rot(poses[v, 3:], "xyz"); pred = R @ (c0 - c) + c + poses[v, :3] - c0
        meas = com_world(mag[..., v]) - c0
        errs.append(np.linalg.norm(meas - pred))
        if cname == "fov(n-1)/2":
            print(f"  pose {poses[v]}: measured {np.round(meas, 2)}  predicted {np.round(pred, 2)}")
    print(f"  {cname}: |measured - predicted| per pose {np.round(errs, 3)} mm")
