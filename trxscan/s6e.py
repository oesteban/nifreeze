import time
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation
from common import *
ph = phantom(100_000)
g1 = gradient_table(np.zeros(1), bvecs=np.zeros((1, 3)))
for label, s0 in (("CSF only", (0.0, 0.0, 1.0)), ("T2 b0", None)):
    P = clean_protocol(3.0) if s0 is None else clean_protocol(3.0, t2=False).replace(tissue_s0=s0)
    ref_obj = ph.grid(P)
    ref, aff, *_ = run(ref_obj, g1, P, clean_artifacts())
    ref = np.abs(ref[..., 0]); com = lambda img: aff[:3, :3] @ np.array(center_of_mass(img)) + aff[:3, 3]
    c0 = com(ref); out = []
    for t in np.arange(0, 6.5, 1.0):
        T = np.eye(4); T[0, 3] = t
        moved = ph.moved(T)
        img, *_ = run(moved.grid(P, like=ref_obj), g1, P, clean_artifacts())
        out.append(round(float(com(np.abs(img[..., 0]))[0] - c0[0]), 2))
    print(f"{label}: Phantom.moved x-translation 0..6 -> COM x shift {out}")
# a rotation about the FOV centre, checked against centres of mass
P = clean_protocol(3.0)
ref_obj = ph.grid(P); ref, aff, *_ = run(ref_obj, g1, P, clean_artifacts()); ref = np.abs(ref[..., 0])
c = aff[:3, :3] @ ((np.array(ref.shape) - 1) / 2) + aff[:3, 3]
c0 = aff[:3, :3] @ np.array(center_of_mass(ref)) + aff[:3, 3]
for pose in ([0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 6, 0], [3, -2, 1, 4, -5, 3]):
    R = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c - R @ c + np.array(pose[:3])
    t0 = time.perf_counter(); img, *_ = run(ph.moved(T).grid(P, like=ref_obj), g1, P, clean_artifacts()); sec = time.perf_counter() - t0
    meas = aff[:3, :3] @ np.array(center_of_mass(np.abs(img[..., 0]))) + aff[:3, 3] - c0
    pred = R @ (c0 - c) + c + np.array(pose[:3]) - c0
    print(f"pose {pose}: COM shift measured {np.round(meas, 2)} predicted {np.round(pred, 2)} ({sec:.1f} s incl. gridding)")
