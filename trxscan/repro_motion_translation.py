"""Sub-voxel translations move CSF (and partly GM) the wrong way in TRXScan's motion path.

TRXScan main 680fc4f4 (mrsim-acq 63fb68a). A b=0 that shows CSF only: the centre of mass of
the moved volume should move by the pose's translation; it moves by 2*round(t/v)*v - t instead
(v = voxel size): the fraction of a voxel is applied with the wrong sign. WM (streamlines) moves
correctly; Phantom.moved(T) + grid(like=) moves everything correctly.
"""

import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import center_of_mass

phantom = ts.Phantom.load("sub-60501").subsample(100_000, seed=0)
trans = np.arange(0.0, 6.5, 1.0)
poses = np.zeros((trans.size, 6))
poses[:, 0] = trans
gtab = gradient_table(np.zeros(trans.size), bvecs=np.zeros((trans.size, 3)))
art = ts.Artifacts(distortion=False, relaxation=False)

for label, s0 in (("WM only", (1.0, 0.0, 0.0)), ("CSF only", (0.0, 0.0, 1.0))):
    proto = ts.Protocol.DEFAULT.replace(
        voxel_mm=3.0, oversample=1, phase_model="none", tissue_s0=s0
    )
    sim = phantom.simulate(
        gtab, proto, art.replace(motion=ts.Motion.from_arrays(poses, None)),
        kspace=False, clean_b0=False, context=12,
    )
    mag, aff = np.asarray(sim.magnitude.dataobj), sim.magnitude.affine

    def com(v):
        return aff[:3, :3] @ np.array(center_of_mass(mag[..., v])) + aff[:3, 3]

    print(label, "x translation", trans.tolist())
    print("  centre-of-mass x shift", [round(float(com(v)[0] - com(0)[0]), 2) for v in range(trans.size)])
