import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import center_of_mass
from common import *
obj = phantom(100_000)
ts_ = np.arange(0, 6.5, 1.0)
for vox in (2.5, 3.0, 4.0):
    poses = np.zeros((len(ts_), 6)); poses[:, 0] = ts_
    g = gradient_table(np.zeros(len(ts_)), bvecs=np.zeros((len(ts_), 3)))
    data, aff, *_ = run(obj, g, clean_protocol(vox), clean_artifacts(motion=ts.Motion.from_arrays(poses, None)), context=12)
    mag = np.abs(data)
    com = lambda img: aff[:3, :3] @ np.array(center_of_mass(img)) + aff[:3, 3]
    c0 = com(mag[..., 0])
    print(f"voxel {vox}: x-translation -> COM x shift", [round(float(com(mag[..., v])[0] - c0[0]), 2) for v in range(len(ts_))])
