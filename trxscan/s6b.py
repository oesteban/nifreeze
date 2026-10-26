import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import center_of_mass
from common import *
obj = phantom(100_000)
for vox in (2.0, 3.0):
    poses = np.array([[0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0],
                      [12, 0, 0, 0, 0, 0], [3, -2, 1, 0, 0, 0], [0, 0, 0, 4, -5, 3], [3, -2, 1, 4, -5, 3],
                      [4, 0, 0, 0.001, 0, 0]], float)
    n = len(poses)
    g = gradient_table(np.zeros(n), bvecs=np.zeros((n, 3)))
    data, aff, *_ = run(obj, g, clean_protocol(vox), clean_artifacts(motion=ts.Motion.from_arrays(poses, None)), context=12)
    mag = np.abs(data)
    com = lambda img: aff[:3, :3] @ np.array(center_of_mass(img)) + aff[:3, 3]
    c0 = com(mag[..., 0])
    print(f"voxel {vox} mm")
    for v in range(1, n):
        print(f"  pose {poses[v]} -> COM shift {np.round(com(mag[..., v]) - c0, 2)}")
