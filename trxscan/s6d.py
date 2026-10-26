import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table
from scipy.ndimage import center_of_mass
from common import *
obj = phantom(100_000)
ts_ = np.arange(0, 6.5, 1.0)
poses = np.zeros((len(ts_), 6)); poses[:, 0] = ts_
g = gradient_table(np.zeros(len(ts_)), bvecs=np.zeros((len(ts_), 3)))
for label, s0 in (("WM only", (1.0, 0.0, 0.0)), ("GM only", (0.0, 1.0, 0.0)), ("CSF only", (0.0, 0.0, 1.0))):
    P = clean_protocol(3.0, t2=False).replace(tissue_s0=s0)
    data, aff, *_ = run(obj, g, P, clean_artifacts(motion=ts.Motion.from_arrays(poses, None)), context=12)
    mag = np.abs(data)
    com = lambda img: aff[:3, :3] @ np.array(center_of_mass(img)) + aff[:3, 3]
    c0 = com(mag[..., 0])
    print(f"{label}: x-translation 0..6 -> COM x shift", [round(float(com(mag[..., v])[0] - c0[0]), 2) for v in range(len(ts_))])
