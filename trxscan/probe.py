import numpy as np
from common import *
g = table()
obj = phantom(100_000)
data, aff, sec, sim = run(obj, g, clean_protocol(4.0), clean_artifacts())
print("shape", data.shape, data.dtype, "seconds", round(sec, 1), "rss GB", round(peak_rss_gb(), 2))
print("affine\n", np.round(aff, 2))
mag = np.abs(data); ph = np.angle(data)
inside = mag[..., 0] > 0.05 * mag[..., 0].max()
print("phase inside: max |phase| per volume", np.round(np.abs(ph[inside]).max(0), 4))
print("b0 mean in brain", mag[..., 0][inside].mean(), "sidecar TRT", clean_protocol(4.0).sidecar(data.shape[1]).get("TotalReadoutTime"))
print("readout", sim.readout.total_readout_ms if hasattr(sim, "readout") else None)
