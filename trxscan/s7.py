"""S7: one timing configuration per process. Args: voxel_mm n_dw mode(clean|eddy|moved) [n_streamlines]."""
import sys, time
import numpy as np
from common import *
vox, n_dw, mode = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
nsl = int(sys.argv[4]) if len(sys.argv) > 4 else 400_000
t0 = time.perf_counter()
ph = phantom(nsl)
t_load = time.perf_counter() - t0
P = clean_protocol(vox)
g = table(n_per_shell=max(n_dw // 2, 1), shells=(1000, 2000)) if n_dw > 1 else table(n_per_shell=1, shells=(1000,))
t0 = time.perf_counter(); obj = ph.grid(P); t_grid = time.perf_counter() - t0
art = clean_artifacts(eddy=1e-4, eddy_tau_ms=1e12) if mode == "eddy" else clean_artifacts()
if mode == "moved":
    T = np.eye(4); T[:3, 3] = (1.3, -0.7, 0.9)
    t0 = time.perf_counter(); obj = ph.moved(T).grid(P, like=obj); t_grid = time.perf_counter() - t0
t0 = time.perf_counter()
sim = obj.simulate(g, P, art, kspace=False, clean_b0=False, chunk=8)
t_sim = time.perf_counter() - t0
print(f"RESULT vox={vox} n_vol={len(g.bvals)} mode={mode} streamlines={nsl} dims={obj.dims} load={t_load:.1f}s grid={t_grid:.1f}s sim={t_sim:.1f}s per_vol={t_sim/len(g.bvals):.2f}s rss={peak_rss_gb():.2f}GB")
