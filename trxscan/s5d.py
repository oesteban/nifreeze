import itertools
import numpy as np
exec(open("s5c.py").read().split("results = {}")[0])
obj = ts.objects.from_arrays(wm, None, None, oversample=1, voxel_mm=VOX, fibers=[((0.6, 0.0, 0.8), 1.0)])
clean, _, _, simc = run(obj, g, P, clean_artifacts())
eddy, _, _, sime = run(obj, g, P, clean_artifacts(eddy=EDDY, eddy_tau_ms=1e12))
print("pe_axis", sime.pe_axis, "shape", eddy.shape)
v = 1; gv = bvecs[v] * bvals[v]
best = []
for s_phase, s_ky, rev_t, yoff in itertools.product((1, -1), (1, -1), (False, True), (0.0, 0.5, 1.0)):
    ky = s_ky * (np.arange(N) - N // 2)
    tt = t_s[::-1] if rev_t else t_s
    F = np.exp(-2j * np.pi * np.outer(ky, np.arange(N)) / N)
    Finv = np.exp(2j * np.pi * np.outer(np.arange(N), ky) / N) / N
    pred = np.empty((N, N, NZ), complex)
    for i in range(N):
        for k in range(NZ):
            f = EDDY * (gv[0] * (i - N / 2 + yoff) + gv[1] * (np.arange(N) - N / 2) + gv[2] * (k - NZ / 2))
            pred[i, :, k] = Finv @ ((F * np.exp(s_phase * 2j * np.pi * np.outer(tt, f))) @ clean[i, :, k, v])
    err = float(np.abs(np.abs(pred) - np.abs(eddy[..., v])).max() / np.abs(eddy[..., v]).max())
    best.append((err, s_phase, s_ky, rev_t, yoff))
for b in sorted(best)[:6]:
    print(b)
