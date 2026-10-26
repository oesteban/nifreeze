"""S8: TRXScan's complex noise vs our generator's; the seed collision."""
import numpy as np
from scipy import stats
from common import *
g = table(n_per_shell=4)
obj = phantom(100_000)
P = clean_protocol(4.0)
sigma = 2.0
clean, *_ = run(obj, g, P, clean_artifacts())
res = {}
for seed in (4, 5, 6):
    noisy, *_ = run(obj, g, P, clean_artifacts(noise=sigma**2, seed=seed))
    d = (noisy - clean).ravel()
    res[f"seed{seed}"] = d
    print(f"seed {seed}: var(Re) {d.real.var():.4f} var(Im) {d.imag.var():.4f} (target {sigma**2}); "
          f"KS vs N(0, sigma) p={stats.kstest(d.real / sigma, 'norm').pvalue:.3f}; mean {d.mean():.4f}")
print("seeds 4 and 5 identical:", np.array_equal(res["seed4"], res["seed5"]), "| seeds 5 and 6 identical:", np.array_equal(res["seed5"], res["seed6"]))
ours = np.random.default_rng([0, 1]).normal(0, sigma, size=(2,) + clean.shape)
ours = (ours[0] + 1j * ours[1]).ravel()
print(f"ours: var(Re) {ours.real.var():.4f}; two-sample KS vs TRXScan Re p={stats.ks_2samp(ours.real, res['seed4'].real).pvalue:.3f}")
mag_t = np.abs(clean + (res["seed4"]).reshape(clean.shape)); mag_o = np.abs(clean + ours.reshape(clean.shape))
inside = np.abs(clean[..., 0]) > 0.2 * np.abs(clean[..., 0]).max()
print(f"magnitude inside brain, b0: TRXScan mean {mag_t[..., 0][inside].mean():.3f} ours {mag_o[..., 0][inside].mean():.3f}; background mean TRXScan {mag_t[..., 0][~inside & (np.abs(clean[...,0])==0)].mean():.3f} ours {mag_o[..., 0][~inside & (np.abs(clean[...,0])==0)].mean():.3f} (Rayleigh {sigma*np.sqrt(np.pi/2):.3f})")
