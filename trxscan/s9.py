"""S9: PE along world LR via an oblique grid; frame of written b-vectors; T2 contrast through tissue S0."""
import numpy as np
from common import *
obj_ph = phantom(100_000)
g = table(n_per_shell=3)
for obl in (None, (0.0, 0.0, 90.0)):
    P = clean_protocol(3.0, oblique_deg=obl)
    sim = obj_ph.simulate(g, P, clean_artifacts(), kspace=False, clean_b0=False)
    A = sim.magnitude.affine
    j_dir = A[:3, 1] / np.linalg.norm(A[:3, 1])
    print(f"oblique {obl}: dims {sim.magnitude.shape[:3]} j axis in world {np.round(j_dir, 3)}; pe_axis {sim.pe_axis}")
    sim.to_bids(str(OUT / f"s9_obl{0 if obl is None else 90}"))
    bv = np.loadtxt(OUT / f"s9_obl{0 if obl is None else 90}_dwi.bvec") if (OUT / f"s9_obl{0 if obl is None else 90}_dwi.bvec").exists() else None
    files = sorted(p.name for p in OUT.glob(f"s9_obl{0 if obl is None else 90}*"))
    print("   files:", files)
    for f in OUT.glob(f"s9_obl{0 if obl is None else 90}*.bvec"):
        written = np.loadtxt(f)
        print(f"   {f.name}: written vs prescribed (first DW): {np.round(written[:, 1], 3)} vs {np.round(g.bvecs[1], 3)}")
for t2 in (False, True):
    P = clean_protocol(3.0, te_ms=90.0, t2=t2)
    data, aff, _, sim = run(obj_ph.grid(P), g, P, clean_artifacts())
    b0 = np.abs(data[..., 0])
    o = obj_ph.grid(P)
    fr = {k: np.asarray(o.image(k).dataobj) for k in ("wm", "gm", "csf")}
    means = {k: float(b0[fr[k] > 0.9].mean()) for k in fr}
    print(f"T2 via tissue S0 = {t2}: b0 mean in pure WM {means['wm']:.1f}, GM {means['gm']:.1f}, CSF {means['csf']:.1f}")
