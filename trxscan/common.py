"""Shared settings for the TRXScan spike: clean protocol, small gradient tables, timing."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
T2 = {"wm": 68.0, "gm": 76.0, "csf": 2000.0}


def clean_protocol(voxel_mm=4.0, te_ms=90.0, t2=True, **kw):
    """Clean acquisition: no oversampling, no phase model, full k-space, one coil."""
    s0 = tuple(float(np.exp(-te_ms / T2[t])) for t in ("wm", "gm", "csf")) if t2 else (1.0, 1.0, 1.0)
    return ts.Protocol.DEFAULT.replace(
        voxel_mm=voxel_mm, oversample=1, phase_model="none", partial_fourier=1.0,
        coils=1, accel=1, pe="AP", te_ms=te_ms, tissue_s0=s0, **kw,
    )


def clean_artifacts(**kw):
    return ts.Artifacts(distortion=False, relaxation=False, **kw)


def table(n_per_shell=6, shells=(1000, 2000), n_b0=1, seed=0):
    """b=0 volumes first, then random unit directions per shell."""
    rng = np.random.default_rng(seed)
    vecs = [np.zeros((n_b0, 3))]
    bvals = [np.zeros(n_b0)]
    for b in shells:
        v = rng.normal(size=(n_per_shell, 3))
        vecs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        bvals.append(np.full(n_per_shell, float(b)))
    return gradient_table(np.concatenate(bvals), bvecs=np.vstack(vecs))


def subset(gtab, index):
    index = np.asarray(index)
    return gradient_table(gtab.bvals[index], bvecs=gtab.bvecs[index])


_PHANTOMS = {}


def phantom(n_streamlines=100_000, seed=0):
    key = (n_streamlines, seed)
    if key not in _PHANTOMS:
        _PHANTOMS[key] = ts.Phantom.load("sub-60501").subsample(n_streamlines, seed=seed)
    return _PHANTOMS[key]


def run(obj, gtab, proto, art, **kw):
    """Simulate; return (complex data (x, y, z, n), affine, seconds)."""
    t0 = time.perf_counter()
    sim = obj.simulate(gtab, proto, art, kspace=False, clean_b0=False, **kw)
    seconds = time.perf_counter() - t0
    return np.asarray(sim.complex), sim.magnitude.affine, seconds, sim


def peak_rss_gb():
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30


def record(name, payload):
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))
