# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[elastix]"]
#
# [tool.uv.sources]
# nifreeze = { path = "../..", editable = true }
# ///
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Do real eddy currents follow the global eddy model? (#752, Tests 1 and 2)

The datasets (:func:`names`), each run in a process of its own:

``real``
    DIPY's Sherbrooke three-shell dataset (``sherbrooke_3shell``), phase encoding along
    ``j``, with a brain mask from DIPY's ``median_otsu``.
``<protocol>-m<motion>-e<eddy>``
    A copy of the phantom (``sherbrooke`` or ``hcpya``): every volume rotated by
    ``motion`` degrees and shifted by ``motion`` mm (none for ``m0``), eddy currents whose
    largest displacement in the brain is ``eddy`` mm, and noise.

On each, the head-motion step (gqi-krr then gp-loo, rigid, elastix) runs once. Then, from
its estimates, the methods (:data:`METHODS`):

``A``
    Per-volume registration to gp-loo predictions: ``Estimator("gp-loo",
    estimates="eddy")``, elastix with ``affine.yaml``, as ``--eddy`` runs it.
``B``
    The global model with one matrix ``C`` and one exponent ``p`` (``EddyFieldFit()``),
    with gp-loo predictions.
``Bq``
    The global model with one ``C`` for each shell (``EddyFieldFit(edges=...)``), with
    gp-loo predictions.
``A-gqi``, ``B-gqi``
    ``A`` and ``B`` with gqi-krr predictions.

A dataset already run is resumed: its head-motion step and methods are read back, and
only the methods it lacks run.

The fields' zero point is the centre of the field of view on the real data, and the
phantom's point on the phantoms (#744).

Each volume's residual between two methods is how far apart their fields put each brain
voxel along phase encoding, about the brain's centre: the median over the brain, in mm.
On a phantom, each method's error is the same against the truth's field, and its whole
error is that of head motion and field together, after the offset all the volumes share
against the truth is removed (as #735 scored it).

Writes ``<out>/runs/<name>.{json,npz}`` (the raw estimates are saved before scoring, and
``--score`` scores them again), ``<out>/runs/<name>.tsv`` (one row per volume),
``<out>/logs/<name>.log``, and, gathered, ``<out>/misfit.{tsv,json}``. On CPUs 0-5::

    taskset -c 0-5 python tools/registration/eddy_model_misfit.py \\
        --dataset <phantoms> --out <results> --jobs 2 --threads 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from eddy_targets import _dataset_commit, _remove_offset
from eddy_targets_expansion import NOISE
from experiments import provenance, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.estimator import Estimator
from nifreeze.registration.eddy_field import (
    EddyField,
    EddyFieldFit,
    SegmentedEddyField,
    field_of_view_centre,
    field_slopes,
)
from nifreeze.registration.utils import displacements_within_mask, phase_encoding_vector

SEED = 12345
"""Seed of the estimators."""
CASCADE = ("gqi-krr", "gp-loo")
"""The head-motion step's models, in order."""
METHODS = {
    "A": ("gp-loo", None),
    "B": ("gp-loo", "one"),
    "Bq": ("gp-loo", "shells"),
    "A-gqi": ("gqi-krr", None),
    "B-gqi": ("gqi-krr", "one"),
}
"""Each method's predictions, and its form of the global model (one C, or one C per
shell); per-volume registration has none."""
RESIDUALS = {
    "residual_B": ("A", "B"),
    "residual_Bq": ("A", "Bq"),
    "B_vs_Bq": ("B", "Bq"),
    "residual_B_gqi": ("A-gqi", "B-gqi"),
}
"""The residuals scored: each between the fields of two methods."""
MOTIONS = (0.0, 0.5, 1.0, 2.0, 4.0)
"""Each copy's head motion (degrees and mm)."""
EDDIES = {"sherbrooke": (4.0, 8.0, 16.0), "hcpya": (2.5, 5.0, 10.0)}
"""Each protocol's eddy sizes (mm)."""
REAL_PHASE_ENCODING = "j"
"""The real data's phase-encoding axis (not recorded; the sign does not matter)."""
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""The libraries' thread counts, set to each process's."""


def names() -> list[str]:
    """Every dataset, in the order they run: those that decide Test 1 first."""
    copies = [
        f"{protocol}-m{motion:g}-e{eddy:g}"
        for protocol in EDDIES
        for motion in MOTIONS
        for eddy in EDDIES[protocol]
    ]
    first = [name for name in copies if name.startswith("sherbrooke-m0-")]
    return ["real", *first, *[name for name in copies if name not in first]]


def _parse(name: str) -> tuple[str, float, float]:
    protocol, motion, eddy = name.split("-")
    return protocol, float(motion[1:]), float(eddy[1:])


def _real(out: Path) -> tuple[Any, None, np.ndarray]:
    """DIPY's Sherbrooke data, with a brain mask from ``median_otsu`` (written once)."""
    import dipy.data as dpd
    from dipy.io import read_bvals_bvecs
    from dipy.segment.mask import median_otsu

    from nifreeze.data.dmri.io import from_nii

    dwi_file, bval_file, bvec_file = dpd.get_fnames(name="sherbrooke_3shell")
    mask_file = out / "real-brainmask.nii.gz"
    if not mask_file.exists():
        img = nb.Nifti1Image.from_filename(dwi_file)
        bvals, _ = read_bvals_bvecs(bval_file, bvec_file)
        _, mask = median_otsu(
            np.asanyarray(img.dataobj), vol_idx=np.flatnonzero(bvals < 50).tolist()
        )
        nb.Nifti1Image(mask.astype("uint8"), img.affine).to_filename(mask_file)
    dwi = from_nii(
        dwi_file,
        brainmask_file=mask_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
        phase_encoding=REAL_PHASE_ENCODING,
    )
    assert dwi.brainmask is not None
    zero = field_of_view_centre(dwi.affine, dwi.brainmask.shape)
    return dwi, None, zero


def _phantom(root: Path, name: str) -> tuple[Any, dict[str, np.ndarray], np.ndarray]:
    """A copy of the phantom, its truth and its fields' zero point."""
    protocol, motion, eddy = _parse(name)
    reference, cells = load_phantom(root, protocol, noise=NOISE)
    n_volumes = len(reference)
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == motion
        and perturbation.distortion is not None
        and perturbation.distortion.magnitude == eddy
        and perturbation.count == (n_volumes if motion else 0)
        and perturbation.seed in ((0,) if motion else (None, 0))
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies match {name}, not 1")
    perturbation, load = chosen[0]
    distortion = perturbation.distortion
    assert distortion is not None
    del reference
    dwi = load()
    sidecar = next((root / "sub-60501" / "dwi").glob(f"*_acq-{protocol}_dwi.json"))
    dwi.phase_encoding = json.loads(sidecar.read_text())["PhaseEncodingDirection"]
    truth_motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    for index, matrix in zip(perturbation.moved, perturbation.motions, strict=True):
        truth_motion[index] = matrix
    truth = {"motion": truth_motion, "eddy": distortion.matrices()}
    return dwi, truth, np.asarray(distortion.center, dtype=float)


def _edges(bvals: np.ndarray) -> tuple[float, ...]:
    """The b-values halfway between consecutive shells (b=0 left out)."""
    bvals = np.asarray(bvals, dtype=float)
    shells = np.unique(np.round(bvals[bvals > 50], -2))
    return tuple(float(edge) for edge in (shells[:-1] + shells[1:]) / 2)


def _fitted(field: EddyField | SegmentedEddyField | None) -> Any:
    if field is None:
        return None
    if isinstance(field, EddyField):
        return {"Matrix": field.matrix.tolist(), "Exponent": field.exponent, "BRef": field.b_ref}
    return {"Edges": list(field.edges), "Parts": [_fitted(part) for part in field.parts]}


def run(args: argparse.Namespace, name: str) -> None:
    """Every method on one dataset; the raw estimates are saved, then scored."""
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    dwi, truth, zero = _real(args.out) if name == "real" else _phantom(args.dataset, name)
    loaded = perf_counter() - began
    options = {"omp_nthreads": args.threads, "n_jobs": args.threads, "seed": SEED}
    edges = _edges(dwi.gradients[:, 3])

    estimates: dict[str, np.ndarray]
    if (runs / f"{name}.npz").exists():  # resumed: the head-motion step and methods run
        with np.load(runs / f"{name}.npz") as saved:
            estimates = {
                key: saved[key]
                for key in saved.files
                if key == "head_motion" or key.split("/")[0] in METHODS
            }
        record: dict[str, Any] = json.loads((runs / f"{name}.json").read_text())
        seconds: dict[str, float] = record["Seconds"]
        print(f"{name}: resumed with {sorted({k.split('/')[0] for k in estimates})}", flush=True)
    else:
        seconds = {"load": loaded}
        for model in CASCADE:
            began = perf_counter()
            Estimator(model, registration="elastix").run(dwi, **options)
            seconds[f"motion/{model}"] = perf_counter() - began
            print(f"{name}: head motion, {model}, {seconds[f'motion/{model}']:.0f} s", flush=True)
        assert dwi.motion is not None
        estimates = {"head_motion": dwi.motion.copy()}
        record = {"Name": name, "Edges": list(edges), "ZeroPoint": zero.tolist()}
    head_motion = estimates["head_motion"]

    for method, (model, form) in METHODS.items():
        if f"{method}/eddy" in estimates:
            continue
        dwi.motion, dwi.eddy = head_motion.copy(), None
        fit = None
        if form is None:
            estimator = Estimator(model, estimates="eddy")
        else:
            fit = EddyFieldFit(
                num_threads=args.threads,
                zero_point=zero,
                edges=edges if form == "shells" else None,
            )
            estimator = Estimator(model, estimates="eddy", registration=fit)
        began = perf_counter()
        estimator.run(dwi, **options)
        seconds[method] = perf_counter() - began
        assert dwi.motion is not None and dwi.eddy is not None
        estimates[f"{method}/motion"] = dwi.motion.copy()
        estimates[f"{method}/eddy"] = dwi.eddy.copy()
        if fit is not None:
            record[method] = {
                "Field": _fitted(fit.field),
                "Evaluations": fit.evaluations,
                "Messages": fit.messages,
            }
        print(f"{name}: {method}, {seconds[method]:.0f} s", flush=True)

    record |= {
        "PhaseEncoding": dwi.phase_encoding,
        "Seconds": seconds,
        "Threads": args.threads,
        "Provenance": provenance(),
    }
    # Each launch's provenance, the first included (a resumed dataset has several)
    record.setdefault("Launches", []).append(record["Provenance"])
    arrays: dict[str, Any] = {
        "affine": dwi.affine,
        "mask": np.asanyarray(dwi.brainmask, dtype=bool),
        "gradients": dwi.gradients,
        **estimates,
    }
    if truth is not None:
        arrays |= {f"truth/{key}": value for key, value in truth.items()}
    np.savez_compressed(runs / f"{name}.npz", **arrays)
    (runs / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    score(args, name)


def _brain(affine: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The brain voxels' world coordinates, about the brain's centre, ``(3, n)``."""
    points = nb.affines.apply_affine(affine, np.argwhere(mask)).T
    return points - points.mean(axis=1, keepdims=True)


def _apart(slopes: np.ndarray, other: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Each volume's median distance between two fields' displacements over the brain."""
    return np.array([np.median(np.abs(d @ points)) for d in slopes - other])


def score(args: argparse.Namespace, name: str) -> None:
    """Score a dataset's saved estimates: ``<out>/runs/<name>.tsv`` and its record."""
    runs = args.out / "runs"
    saved = np.load(runs / f"{name}.npz")
    record = json.loads((runs / f"{name}.json").read_text())
    affine, mask, gradients = saved["affine"], saved["mask"], saved["gradients"]
    u = phase_encoding_vector(record["PhaseEncoding"], affine)
    points = _brain(affine, mask)
    methods = [method for method in METHODS if f"{method}/eddy" in saved.files]
    slopes = {method: field_slopes(saved[f"{method}/eddy"], u) for method in methods}
    shells = np.round(gradients[:, 3], -2)
    zero = np.asarray(record["ZeroPoint"])
    world = nb.affines.apply_affine(affine, np.argwhere(mask)).T
    record["EddySizeMm"] = {
        method: float(np.max(np.abs(slopes[method] @ (world - zero[:, np.newaxis]))))
        for method in methods
        if METHODS[method][1] is not None
    }

    columns: dict[str, np.ndarray] = {
        column: _apart(slopes[first], slopes[second], points)
        for column, (first, second) in RESIDUALS.items()
        if first in slopes and second in slopes
    }
    checks: dict[str, float] = {}
    if "truth/eddy" in saved:
        truth_eddy, truth_motion = saved["truth/eddy"], saved["truth/motion"]
        truth_slopes = field_slopes(truth_eddy, u)
        whole = truth_eddy @ truth_motion
        mask_img = nb.Nifti1Image(mask.astype("uint8"), affine)
        checks["TruthResidualMm"] = float(_apart(truth_slopes, truth_slopes, points).max())
        found = EddyField.from_slopes(truth_slopes, gradients)
        checks["TruthModelSlopeResidual"] = float(
            np.abs(found.slopes(gradients) - truth_slopes).max()
        )
        checks["TruthExponent"] = found.exponent
        # The truth, scored as the methods are: zero error
        checks["TruthWholeMm"] = float(
            max(
                np.median(
                    displacements_within_mask(mask_img, nt.linear.Affine(e), nt.linear.Affine(t))
                )
                for e, t in zip(_remove_offset(whole, whole)[:3], whole[:3], strict=True)
            )
        )
        for method in methods:
            columns[f"field_error_{method}"] = _apart(slopes[method], truth_slopes, points)
            estimated = _remove_offset(saved[f"{method}/eddy"] @ saved[f"{method}/motion"], whole)
            columns[f"error_{method}"] = np.array(
                [
                    np.median(
                        displacements_within_mask(
                            mask_img, nt.linear.Affine(e), nt.linear.Affine(t)
                        )
                    )
                    for e, t in zip(estimated, whole, strict=True)
                ]
            )
    record["Checks"] = checks
    (runs / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")

    protocol, motion, eddy = ("real", None, None) if name == "real" else _parse(name)
    with (runs / f"{name}.tsv").open("w") as stream:
        header = ["name", "protocol", "motion", "eddy", "index", "shell", *columns]
        stream.write("\t".join(header) + "\n")
        for index in range(len(shells)):
            values = [name, protocol, motion, eddy, index, shells[index]]
            values += [f"{column[index]:.6g}" for column in columns.values()]
            stream.write("\t".join("" if v is None else str(v) for v in values) + "\n")
    print(f"{name}: scored", flush=True)


def _complete(out: Path, name: str) -> bool:
    """Whether a dataset has every method run and is scored."""
    saved, table = out / "runs" / f"{name}.npz", out / "runs" / f"{name}.tsv"
    if not (saved.exists() and table.exists()):
        return False
    with np.load(saved) as arrays:
        return all(f"{method}/eddy" in arrays.files for method in METHODS)


def _launch(args: argparse.Namespace, name: str) -> int:
    """Run one dataset in a process of its own, its output to ``<out>/logs``."""
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {variable: str(args.threads) for variable in THREAD_VARIABLES}
    command = [sys.executable, str(Path(__file__).resolve()), "--dataset", str(args.dataset)]
    command += ["--out", str(args.out), "--threads", str(args.threads), "--run", name]
    with (logs / f"{name}.log").open("w") as log:
        status = subprocess.run(
            command, env=env, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    print(f"{name}: {'done' if status == 0 else f'failed ({status})'}", flush=True)
    return status


def gather(args: argparse.Namespace) -> None:
    """Every dataset's rows in ``<out>/misfit.tsv``, and the records in ``misfit.json``."""
    rows: list[dict[str, Any]] = []
    records = {}
    for name in names():
        table = args.out / "runs" / f"{name}.tsv"
        if not table.exists():
            continue
        lines = table.read_text().splitlines()
        header = lines[0].split("\t")
        rows += [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]
        records[name] = json.loads((args.out / "runs" / f"{name}.json").read_text())
    columns = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{column: row.get(column, "") for column in columns} for row in rows]
    write_results(
        args.out,
        "misfit",
        rows,
        {
            "Datasets": records,
            "Noise": NOISE,
            "Seed": SEED,
            "DatasetCommit": _dataset_commit(args.dataset),
            "Provenance": provenance(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--jobs", type=int, default=2, help="datasets at a time")
    parser.add_argument("--threads", type=int, default=3, help="threads per dataset")
    parser.add_argument("--only", nargs="+", help="these datasets only")
    parser.add_argument("--score", action="store_true", help="score the saved estimates again")
    parser.add_argument("--gather", action="store_true", help="only gather the scored datasets")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.dataset, args.out = args.dataset.resolve(), args.out.resolve()
    if args.run is not None:
        # A run handed over to another launcher (``<out>/handover``) is skipped
        if (args.out / "handover").exists():
            print(f"{args.run}: handed over, skipped", flush=True)
            return 0
        run(args, args.run)
        return 0
    chosen = args.only or names()
    if args.score:
        for name in chosen:
            if (args.out / "runs" / f"{name}.npz").exists():
                score(args, name)
    elif not args.gather:
        todo = [name for name in chosen if not _complete(args.out, name)]
        with ThreadPoolExecutor(args.jobs) as pool:
            statuses = list(pool.map(lambda name: _launch(args, name), todo))
        failed = [name for name, status in zip(todo, statuses, strict=True) if status]
        if failed:
            print(f"failed: {failed}", flush=True)
    gather(args)
    print(f"wrote {args.out / 'misfit.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
