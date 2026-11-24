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
"""Can a second fit of the field model, on retrained targets, take the place of the
9-parameter pass? (#735, Test 9)

Test 4's four noisy copies of the phantom. The fit
(:class:`~nifreeze.registration.eddy_field.EddyFieldFit`, the ``itk`` section of
:data:`~nifreeze.registration.base.AFFINE`) runs on Test 4's targets, from the head-motion
step's estimate ``m̂``; then the targets are retrained on the copy realigned by the fit,
and the same fit runs again on them, holding ``m̂`` and starting from the first fit's
``C`` and ``p``. It comes in two variants (:data:`VARIANTS`):

``free``
    Each volume's shift along phase encoding estimated: Test 7's fit, read from its
    results.
``derived``
    Each volume's shift derived from the fields' zero point. The model takes that point
    as the centre of the field of view; the phantom's fields are zero half a voxel off it,
    at index ``N // 2``, so the fit is given the phantom's point (:func:`phantom_zero`).

Each variant gives the arms ``fit-<variant>`` (the first fit) and ``refit-<variant>`` (the
second). Read from Test 7: ``chain`` (its 9-parameter pass after the fit), scored with the
shared field removed as ``chain-removed``; from Test 4: ``hm-i9-hm`` (the delayed pass),
scored with the shared field removed as ``hm-i9-hm-removed``, and ``pass-e12-u`` (today's
pass).

Every arm is scored as Test 7 scores it: the fits and the arms with Test 4's targets in
the frame of the head-motion step; the arms with retrained targets in the frame that all
of their first fit's estimates share against the truth's whole mapping.

Writes ``<out>/<protocol>.{tsv,json,npz}``, and each step's results to ``<out>/arms/`` (a
step already written is not run again). With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_field_refit.py --dataset <dataset> \\
        --test4 <Test 4's results>/arms --test7 <Test 7's results>/arms --out <out> \\
        --jobs 6 --threads 6
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import eddy_pass_parts as parts
import nibabel as nb
import numpy as np
from eddy_field_start import _targets
from eddy_pass_rounds import _removed
from eddy_targets import (
    CASCADE,
    NOISE,
    SEED,
    _common,
    _dataset_commit,
    _directions,
    _distances,
    _order,
)
from eddy_targets_expansion import COPIES
from eddy_targets_model import EDDY_MM, MODEL
from experiments import provenance, subset, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.base import AFFINE
from nifreeze.registration.eddy_field import EddyField, EddyFieldFit, field_of_view_centre
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

VARIANTS = ("free", "derived")
"""How each volume's shift along phase encoding is found."""
FROM_TEST7 = {"fit-free": "hm-field", "chain": "field-i9-field"}
"""The arms read from Test 7."""
READ = ("hm-i9-hm", "pass-e12-u")
"""The arms read from Test 4."""
REMOVED = {"hm-i9-hm-removed": "hm-i9-hm", "chain-removed": "chain"}
"""The arms that are another arm's estimates with the shared field removed."""


def phantom_zero(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Where the phantom's fields are zero (RAS+, mm): index ``N // 2``, as TRXScan's FFT
    indexing takes it, half a voxel off the centre of the field of view."""
    middle = np.asarray(shape[:3]) // 2
    return np.asarray(affine, dtype=float)[:3] @ np.r_[middle, 1.0]


def _path(args: argparse.Namespace, name: str, arm: str) -> Path:
    """Where an arm's results are: Test 4's or Test 7's for the arms read from them."""
    if arm in READ or arm == "cascade":
        return args.test4 / f"{args.protocol}-{name}-{arm}.npz"
    if arm in FROM_TEST7:
        return args.test7 / f"{args.protocol}-{name}-{FROM_TEST7[arm]}.npz"
    return args.out / "arms" / f"{args.protocol}-{name}-{arm}.npz"


def _fit() -> dict[str, Any]:
    """One fit on every volume of the copy in ``parts._SHARED``, in a process of its own."""
    shared = parts._SHARED
    copy, targets, held = shared["copy"], shared["targets"], shared["held"]
    began = perf_counter()
    fit = EddyFieldFit(
        AFFINE,
        phase_encoding=copy.phase_encoding,
        seed=SEED,
        num_threads=shared["threads"],
        shifts=shared["variant"],
        zero_point=shared["zero"] if shared["variant"] == "derived" else None,
    )
    for index in sorted(targets):
        fit.add(index, targets[index], copy[index][0], copy.affine, copy.brainmask, held[index])
    field = fit.fit(np.asarray(copy.gradients), start=shared["start"])
    n_volumes = len(copy)
    motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    eddy = motion.copy()
    for index, (head_motion, matrix) in fit.mappings().items():
        motion[index], eddy[index] = head_motion, matrix
    return {
        "estimates": eddy @ motion,
        "motion": motion,
        "eddy": eddy,
        "matrix": field.matrix,
        "exponent": field.exponent,
        "b_ref": field.b_ref,
        "zero_point": np.full(3, np.nan) if fit.zero_point is None else fit.zero_point,
        "shifts": np.array([fit.shifts[i] for i in sorted(fit.shifts)]),
        "evaluations": np.asarray(fit.evaluations),
        "messages": np.asarray(fit.messages),
        "fit_seconds": perf_counter() - began,
    }


def _save_fit(args, name, arm, cascade, targets_seconds, began) -> dict[str, Any]:
    """Run :func:`_fit` in a child process (no ITK threads here), and store its result."""
    with ProcessPoolExecutor(1, mp_context=get_context("fork")) as pool:
        fitted = pool.submit(_fit).result()
    n_volumes = len(cascade)
    np.savez_compressed(
        _path(args, name, arm),
        **fitted,
        starts=cascade,
        volumes=np.arange(n_volumes),
        seconds=np.full(n_volumes, np.nan),
        failed=np.zeros(n_volumes, dtype=bool),
        iterations=np.zeros((n_volumes, 0), dtype=int),
        targets_seconds=targets_seconds,
        wall=perf_counter() - began,
    )
    print(
        f"{name} {arm}: p = {fitted['exponent']:.3f}, {fitted['evaluations'].tolist()}"
        f" evaluations, {perf_counter() - began:.0f} s",
        flush=True,
    )
    return fitted


def run_fit(args: argparse.Namespace, name: str) -> None:
    """On Test 4's targets, the first fit with derived shifts, from the head-motion step."""
    reference, _, load, pe = parts._load(args, name)
    volumes = list(range(len(reference)))
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, cascade, None, volumes)
    print(f"{name} hm: targets in {targets_seconds:.0f} s", flush=True)
    zero = phantom_zero(copy.affine, np.asarray(copy.brainmask).shape)
    parts._SHARED.update(
        copy=copy,
        targets=targets,
        held=cascade,
        threads=args.threads,
        variant="derived",
        zero=zero,
        start=None,
    )
    _save_fit(args, name, "fit-derived", cascade, targets_seconds, perf_counter())


def run_refit(args: argparse.Namespace, name: str, variant: str) -> None:
    """The second fit: targets from the copy the first fit realigned, ``m̂`` held."""
    reference, _, load, pe = parts._load(args, name)
    volumes = list(range(len(reference)))
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    first = np.load(_path(args, name, f"fit-{variant}"))
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, first["motion"], first["eddy"], volumes)
    print(f"{name} {variant}: targets in {targets_seconds:.0f} s", flush=True)
    start = EddyField(first["matrix"], float(first["exponent"]), float(first["b_ref"]))
    parts._SHARED.update(
        copy=copy,
        targets=targets,
        held=cascade,
        threads=args.threads,
        variant=variant,
        zero=phantom_zero(copy.affine, np.asarray(copy.brainmask).shape),
        start=start,
    )
    _save_fit(args, name, f"refit-{variant}", cascade, targets_seconds, perf_counter())


def _run_steps(args: argparse.Namespace) -> None:
    """Run each copy's steps not yet written, each in a process of its own."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out)]
    command += ["--test4", str(args.test4), "--test7", str(args.test7)]
    command += ["--jobs", str(args.jobs), "--threads", str(args.threads)]
    for name in args.copies:
        if not _path(args, name, "fit-derived").exists():
            subprocess.run([*command, "--run", f"fit:{name}"], check=True)
    for name in args.copies:
        for variant in VARIANTS:
            if not _path(args, name, f"refit-{variant}").exists():
                subprocess.run([*command, "--run", f"refit:{name}:{variant}"], check=True)


def _arm_rows(name: str, arm: str, values: Any, context: dict, frame: np.ndarray) -> list:
    """An arm's rows, scored as Test 4 scores its delayed pass, in ``frame``."""
    proxy = arm if arm in READ else "hm-i9-hm"
    rows = parts._rows(name, proxy, values, context | {"offset": frame})
    for row in rows:
        index = row["index"]
        row["arm"] = arm
        row["moved_mean"] = float(
            _distances(
                context["mask"],
                np.asarray(values["estimates"][index]),
                np.asarray(values["starts"][index]),
            ).mean()
        )
    return rows


def _field(values: Any) -> dict[str, Any]:
    """A fit's model, its zero point, its stages and its time."""
    return {
        "Matrix": np.asarray(values["matrix"]).tolist(),
        "Exponent": float(values["exponent"]),
        "BRef": float(values["b_ref"]),
        "ZeroPoint": np.asarray(values.get("zero_point", np.full(3, np.nan))).tolist(),
        "Evaluations": np.asarray(values["evaluations"]).tolist(),
        "Messages": [str(m) for m in values["messages"]],
        "FitSeconds": float(values["fit_seconds"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(EDDY_MM))
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--test4", type=Path, required=True, help="Test 4's arms/ folder")
    parser.add_argument("--test7", type=Path, required=True, help="Test 7's arms/ folder")
    parser.add_argument("--copies", nargs="+", default=list(COPIES), choices=list(COPIES))
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the models and the fit")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        step, *names = args.run.split(":")
        if step == "fit":
            run_fit(args, names[0])
        else:
            run_refit(args, *names)
        return 0

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    _run_steps(args)

    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(reference)
    distorted, _ = parts._choose(reference, cells, "e2")
    pe = "ijk"[distorted.distortion.axis]
    u = phase_encoding_vector(pe, reference.affine)
    mask_data = np.asanyarray(reference.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), reference.affine)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask_data)).T
    order = _order(n_volumes)
    which = dict.fromkeys(subset(args.protocol, "dev", n_volumes), "dev")
    which |= dict.fromkeys(subset(args.protocol, "confirm", n_volumes), "confirm")
    base = {
        "u": u,
        "mask": mask,
        "points": points,
        "shells": np.round(reference.gradients[:, -1], -2),
        "rank": {index: position for position, index in enumerate(order)},
        "directions": _directions(reference.gradients),
        "which": which,
        "all": list(range(n_volumes)),
    }
    volumes = list(range(n_volumes))
    arms = [*READ, "chain", *(f"{kind}-{v}" for v in VARIANTS for kind in ("fit", "refit"))]

    rows, arrays, walls, fields = [], {}, {}, {}
    checks: dict[str, Any] = {
        "PhaseEncoding": pe,
        "PhaseEncodingDirection": u.tolist(),
        "FieldOfViewCentre": field_of_view_centre(reference.affine, mask_data.shape).tolist(),
        "PhantomZero": phantom_zero(reference.affine, mask_data.shape).tolist(),
    }
    for name in args.copies:
        perturbation, _ = parts._choose(reference, cells, name)
        truth = parts._truth(perturbation, n_volumes)
        cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
        stored = {arm: dict(np.load(_path(args, name, arm))) for arm in arms}
        for arm, source in REMOVED.items():
            removed = _removed(np.asarray(stored[source]["estimates"]), reference, pe, volumes)
            stored[arm] = stored[source] | {"estimates": removed}
        frames = {"hm": _common(cascade, truth["motion"])}
        for variant in VARIANTS:
            fitted = stored[f"fit-{variant}"]
            frames[variant] = _common(np.asarray(fitted["estimates"]), truth["whole"])
            fields[f"{name}/fit-{variant}"] = _field(fitted)
            fields[f"{name}/refit-{variant}"] = _field(stored[f"refit-{variant}"])
        # The chain's targets come from the copy that the free first fit realigned
        retrained = {"chain": "free", "chain-removed": "free"}
        retrained |= {f"refit-{v}": v for v in VARIANTS}
        context = base | {
            "truth": truth,
            "splits": [split_phase_encoding(whole, u)[0] for whole in truth["whole"]],
        }
        for arm, values in stored.items():
            frame = frames[retrained[arm]] if arm in retrained else frames["hm"]
            arm_rows = _arm_rows(name, arm, values, context, frame)
            rows += arm_rows
            walls[f"{name}/{arm}"] = float(values["wall"])
            if "targets_seconds" in values:
                walls[f"{name}/{arm}/targets"] = float(values["targets_seconds"])
            arrays[f"{name}/{arm}"] = values["estimates"]
            median = np.median([row["whole_frame_mean"] for row in arm_rows])
            print(f"{name} {arm}: median error {median:.3f} mm", flush=True)
        # Every fit keeps the head-motion step's rotations, and its motion across PE
        across = np.eye(3) - np.outer(u, u)
        for arm in (f"{kind}-{v}" for v in VARIANTS for kind in ("fit", "refit")):
            change = stored[arm]["motion"] @ np.linalg.inv(cascade) - np.eye(4)
            checks[f"{name}/{arm}/RotationKept"] = float(np.abs(change[:, :3, :3]).max())
            checks[f"{name}/{arm}/AcrossPE"] = float(np.abs(across @ change[:, :3, 3, None]).max())

    table, _ = write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copies": {name: COPIES[name] for name in args.copies},
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            # The brain's mean distance to its centroid, as Test 2's
            "MeanRadiusMm": float(
                np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=0).mean()
            ),
            "Noise": NOISE,
            "Seed": SEED,
            "Model": MODEL,
            "Variants": list(VARIANTS),
            "Arms": [*arms, *REMOVED],
            "ReadFromTest4": list(READ),
            "ReadFromTest7": FROM_TEST7,
            "Fields": fields,
            "Order": order,
            "Jobs": args.jobs,
            "Threads": args.threads,
            "ArmSeconds": walls,
            "WallSeconds": perf_counter() - wall,
            "Checks": checks,
            "DatasetCommit": commit,
            "DatasetCommitAtEnd": _dataset_commit(args.dataset),
            "Provenance": provenance() | {"ItkElastix": version("itk-elastix")},
        },
    )
    np.savez_compressed(args.out / f"{args.protocol}.npz", order=np.asarray(order), **arrays)
    print(f"wrote {table}; checks {checks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
