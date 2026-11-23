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
"""Fitted with one model for all volumes, are the fields closer to the truth, and does the
9-parameter pass do better started from them? (#735, Test 7)

Test 4's four noisy copies of the phantom. On each, with Test 4's targets (gp-loo, each
volume left out, predicted from the copy realigned by the head-motion step's estimate
``m̂``):

``hm-field``
    :class:`~nifreeze.registration.eddy_field.EddyFieldFit` (the ``itk`` section of
    :data:`~nifreeze.registration.base.AFFINE`) on every volume, from ``m̂``: a matrix
    ``C`` and an exponent ``p`` of the b-value shared by all volumes, each volume's shift
    along phase encoding, and ``m̂``'s rotations.
``hm-i9-field``
    The 9-parameter pass (ITK, the ``itk`` section of ``AFFINE``) with the same targets,
    each volume registered from its ``hm-field`` estimate: isolates the start.
``field-i9-field``
    The 9-parameter pass with targets predicted from the copy realigned by ``hm-field``'s
    head motion and field (the head motion turns the b-vectors), each volume registered
    from its ``hm-field`` estimate: the two passes, one after the other.
``hm-i9-hm-rerun``
    On ``m2s0e2``, Test 4's delayed pass run again: the check that the targets are Test
    4's.

Read from Test 4: ``hm-i9-hm`` (the delayed pass), ``hm-i9-truth`` (the same from the
truth) and ``pass-e12-u`` (today's pass).

The arms with Test 4's targets are scored in their frame, as Test 4 scores them;
``field-i9-field`` in the frame that all of ``hm-field``'s estimates share against the
truth's whole mapping, as Test 5's rounds. Each row has Test 4's scores
(:func:`eddy_pass_parts._rows`, the new arms as its delayed pass: the stored head motion is
each estimate's split), and ``moved_mean``, the mean displacement (mm) within the brain
between the estimate and its start.

Writes ``<out>/<protocol>.{tsv,json,npz}``, and each step's results to ``<out>/arms/`` (a
step already written is not run again). With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_field_start.py --dataset <dataset> \\
        --test4 <Test 4's results>/arms --out <out> --jobs 6 --threads 6
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import eddy_pass_parts as parts
import nibabel as nb
import numpy as np
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
from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.registration.base import AFFINE
from nifreeze.registration.eddy_field import EddyFieldFit
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

FIT = "hm-field"
"""The fit of one model for all volumes."""
FROM_FIT = ("hm-i9-field", "field-i9-field")
"""The 9-parameter passes started from the fit."""
CHECK = "hm-i9-hm-rerun"
"""Test 4's delayed pass run again, on ``m2s0e2``."""
READ = ("hm-i9-hm", "hm-i9-truth", "pass-e12-u")
"""The arms read from Test 4."""
ARMS = (FIT, *FROM_FIT, CHECK, *READ)
"""Every arm, in the order they run."""


def _path(args: argparse.Namespace, name: str, arm: str) -> Path:
    """Where an arm's results are: Test 4's for the arms read from it."""
    if arm in READ or arm == "cascade":
        return args.test4 / f"{args.protocol}-{name}-{arm}.npz"
    return args.out / "arms" / f"{args.protocol}-{name}-{arm}.npz"


def _fit() -> dict[str, Any]:
    """The fit on every volume of the copy in ``parts._SHARED``, in a process of its own."""
    shared = parts._SHARED
    copy, targets, starts = shared["copy"], shared["targets"], shared["starts"]
    began = perf_counter()
    fit = EddyFieldFit(
        AFFINE, phase_encoding=copy.phase_encoding, seed=SEED, num_threads=shared["threads"]
    )
    for index in sorted(targets):
        fit.add(index, targets[index], copy[index][0], copy.affine, copy.brainmask, starts[index])
    field = fit.fit(np.asarray(copy.gradients))
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
        "shifts": np.array([fit.shifts[i] for i in sorted(fit.shifts)]),
        "evaluations": np.asarray(fit.evaluations),
        "messages": np.asarray(fit.messages),
        "fit_seconds": perf_counter() - began,
    }


def _pass(volumes: list[int], n_volumes: int, jobs: int) -> dict[str, np.ndarray]:
    """The 9-parameter registration of ``volumes`` (``parts._SHARED``), in a pool."""
    estimates = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    seconds = np.full(n_volumes, np.nan)
    failed = np.zeros(n_volumes, dtype=bool)
    iterations = np.zeros((n_volumes, 2), dtype=int)
    with ProcessPoolExecutor(
        jobs, mp_context=get_context("fork"), initializer=parts._warm, initargs=(volumes[0],)
    ) as pool:
        for future in as_completed([pool.submit(parts._register, i) for i in volumes]):
            index, estimate, took, broke, stops = future.result()
            estimates[index], seconds[index], failed[index] = estimate, took, broke
            iterations[index, : len(stops)] = stops
    return {"estimates": estimates, "seconds": seconds, "failed": failed, "iterations": iterations}


def _targets(load: Any, motion: np.ndarray, eddy: np.ndarray | None, volumes: list[int]):
    """Targets predicted from a copy realigned by ``motion`` and ``eddy``, and their time."""
    began = perf_counter()
    training = load()
    training.motion = motion.copy()
    training.eddy = None if eddy is None else eddy.copy()
    targets = held_out_predictions(MODEL, training.realigned(), volumes)
    del training
    return targets, perf_counter() - began


def run_hm(args: argparse.Namespace, name: str) -> None:
    """On Test 4's targets: the fit, the pass started from it, and (on m2s0e2) the check."""
    reference, _, load, pe = parts._load(args, name)
    n_volumes = len(reference)
    volumes = list(range(n_volumes))
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, cascade, None, volumes)
    print(f"{name} hm: targets in {targets_seconds:.0f} s", flush=True)
    parts._SHARED.update(
        copy=copy, targets=targets, starts=cascade, registration="i9", threads=args.threads
    )

    if not _path(args, name, FIT).exists():
        began = perf_counter()
        # A child process, so that no ITK threads exist when the pools below fork
        with ProcessPoolExecutor(1, mp_context=get_context("fork")) as pool:
            fitted = pool.submit(_fit).result()
        np.savez_compressed(
            _path(args, name, FIT),
            **fitted,
            starts=cascade,
            volumes=np.asarray(volumes),
            seconds=np.full(n_volumes, np.nan),
            failed=np.zeros(n_volumes, dtype=bool),
            iterations=np.zeros((n_volumes, 0), dtype=int),
            targets_seconds=targets_seconds,
            wall=perf_counter() - began,
        )
        print(
            f"{name} {FIT}: p = {fitted['exponent']:.3f}, {fitted['evaluations'].tolist()}"
            f" evaluations, {perf_counter() - began:.0f} s",
            flush=True,
        )
    estimates = np.asarray(np.load(_path(args, name, FIT))["estimates"], dtype=float)

    for arm, starts in (("hm-i9-field", estimates), (CHECK, cascade)):
        if (arm == CHECK and name != "m2s0e2") or _path(args, name, arm).exists():
            continue
        began = perf_counter()
        parts._SHARED.update(starts=starts)
        result = _pass(volumes, n_volumes, args.jobs)
        np.savez_compressed(
            _path(args, name, arm),
            **result,
            starts=starts,
            volumes=np.asarray(volumes),
            targets_seconds=targets_seconds,
            wall=perf_counter() - began,
        )
        print(f"{name} {arm}: {perf_counter() - began:.0f} s", flush=True)


def run_field(args: argparse.Namespace, name: str) -> None:
    """The 9-parameter pass after the fit: targets from the copy the fit realigned."""
    reference, _, load, pe = parts._load(args, name)
    n_volumes = len(reference)
    volumes = list(range(n_volumes))
    fitted = np.load(_path(args, name, FIT))
    starts = np.asarray(fitted["estimates"], dtype=float)
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, fitted["motion"], fitted["eddy"], volumes)
    print(f"{name} field: targets in {targets_seconds:.0f} s", flush=True)
    began = perf_counter()
    parts._SHARED.update(copy=copy, targets=targets, starts=starts, registration="i9")
    result = _pass(volumes, n_volumes, args.jobs)
    np.savez_compressed(
        _path(args, name, "field-i9-field"),
        **result,
        starts=starts,
        volumes=np.asarray(volumes),
        targets_seconds=targets_seconds,
        wall=perf_counter() - began,
    )
    print(f"{name} field-i9-field: {perf_counter() - began:.0f} s", flush=True)


def _run_steps(args: argparse.Namespace) -> None:
    """Run each copy's steps not yet written, each in a process of its own."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out)]
    command += ["--test4", str(args.test4), "--jobs", str(args.jobs)]
    command += ["--threads", str(args.threads)]
    for name in args.copies:
        wanted = [FIT, "hm-i9-field"] + ([CHECK] if name == "m2s0e2" else [])
        if not all(_path(args, name, arm).exists() for arm in wanted):
            subprocess.run([*command, "--run", f"hm:{name}"], check=True)
    for name in args.copies:
        if not _path(args, name, "field-i9-field").exists():
            subprocess.run([*command, "--run", f"field:{name}"], check=True)


def _arm_rows(name: str, arm: str, values: Any, context: dict, frames: dict) -> list[dict]:
    """An arm's rows, scored as Test 4 scores them, in its targets' frame."""
    proxy = arm if arm in READ else "hm-i9-hm"
    frame = frames["field"] if arm == "field-i9-field" else frames["hm"]
    rows = parts._rows(name, proxy, values, context | {"offset": frame})
    for row in rows:
        index = row["index"]
        row["arm"] = arm
        if arm == "field-i9-field":
            row["targets"] = "field"
        if arm in (FIT, *FROM_FIT):
            row["registration"] = "field" if arm == FIT else "i9"
            row["start"] = "hm" if arm == FIT else "field"
        row["moved_mean"] = float(
            _distances(
                context["mask"],
                np.asarray(values["estimates"][index]),
                np.asarray(values["starts"][index]),
            ).mean()
        )
    return rows


def _checks(name: str, stored: dict, cascade: np.ndarray, context: dict) -> dict[str, Any]:
    """The fit keeps ``m̂``'s rotations and its motion across phase encoding; on ``m2s0e2``,
    Test 4's delayed pass run again against Test 4's."""
    checks: dict[str, Any] = {}
    mask, u = context["mask"], context["u"]
    if FIT in stored:
        across = np.eye(3) - np.outer(u, u)
        change = stored[FIT]["motion"] @ np.linalg.inv(cascade) - np.eye(4)
        checks[f"{name}/{FIT}/RotationKept"] = float(np.abs(change[:, :3, :3]).max())
        checks[f"{name}/{FIT}/AcrossPE"] = float(np.abs(across @ change[:, :3, 3, None]).max())
    if CHECK in stored and "hm-i9-hm" in stored:
        ours = np.asarray(stored[CHECK]["estimates"])
        theirs = np.asarray(stored["hm-i9-hm"]["estimates"])
        gaps = [float(_distances(mask, a, b).max()) for a, b in zip(ours, theirs, strict=True)]
        checks[f"{name}/{CHECK}"] = {
            "identical": bool(np.array_equal(ours, theirs)),
            "max_mm": max(gaps),
        }
    return checks


def _field(values: Any) -> dict[str, Any]:
    """The fit's model, stages and time."""
    return {
        "Matrix": np.asarray(values["matrix"]).tolist(),
        "Exponent": float(values["exponent"]),
        "BRef": float(values["b_ref"]),
        "Evaluations": np.asarray(values["evaluations"]).tolist(),
        "Messages": [str(message) for message in values["messages"]],
        "FitSeconds": float(values["fit_seconds"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(EDDY_MM))
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--test4", type=Path, required=True, help="Test 4's arms/ folder")
    parser.add_argument("--copies", nargs="+", default=list(COPIES), choices=list(COPIES))
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the models and the fit")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        step, name = args.run.split(":")
        (run_hm if step == "hm" else run_field)(args, name)
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

    rows, arrays, walls, fields = [], {}, {}, {}
    checks: dict[str, Any] = {"PhaseEncoding": pe, "PhaseEncodingDirection": u.tolist()}
    for name in args.copies:
        perturbation, _ = parts._choose(reference, cells, name)
        truth = parts._truth(perturbation, n_volumes)
        cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
        stored = {
            arm: dict(np.load(_path(args, name, arm)))
            for arm in ARMS
            if _path(args, name, arm).exists()
        }
        frames = {"hm": _common(cascade, truth["motion"])}
        if FIT in stored:
            frames["field"] = _common(np.asarray(stored[FIT]["estimates"]), truth["whole"])
            fields[name] = _field(stored[FIT])
        context = base | {
            "truth": truth,
            "splits": [split_phase_encoding(whole, u)[0] for whole in truth["whole"]],
        }
        for arm, values in stored.items():
            arm_rows = _arm_rows(name, arm, values, context, frames)
            rows += arm_rows
            walls[f"{name}/{arm}"] = float(values["wall"])
            if "targets_seconds" in values:
                walls[f"{name}/{arm}/targets"] = float(values["targets_seconds"])
            arrays[f"{name}/{arm}"] = values["estimates"]
            median = np.median([row["whole_frame_mean"] for row in arm_rows])
            print(f"{name} {arm}: median error {median:.3f} mm", flush=True)
        checks |= _checks(name, stored, cascade, context)

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
            "BrainCentroidMm": points.mean(axis=1).tolist(),
            "Noise": NOISE,
            "Seed": SEED,
            "Model": MODEL,
            "Arms": list(ARMS),
            "ReadFromTest4": list(READ),
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
