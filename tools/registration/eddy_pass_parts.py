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
"""Does the eddy-current pass find only what is there, and where does its error come from?
(#735, Test 4)

Four noisy copies of the phantom (:data:`COPIES`: neither head motion nor eddy currents,
head motion only, eddy currents only, both), each registered by every arm. The
registrations (:data:`REGISTRATIONS`):

``e12``
    elastix, :data:`~nifreeze.registration.base.AFFINE` (today's pass).
``i9``
    ITK, the ``itk`` section of ``AFFINE``: head motion, then a field along phase encoding.
``i3``
    ITK, the same with ``motion: fixed``: the start's head motion, then the field only.

The arms of a copy, ``<targets>-<registration>[-<start>]``:

``oracle-<registration>``
    Each volume registered to its noise-free original, from the true head motion ``M``
    (Test 1's oracle).
``hm-<registration>-<start>``
    Targets predicted (gp-loo, each volume left out) from the copy realigned by the
    head-motion step's estimate ``m̂``: the delayed update's targets. From ``m̂`` (``hm``:
    the delayed pass) or from the truth in the targets' frame, ``T F`` (``truth``), with
    ``F`` the transform all of ``m̂`` share against ``M`` (:func:`eddy_targets._common`).
``truth-<registration>-<start>``
    Targets predicted from the copy realigned by the truth ``E M``: the best training data
    a repeated pass could reach. From ``m̂ F^-1`` (``hm``) or from ``T`` (``truth``).
``pass-e12-u``
    Today's pass: the :class:`~nifreeze.estimator.Estimator` with the instant update.

The head-motion step is the cascade of gqi-krr then gp-loo, rigid, elastix, as Test 1 runs
it, on each copy. Each step runs in a process of its own (:func:`main`), and each arm's
registrations in a pool of its own: once the ITK backend has run in a process, elastix's
estimates and times change there (#732).

Each row has Test 1's scores (:func:`eddy_targets._scores`, in the frame of the arm's
targets), Test 3's (:func:`eddy_targets_model._residual` and ``_split_scores``, against the
truth's split ``E' M'``, in which the field's shift along phase encoding is head motion),
and the head motion the pass would store, which turns the b-vectors: ``i9``'s split, the
start's for ``e12`` and ``i3``. Against the true head motion ``M``:

``motion_rotation_deg``
    The angle (degrees) of the stored rotation against ``M``'s.
``motion_across_mean``
    The mean displacement (mm) within the brain between the stored head motion and ``M``,
    across phase encoding. Neither depends on where the split puts the field's shift.
``motion_along_mean``
    The same along phase encoding, against ``M'`` for ``i9`` and ``M`` otherwise.

Writes ``<out>/<protocol>.{tsv,json,npz}`` and ``<out>/<protocol>-cascade.tsv``, and each
step's results to ``<out>/arms/`` (a step already written is not run again). With 6
cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_pass_parts.py --dataset <dataset> \\
        --out <out> --jobs 6 --threads 6
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy as shallow
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import numpy as np
from eddy_targets import (
    CASCADE,
    NOISE,
    SEED,
    _cascade,
    _common,
    _dataset_commit,
    _directions,
    _distances,
    _order,
    _scores,
    _truth_check,
)
from eddy_targets_expansion import COPIES
from eddy_targets_model import EDDY_MM, MODEL, _residual, _split_scores
from experiments import provenance, subset, write_results
from scipy.linalg import polar
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.estimator import Estimator
from nifreeze.registration.base import AFFINE, RegistrationFactory
from nifreeze.registration.itk_backend import get_settings
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

MAGNITUDE = 2.0
"""Rotation (degrees) and translation (mm) of every moved volume (``m2s0``)."""
REGISTRATIONS = ("e12", "i9", "i3")
"""The registrations, as the module's docstring describes them."""
GROUPS = {
    "oracle": ("oracle-e12", "oracle-i9", "oracle-i3"),
    "hm": ("hm-e12-hm", "hm-i9-hm", "hm-i9-truth", "hm-i3-hm", "hm-i3-truth"),
    "truth": ("truth-i9-hm", "truth-i9-truth", "truth-i3-hm", "truth-i3-truth"),
}
"""The arms by the targets they share, which one process predicts once."""
ARMS = tuple(arm for arms in GROUPS.values() for arm in arms) + ("pass-e12-u",)
"""Every arm of a copy."""
TRUTH_TOLERANCE = 1e-9
"""Largest difference (mm) between the matrices and the sidecar's field."""

_SHARED: dict[str, Any] = {}
"""An arm's data, set before its workers fork (copy on write)."""


def _parts(arm: str) -> tuple[str, str, str]:
    """An arm's targets, registration and start."""
    targets, registration, *start = arm.split("-")
    return targets, registration, start[0] if start else "motion"


def _backend(registration: str, phase_encoding: str) -> Any:
    """A registration, as the eddy-current pass builds it (one thread)."""
    if registration == "e12":
        return RegistrationFactory.init("elastix", align_config=AFFINE, seed=SEED, num_threads=1)
    settings = AFFINE if registration == "i9" else get_settings() | {"motion": "fixed"}
    return RegistrationFactory.init(
        "itk", align_config=settings, seed=SEED, num_threads=1, phase_encoding=phase_encoding
    )


def _register(index: int) -> tuple[int, np.ndarray, float, bool, list[int]]:
    """Register volume ``index`` to its target, from its start; ITK's iterations."""
    copy, start = _SHARED["copy"], np.array(_SHARED["starts"][index], dtype=float)
    registration = _backend(_SHARED["registration"], copy.phase_encoding)
    began = perf_counter()
    try:
        estimate = registration(
            _SHARED["targets"][index], copy[index][0], copy.affine, copy.brainmask, index, start
        )
    except RuntimeError:  # elastix's, such as samples mapped outside the moving image
        return index, start, perf_counter() - began, True, []
    iterations = [stop[0] for stop in getattr(registration, "stops", [])]
    return index, np.asarray(estimate.matrix), perf_counter() - began, False, iterations


def _warm(index: int) -> None:
    """Load the backend's modules in a worker, after the fork, outside the timings."""
    _register(index)


def _choose(reference: Any, cells: list, name: str) -> tuple[Any, Any]:
    """A copy's perturbation (:obj:`None` for the reference) and its loader."""
    if name == "unmoved":
        return None, lambda: shallow(reference)
    magnitude, eddy = COPIES[name]
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == magnitude
        and perturbation.seed in ((0,) if magnitude else (None, 0))
        and perturbation.count == (len(reference) if magnitude else 0)
        and (None if perturbation.distortion is None else perturbation.distortion.magnitude)
        == eddy
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies match {name}, not 1")
    return chosen[0]


def _truth(perturbation: Any, n_volumes: int) -> dict[str, np.ndarray]:
    """A copy's head motion ``M``, eddy currents ``E`` and whole mapping ``E M``."""
    motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    eddy = motion.copy()
    if perturbation is not None:
        for index, matrix in zip(perturbation.moved, perturbation.motions, strict=True):
            motion[index] = matrix
        if perturbation.distortion is not None:
            eddy = perturbation.distortion.matrices()
    return {"motion": motion, "eddy": eddy, "whole": eddy @ motion}


def _load(args: argparse.Namespace, name: str) -> tuple[Any, Any, Any, str]:
    """The noisy reference, a copy's perturbation and loader, and the PE direction."""
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    perturbation, load = _choose(reference, cells, name)
    # The PE axis of the eddy currents, which every copy shares
    distorted, _ = _choose(reference, cells, "e2")
    return reference, perturbation, load, "ijk"[distorted.distortion.axis]


def _path(args: argparse.Namespace, name: str, step: str) -> Path:
    return args.out / "arms" / f"{args.protocol}-{name}-{step}.npz"


def _volumes(args: argparse.Namespace, n_volumes: int) -> list[int]:
    return (
        list(range(n_volumes))
        if args.volumes == "all"
        else subset(args.protocol, args.volumes, n_volumes)
    )


def run_cascade(args: argparse.Namespace, name: str) -> None:
    """The head-motion step on a copy, as Test 1 runs it; its head motion by level."""
    _, _, load, _ = _load(args, name)
    seconds: dict[str, float] = {}
    run = {"omp_nthreads": args.threads, "n_jobs": args.threads, "seed": SEED}
    levels = _cascade(load(), run, seconds)
    arrays: dict[str, Any] = {f"cascade/{model}": value for model, value in levels.items()}
    arrays["seconds"] = np.array([seconds[f"cascade/{model}"] for model in CASCADE])
    np.savez_compressed(_path(args, name, "cascade"), **arrays)


def run_pass(args: argparse.Namespace, name: str) -> None:
    """Today's pass on a copy (elastix, the instant update), from the head-motion step."""
    began = perf_counter()
    _, _, load, pe = _load(args, name)
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    data = load()
    data.phase_encoding = pe
    data.motion = cascade.copy()
    Estimator(MODEL, registration="elastix", estimates="eddy", align_config=AFFINE).run(
        data, omp_nthreads=args.threads, n_jobs=args.threads, seed=SEED, num_threads=1
    )
    np.savez_compressed(
        _path(args, name, "pass-e12-u"),
        estimates=np.stack([np.asarray(data.get_transform(i)) for i in range(len(data))]),
        motion=np.asarray(data.motion),
        eddy=np.asarray(data.eddy),
        starts=cascade,
        seconds=np.full(len(data), np.nan),
        failed=np.zeros(len(data), dtype=bool),
        iterations=np.zeros((len(data), 0), dtype=int),
        wall=perf_counter() - began,
    )
    print(f"{name} pass-e12-u: {perf_counter() - began:.0f} s", flush=True)


def run_group(args: argparse.Namespace, name: str, group: str) -> None:
    """The arms of a copy that share their targets, which are predicted once."""
    began = perf_counter()
    reference, perturbation, load, pe = _load(args, name)
    n_volumes = len(reference)
    volumes = _volumes(args, n_volumes)
    truth = _truth(perturbation, n_volumes)
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    offset = _common(cascade, truth["motion"])
    copy = load()
    copy.phase_encoding = pe
    if group == "oracle":
        clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
        targets = {i: np.asarray(clean.dataobj[i], dtype="float32") for i in volumes}
        starts = {"motion": truth["motion"]}
    else:
        training = load()
        moved = perturbation is not None and perturbation.count > 0
        distorted = perturbation is not None and perturbation.distortion is not None
        if group == "hm":
            training.motion = cascade.copy()
            frame = offset
            starts = {"hm": cascade, "truth": truth["whole"] @ frame}
        else:
            training.motion = truth["motion"].copy() if moved else None
            training.eddy = truth["eddy"].copy() if distorted else None
            starts = {"hm": cascade @ np.linalg.inv(offset), "truth": truth["whole"]}
        targets = held_out_predictions(MODEL, training.realigned(), volumes)
        del training
    print(f"{name} {group}: targets in {perf_counter() - began:.0f} s", flush=True)

    for arm in GROUPS[group]:
        if arm not in args.arms or _path(args, name, arm).exists():
            continue
        arm_began = perf_counter()
        _, registration, start = _parts(arm)
        _SHARED.update(copy=copy, targets=targets, starts=starts[start], registration=registration)
        estimates = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
        seconds = np.full(n_volumes, np.nan)
        failed = np.zeros(n_volumes, dtype=bool)
        iterations = np.zeros((n_volumes, 2), dtype=int)
        with ProcessPoolExecutor(
            args.jobs, mp_context=get_context("fork"), initializer=_warm, initargs=(volumes[0],)
        ) as pool:
            for future in as_completed([pool.submit(_register, i) for i in volumes]):
                index, estimate, took, broke, stops = future.result()
                estimates[index], seconds[index], failed[index] = estimate, took, broke
                iterations[index, : len(stops)] = stops
        np.savez_compressed(
            _path(args, name, arm),
            estimates=estimates,
            starts=np.asarray(starts[start]),
            volumes=np.asarray(volumes),
            seconds=seconds,
            failed=failed,
            iterations=iterations,
            wall=perf_counter() - arm_began,
        )
        print(f"{name} {arm}: {perf_counter() - arm_began:.0f} s", flush=True)


def _rotation_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    """The angle (degrees) between two matrices' rotations (their polar factors)."""
    rotations = [polar(matrix[:3, :3])[0] for matrix in (estimated, truth)]
    return float(np.degrees(Rotation.from_matrix(rotations[0] @ rotations[1].T).magnitude()))


def _displacements(estimated: np.ndarray, truth: np.ndarray, points: np.ndarray, u: np.ndarray):
    """The mean displacement (mm) between two matrices in the brain, across and along u."""
    homogeneous = np.vstack([points, np.ones(points.shape[1])])
    moved = ((estimated - truth) @ homogeneous)[:3].T
    along = moved @ u
    across = np.linalg.norm(moved - np.outer(along, u), axis=1)
    return float(across.mean()), float(np.abs(along).mean())


def _rows(name: str, arm: str, stored: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    """An arm's rows on a copy (see the module's docstring)."""
    group, registration, start = _parts(arm)
    start = "hm" if group == "pass" else start
    truth, u, mask, points = context["truth"], context["u"], context["mask"], context["points"]
    frame = context["offset"] if group in ("hm", "pass") else np.eye(4)
    volumes = [int(i) for i in stored["volumes"]] if "volumes" in stored else context["all"]
    estimates = {i: np.asarray(stored["estimates"][i]) for i in volumes}
    scores = _scores(estimates, stored["starts"], truth, mask, points, frame)
    rows = []
    for index in volumes:
        in_frame = estimates[index] @ np.linalg.inv(frame)
        whole, split_motion = truth["whole"][index], context["splits"][index]
        if registration == "i9":
            motion = split_phase_encoding(in_frame, u)[0]
        else:  # the start's head motion, which the pass keeps
            begin = np.asarray(stored["starts"][index]) @ np.linalg.inv(frame)
            motion = split_phase_encoding(begin, u)[0]
        # Along phase encoding, against the truth split as the stored head motion is
        split = registration == "i9" or start == "truth"
        reference = split_motion if split else truth["motion"][index]
        across, _ = _displacements(motion, truth["motion"][index], points, u)
        _, along = _displacements(motion, reference, points, u)
        rows.append(
            {
                "copy": name,
                "arm": arm,
                "targets": group,
                "registration": registration,
                "start": start,
                "index": index,
                "shell": float(context["shells"][index]),
                "rank": context["rank"][index],
                "direction": int(context["directions"][index]),
                "subset": context["which"].get(index, "other"),
                **scores[index],
                **_residual(in_frame, whole, u),
                **_split_scores(in_frame, whole, split_motion, u, mask),
                "motion_rotation_deg": _rotation_deg(motion, truth["motion"][index]),
                "motion_across_mean": across,
                "motion_along_mean": along,
                "seconds": float(stored["seconds"][index]),
                "failed": bool(stored["failed"][index]),
                "iterations": ",".join(str(n) for n in stored["iterations"][index]),
            }
        )
    return rows


def _cascade_rows(name: str, cascade: np.ndarray, context: dict[str, Any]) -> list[dict]:
    """The head-motion step's errors on a copy, in its own frame and without it."""
    truth, u, points = context["truth"], context["u"], context["points"]
    rows = []
    for framed, frame in (("no", np.eye(4)), ("yes", context["offset"])):
        for index in context["all"]:
            motion = cascade[index] @ np.linalg.inv(frame)
            across, along = _displacements(motion, truth["motion"][index], points, u)
            rows.append(
                {
                    "copy": name,
                    "in_frame": framed,
                    "index": index,
                    "shell": float(context["shells"][index]),
                    "subset": context["which"].get(index, "other"),
                    "motion_mean": float(
                        _distances(context["mask"], motion, truth["motion"][index]).mean()
                    ),
                    "motion_rotation_deg": _rotation_deg(motion, truth["motion"][index]),
                    "motion_across_mean": across,
                    "motion_along_mean": along,
                }
            )
    return rows


def _checks(
    name: str, stored: dict[str, Any], context: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """A copy's reproductions of Tests 1 and 3, and the fixed head motion of ``i3``."""
    mask, u = context["mask"], context["u"]
    checks: dict[str, Any] = {}
    for arm, values in stored.items():
        if arm.split("-")[1] != "i3":
            continue
        volumes = [int(i) for i in values["volumes"]]
        # The estimate is the start's head motion, then a field along phase encoding
        across = np.eye(3) - np.outer(u, u)
        checks[f"{name}/{arm}/AcrossPE"] = max(
            float(np.abs(across @ (est @ np.linalg.inv(start) - np.eye(4))[:3]).max())
            for est, start in zip(
                values["estimates"][volumes], values["starts"][volumes], strict=True
            )
        )
    if name == "m2s0e2" and args.test1 is not None:
        recorded = np.load(args.test1)[f"cascade/{CASCADE[-1]}"]
        checks["Test1/cascade/identical"] = bool(np.array_equal(context["cascade"], recorded))
    if name == "m2s0e2" and args.test3 is not None:
        for arm, recorded in (
            ("hm-e12-hm", "pass-e12"),
            ("hm-i9-hm", "pass-i9"),
            ("pass-e12-u", "pass-e12-u"),
            ("oracle-e12", "oracle-e12"),
        ):
            path = args.test3 / f"{args.protocol}-{recorded}.npz"
            if arm not in stored or not path.exists():
                continue
            volumes = context["all"] if arm == "pass-e12-u" else stored[arm]["volumes"]
            ours, theirs = stored[arm]["estimates"][volumes], np.load(path)["estimates"][volumes]
            gaps = [float(_distances(mask, a, b).max()) for a, b in zip(ours, theirs, strict=True)]
            checks[f"Test3/{arm}"] = {
                "identical": bool(np.array_equal(ours, theirs)),
                "max_mm": max(gaps),
            }
    return checks


def _run_steps(args: argparse.Namespace) -> None:
    """Run each copy's steps not yet written, each in a process of its own."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out), "--volumes", args.volumes]
    command += ["--jobs", str(args.jobs), "--threads", str(args.threads), "--arms", *args.arms]
    for name in args.copies:
        steps = [("cascade", "cascade")]
        if "pass-e12-u" in args.arms:
            steps.append(("pass", "pass-e12-u"))
        for step, written in steps:
            if not _path(args, name, written).exists():
                subprocess.run([*command, "--run", f"{step}:{name}"], check=True)
        for group, arms in GROUPS.items():
            wanted = [arm for arm in arms if arm in args.arms]
            if any(not _path(args, name, arm).exists() for arm in wanted):
                subprocess.run([*command, "--run", f"group:{name}:{group}"], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(EDDY_MM))
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--copies", nargs="+", default=list(COPIES), choices=list(COPIES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--volumes", default="all", choices=("all", "dev", "confirm"))
    parser.add_argument("--test1", type=Path, help="Test 1's .npz: its head-motion step")
    parser.add_argument("--test3", type=Path, help="Test 3's arms/ folder: its estimates")
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the cascade and passes")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        step, name, *group = args.run.split(":")
        {"cascade": run_cascade, "pass": run_pass}.get(step, run_group)(args, name, *group)
        return 0

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    _run_steps(args)

    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(reference)
    distorted, _ = _choose(reference, cells, "e2")
    pe = "ijk"[distorted.distortion.axis]
    u = phase_encoding_vector(pe, reference.affine)
    mask_data = np.asanyarray(reference.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), reference.affine)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask_data)).T
    shells = np.round(reference.gradients[:, -1], -2)
    order = _order(n_volumes)
    which = dict.fromkeys(subset(args.protocol, "dev", n_volumes), "dev")
    which |= dict.fromkeys(subset(args.protocol, "confirm", n_volumes), "confirm")
    base = {
        "u": u,
        "mask": mask,
        "points": points,
        "shells": shells,
        "rank": {index: position for position, index in enumerate(order)},
        "directions": _directions(reference.gradients),
        "which": which,
        "all": list(range(n_volumes)),
    }

    rows, cascade_rows, arrays, walls = [], [], {}, {}
    checks: dict[str, Any] = {"PhaseEncoding": pe, "PhaseEncodingDirection": u.tolist()}
    frames: dict[str, Any] = {}
    for name in args.copies:
        perturbation, _ = _choose(reference, cells, name)
        truth = _truth(perturbation, n_volumes)
        if perturbation is not None and perturbation.distortion is not None:
            checks[f"{name}/TruthMm"] = _truth_check(
                perturbation.distortion, truth["motion"], truth["eddy"], points
            )
            if checks[f"{name}/TruthMm"] > TRUTH_TOLERANCE:
                raise RuntimeError(f"{name}: the matrices miss the sidecar's field")
        cascade_file = np.load(_path(args, name, "cascade"))
        cascade = cascade_file[f"cascade/{CASCADE[-1]}"]
        offset = _common(cascade, truth["motion"])
        frames[name] = {
            "RotationDeg": float(np.degrees(Rotation.from_matrix(offset[:3, :3]).magnitude())),
            "TranslationMm": offset[:3, 3].tolist(),
            "CascadeSeconds": cascade_file["seconds"].tolist(),
        }
        context = base | {
            "truth": truth,
            "splits": [split_phase_encoding(whole, u)[0] for whole in truth["whole"]],
            "offset": offset,
            "cascade": cascade,
        }
        stored = {
            arm: dict(np.load(_path(args, name, arm)))
            for arm in args.arms
            if _path(args, name, arm).exists()
        }
        for arm, values in stored.items():
            arm_rows = _rows(name, arm, values, context)
            rows += arm_rows
            walls[f"{name}/{arm}"] = float(values["wall"])
            arrays[f"{name}/{arm}"] = values["estimates"]
            median = np.median([row["whole_frame_mean"] for row in arm_rows])
            print(f"{name} {arm}: median error {median:.3f} mm", flush=True)
        cascade_rows += _cascade_rows(name, cascade, context)
        checks |= _checks(name, stored, context, args)
        arrays[f"{name}/cascade"] = cascade
        arrays |= {f"{name}/truth/{key}": value for key, value in truth.items()}

    write_results(args.out, f"{args.protocol}-cascade", cascade_rows, {"Protocol": args.protocol})
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
            "Cascade": list(CASCADE),
            "Arms": args.arms,
            "Volumes": args.volumes,
            "Order": order,
            "Frames": frames,
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
