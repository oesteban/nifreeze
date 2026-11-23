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
"""On HCP-YA, does one delayed 9-parameter pass with the shared field removed beat today's
pass at every shell, and is its loss at b=3000 the registration's sampling? (#735, Test 6)

HCP-YA's copy with head motion and eddy currents (``m2s0e2``), as Test 3 ran it: the same
noise, and Test 3's head-motion step and arms (``--test3``). The registrations:

``i9``
    ITK, the ``itk`` section of :data:`~nifreeze.registration.base.AFFINE` as packaged:
    its last stage samples 17,000 points.
``i9s``
    The same, with the last stage's samples scaled to the share of the brain that they are
    on Sherbrooke (:func:`samples`, from the two phantoms' brain masks).

The arms:

``hm-<registration>-hm``
    The delayed pass: targets (gp-loo, each volume left out) predicted once from the copy
    realigned by the head-motion step, and registrations starting from the step's
    estimates. ``hm-i9-hm`` is Test 3's ``pass-i9`` again, as a check.
``oracle-i9s``
    Each volume registered to its noise-free original, from its true head motion.
``pass-i9-u``
    The 9-parameter pass with the instant update (the :class:`~nifreeze.estimator.Estimator`),
    from the head-motion step, with 17,000 samples.
``<arm>-removed``
    An arm's estimates with the shared field removed
    (:func:`~nifreeze.registration.utils.remove_shared_field`, per shell, about the
    brain's centroid), as Test 5 removes it.

Read from Test 3: ``pass-e12-u`` (today's pass), ``pass-i9``, ``oracle-e12`` and
``oracle-i9``. Each row has Test 4's scores (:func:`eddy_pass_parts._rows`), in the frame
of the arm's targets. Each step runs in a process of its own (:func:`main`). Writes
``<out>/<protocol>.{tsv,json,npz}``, and each arm to ``<out>/arms/``; a step already
written is not run again. With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_pass_hcpya.py --dataset <dataset> \\
        --test3 <eddy-model>/arms --out <out> --jobs 6 --threads 6
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
import eddy_pass_rounds as rounds
import eddy_targets_model as model
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
    _truth_check,
)
from eddy_targets_model import EDDY_MM, MODEL
from experiments import provenance, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.estimator import Estimator
from nifreeze.registration.base import AFFINE, RegistrationFactory
from nifreeze.registration.itk_backend import get_settings
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

NAME = "m2s0e2"
"""The copy, Test 3's."""
GROUPS = {"hm": ("hm-i9-hm", "hm-i9s-hm"), "oracle": ("oracle-i9s",)}
"""The arms run, by the targets they share."""
INSTANT = "pass-i9-u"
"""The arm with the instant update, run by the Estimator."""
READ = ("pass-e12-u", "pass-i9", "oracle-e12", "oracle-i9")
"""The arms read from Test 3."""
DERIVED = {f"{arm}-removed": arm for arm in ("pass-i9", "hm-i9-hm", "hm-i9s-hm", INSTANT)}
"""The arms with the shared field removed, and the arm each comes from."""
PROXY = {
    "pass-i9": "hm-i9-hm",
    "hm-i9s-hm": "hm-i9-hm",
    "oracle-i9s": "oracle-i9",
    INSTANT: "hm-i9-hm",
}
"""How Test 4's scoring reads an arm: its targets, registration and start."""
REFERENCE = "sherbrooke"
"""The protocol whose share of the brain the scaled samples keep."""

_SHARED: dict[str, Any] = {}
"""An arm's data, set before its workers fork (copy on write)."""


def samples(dataset: Path, protocol: str) -> dict[str, Any]:
    """The last stage's samples, scaled by the brain's voxels against :data:`REFERENCE`'s."""
    count = int(get_settings()["stages"][-1]["samples"])
    voxels = {}
    for name in (REFERENCE, protocol):
        clean, _ = load_phantom(dataset, name, counts=[])
        voxels[name] = int(np.asanyarray(clean.brainmask, dtype=bool).sum())
    return {
        "Packaged": count,
        "Scaled": round(count * voxels[protocol] / voxels[REFERENCE]),
        "BrainVoxels": voxels,
    }


def _backend(registration: str, phase_encoding: str, scaled: int) -> Any:
    """``i9`` as Test 4 builds it, or ``i9s`` with the last stage's samples scaled."""
    if registration == "i9":
        return parts._backend("i9", phase_encoding)
    settings = get_settings()
    settings["stages"][-1]["samples"] = scaled
    return RegistrationFactory.init(
        "itk", align_config=settings, seed=SEED, num_threads=1, phase_encoding=phase_encoding
    )


def _register(index: int) -> tuple[int, np.ndarray, float, bool, list[int]]:
    """Register volume ``index`` to its target, from its start; ITK's iterations."""
    copy, start = _SHARED["copy"], np.array(_SHARED["starts"][index], dtype=float)
    registration = _backend(_SHARED["registration"], copy.phase_encoding, _SHARED["samples"])
    began = perf_counter()
    try:
        estimate = registration(
            _SHARED["targets"][index], copy[index][0], copy.affine, copy.brainmask, index, start
        )
    except RuntimeError:
        return index, start, perf_counter() - began, True, []
    iterations = [stop[0] for stop in getattr(registration, "stops", [])]
    return index, np.asarray(estimate.matrix), perf_counter() - began, False, iterations


def _warm(index: int) -> None:
    """Load the backend's modules in a worker, after the fork, outside the timings."""
    _register(index)


def _load(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    """The noisy reference, the copy's perturbation and loader, and the PE direction."""
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    perturbation, load = model._choose(cells, len(reference), EDDY_MM[args.protocol])
    return reference, perturbation, load, "ijk"[perturbation.distortion.axis]


def _path(args: argparse.Namespace, arm: str) -> Path:
    if arm in READ or arm == "cascade":
        return args.test3 / f"{args.protocol}-{arm}.npz"
    return args.out / "arms" / f"{args.protocol}-{NAME}-{arm}.npz"


def run_group(args: argparse.Namespace, group: str) -> None:
    """The arms that share their targets, which are predicted once."""
    began = perf_counter()
    reference, perturbation, load, pe = _load(args)
    n_volumes = len(reference)
    volumes = parts._volumes(args, n_volumes)
    truth = parts._truth(perturbation, n_volumes)
    cascade = np.load(_path(args, "cascade"))[f"cascade/{CASCADE[-1]}"]
    copy = load()
    copy.phase_encoding = pe
    if group == "oracle":
        clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
        targets = {i: np.asarray(clean.dataobj[i], dtype="float32") for i in volumes}
        starts = truth["motion"]
    else:
        training = load()
        training.motion = cascade.copy()
        targets = held_out_predictions(MODEL, training.realigned(), volumes)
        starts = cascade
        del training
    print(f"{group}: targets in {perf_counter() - began:.0f} s", flush=True)

    for arm in GROUPS[group]:
        if _path(args, arm).exists():
            continue
        arm_began = perf_counter()
        registration = arm.split("-")[1]
        _SHARED.update(
            copy=copy,
            targets=targets,
            starts=starts,
            registration=registration,
            samples=args.samples,
        )
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
            _path(args, arm),
            estimates=estimates,
            starts=np.asarray(starts),
            volumes=np.asarray(volumes),
            seconds=seconds,
            failed=failed,
            iterations=iterations,
            wall=perf_counter() - arm_began,
        )
        print(f"{arm}: {perf_counter() - arm_began:.0f} s", flush=True)


def run_instant(args: argparse.Namespace) -> None:
    """The 9-parameter pass with the instant update, from the head-motion step."""
    began = perf_counter()
    _, _, load, pe = _load(args)
    cascade = np.load(_path(args, "cascade"))[f"cascade/{CASCADE[-1]}"]
    data = load()
    data.phase_encoding = pe
    data.motion = cascade.copy()
    Estimator(MODEL, registration="itk", estimates="eddy", align_config=AFFINE).run(
        data, omp_nthreads=args.threads, n_jobs=args.threads, seed=SEED, num_threads=1
    )
    n_volumes = len(data)
    np.savez_compressed(
        _path(args, INSTANT),
        estimates=np.stack([np.asarray(data.get_transform(i)) for i in range(n_volumes)]),
        motion=np.asarray(data.motion),
        eddy=np.asarray(data.eddy),
        starts=cascade,
        seconds=np.full(n_volumes, np.nan),
        failed=np.zeros(n_volumes, dtype=bool),
        iterations=np.zeros((n_volumes, 0), dtype=int),
        wall=perf_counter() - began,
    )
    print(f"{INSTANT}: {perf_counter() - began:.0f} s", flush=True)


def _read(args: argparse.Namespace, arm: str, starts: np.ndarray) -> dict[str, Any]:
    """An arm of Test 3's, with what Test 4's scoring reads."""
    values = dict(np.load(_path(args, arm)))
    n_volumes = len(values["estimates"])
    values.setdefault("starts", starts)
    values.setdefault("iterations", np.zeros((n_volumes, 0), dtype=int))
    return values


def _run_steps(args: argparse.Namespace, scaled: int) -> None:
    """Run each step not yet written, each in a process of its own."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--test3", str(args.test3), "--out", str(args.out)]
    command += ["--volumes", args.volumes, "--jobs", str(args.jobs)]
    command += ["--threads", str(args.threads), "--samples", str(scaled)]
    for group, arms in GROUPS.items():
        if any(not _path(args, arm).exists() for arm in arms):
            subprocess.run([*command, "--run", group], check=True)
    if args.instant and not _path(args, INSTANT).exists():
        subprocess.run([*command, "--run", "instant"], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="hcpya", choices=("hcpya",))
    parser.add_argument("--test3", type=Path, required=True, help="Test 3's arms/ folder")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--volumes", default="all", choices=("all", "dev", "confirm"))
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the model")
    parser.add_argument("--no-instant", dest="instant", action="store_false")
    parser.add_argument("--samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        run_instant(args) if args.run == "instant" else run_group(args, args.run)
        return 0

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    scaled = samples(args.dataset, args.protocol)
    print(f"samples: {scaled}", flush=True)
    _run_steps(args, scaled["Scaled"])

    reference, perturbation, load, pe = _load(args)
    n_volumes = len(reference)
    u = phase_encoding_vector(pe, reference.affine)
    mask_data = np.asanyarray(reference.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), reference.affine)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask_data)).T
    truth = parts._truth(perturbation, n_volumes)
    truth_mm = _truth_check(perturbation.distortion, truth["motion"], truth["eddy"], points)
    if truth_mm > parts.TRUTH_TOLERANCE:
        raise RuntimeError("the matrices miss the sidecar's field")
    cascade = np.load(_path(args, "cascade"))[f"cascade/{CASCADE[-1]}"]
    offset = _common(cascade, truth["motion"])
    order = _order(n_volumes)
    context = {
        "u": u,
        "mask": mask,
        "points": points,
        "shells": np.round(reference.gradients[:, -1], -2),
        "rank": {index: position for position, index in enumerate(order)},
        "directions": _directions(reference.gradients),
        "which": {},
        "all": list(range(n_volumes)),
        "truth": truth,
        "splits": [split_phase_encoding(whole, u)[0] for whole in truth["whole"]],
        "offset": offset,
    }

    stored = {arm: _read(args, arm, cascade) for arm in ("pass-e12-u", "pass-i9")}
    stored |= {arm: _read(args, arm, truth["motion"]) for arm in ("oracle-e12", "oracle-i9")}
    stored |= {
        arm: dict(np.load(_path(args, arm)))
        for arm in [*(arm for arms in GROUPS.values() for arm in arms), INSTANT]
        if _path(args, arm).exists()
    }
    data = load()
    for arm, source in DERIVED.items():
        if source in stored:
            values = dict(stored[source])
            volumes = [int(i) for i in values.get("volumes", context["all"])]
            values["estimates"] = rounds._removed(values["estimates"], data, pe, volumes)
            stored[arm] = values
    del data

    rows, arrays, walls = [], {}, {}
    for arm, values in stored.items():
        source = DERIVED.get(arm, arm)
        arm_rows = parts._rows(NAME, PROXY.get(source, source), values, context)
        for row in arm_rows:
            row["arm"] = arm
        rows += arm_rows
        arrays[arm] = values["estimates"]
        walls[arm] = float(values["wall"])
        median = np.median([row["whole_frame_mean"] for row in arm_rows])
        print(f"{arm}: median error {median:.3f} mm", flush=True)

    checks: dict[str, Any] = {
        "PhaseEncoding": pe,
        "PhaseEncodingDirection": u.tolist(),
        "TruthMm": truth_mm,
        "Failed": {arm: int(np.sum(values["failed"])) for arm, values in stored.items()},
    }
    if "hm-i9-hm" in stored:
        volumes = [int(i) for i in stored["hm-i9-hm"]["volumes"]]
        ours = stored["hm-i9-hm"]["estimates"][volumes]
        theirs = stored["pass-i9"]["estimates"][volumes]
        gaps = [float(_distances(mask, a, b).max()) for a, b in zip(ours, theirs, strict=True)]
        checks["Test3/pass-i9"] = {
            "identical": bool(np.array_equal(ours, theirs)),
            "max_mm": max(gaps),
        }

    table, _ = write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copies": {NAME: "head motion and eddy currents"},
            "Copy": NAME,
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            "MeanRadiusMm": float(
                np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=0).mean()
            ),
            "BrainCentroidMm": points.mean(axis=1).tolist(),
            "Noise": NOISE,
            "Seed": SEED,
            "Model": MODEL,
            "Samples": scaled,
            "Arms": list(stored),
            "ReadFromTest3": list(READ),
            "Derived": DERIVED,
            "Volumes": args.volumes,
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
