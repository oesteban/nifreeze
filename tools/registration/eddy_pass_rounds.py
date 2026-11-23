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
"""Do repeated delayed passes bring the eddy-current pass to its truth-trained targets?
(#735, Test 5)

Test 4's four noisy copies of the phantom (neither head motion nor eddy currents, head
motion only, eddy currents only, both). On each, the 9-parameter pass (ITK, the ``itk``
section of :data:`~nifreeze.registration.base.AFFINE`: head motion, then a field along
phase encoding) runs in rounds, of two kinds:

``round1-i9``
    Test 4's delayed pass, ``hm-i9-hm``, read from its results: targets predicted (gp-loo,
    each volume left out) from the copy realigned by the head-motion step's estimate, and
    each volume registered from it.
``round<k>-i9``
    Targets predicted from the copy realigned by round ``k - 1``'s estimates, split into
    head motion and field as the pass stores them (the head motion turns the b-vectors),
    and each volume registered from its estimate of round ``k - 1``. Every volume is
    registered against the same targets, in parallel: the delayed update, repeated.
``removed<k>-i9``
    The same, with the part of the field that does not follow the diffusion gradient
    removed from each round's estimates before they are stored and the next targets are
    trained (:func:`~nifreeze.registration.utils.remove_shared_field`, per shell, about
    the brain's centroid; #738). ``removed1-i9`` is ``round1-i9`` with it removed.
``pass-i9-u``
    The :class:`~nifreeze.estimator.Estimator` with the 9 parameters and the instant
    update, from the head-motion step's estimate (Test 3 ran it on ``m2s0e2`` only).

Read from Test 4 as well: ``truth-i9-truth``, targets from the copy realigned by the
truth, registered from the truth (the ceiling of better training data), and ``pass-e12-u``,
today's pass (elastix, 12 parameters, the instant update). ``truth-i9-truth-removed`` and
``pass-i9-u-removed`` are those arms' estimates with the shared field removed.

Round ``k``'s targets are in the frame that all of round ``k - 1``'s estimates share
against the truth's whole mapping ``E M`` (:func:`eddy_targets._common`); round 1's, in the
frame of the head-motion step's estimates against ``M``, as in Test 4. Each row has Test
4's scores (:func:`eddy_pass_parts._rows`) in that frame.

Checks, on ``m2s0e2``: ``pass-i9-u`` against Test 3's, and round 2 as the
:class:`~nifreeze.estimator.Estimator` runs it (``round2-i9-estimator``: the delayed
update, its model built on the copy realigned by round 1's stored estimates) against
``round2-i9``.

Writes ``<out>/<protocol>.{tsv,json,npz}``, and each step's results to ``<out>/arms/`` (a
step already written is not run again). With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_pass_rounds.py --dataset <dataset> \\
        --test4 <Test 4's results>/arms --test3 <Test 3's results>/arms --out <out> \\
        --jobs 6 --threads 6
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
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.estimator import Estimator
from nifreeze.model.base import ModelFactory
from nifreeze.registration.base import AFFINE
from nifreeze.registration.utils import (
    phase_encoding_vector,
    remove_shared_field,
    split_phase_encoding,
)

ROUNDS = 4
"""The rounds of the delayed pass, the first being Test 4's."""
KINDS = ("removed", "round")
"""The kinds of rounds, in the order they run."""
READ = {"round1-i9": "hm-i9-hm", "truth-i9-truth": "truth-i9-truth", "pass-e12-u": "pass-e12-u"}
"""The arms read from Test 4, and their names there."""
DERIVED = {
    "removed1-i9": "round1-i9",
    "truth-i9-truth-removed": "truth-i9-truth",
    "pass-i9-u-removed": "pass-i9-u",
}
"""The arms that are another arm's estimates with the shared field removed."""
CHECK = "round2-i9-estimator"
"""Round 2 through the Estimator, on ``m2s0e2`` (the module's docstring)."""


def _round(arm: str) -> tuple[str, int] | None:
    """A round's kind and number, from its arm's name."""
    for kind in KINDS:
        number = arm[len(kind) :].split("-")[0]
        if arm.startswith(kind) and number.isdigit():
            return kind, int(number)
    return None


def _path(args: argparse.Namespace, name: str, arm: str) -> Path:
    """Where an arm's results are: Test 4's for the arms read from it."""
    if arm in READ:
        return args.test4 / f"{args.protocol}-{name}-{READ[arm]}.npz"
    if arm == "cascade":
        return args.test4 / f"{args.protocol}-{name}-cascade.npz"
    return args.out / "arms" / f"{args.protocol}-{name}-{arm}.npz"


def _split(estimates: np.ndarray, pe: str, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The head motion and the field of each estimate, as the pass stores them."""
    backend = parts._backend("i9", pe)
    split = [backend.split(matrix, affine) for matrix in estimates]
    return np.stack([motion for motion, _ in split]), np.stack([field for _, field in split])


def _centre(data: Any) -> np.ndarray:
    """The brain's centroid (world, mm)."""
    mask = np.asanyarray(data.brainmask, dtype=bool)
    return nb.affines.apply_affine(data.affine, np.argwhere(mask)).mean(axis=0)


def _removed(estimates: np.ndarray, data: Any, pe: str, volumes: list[int]) -> np.ndarray:
    """Estimates of ``volumes`` with the shared field removed (others as they are)."""
    motion, field = _split(estimates[volumes], pe, data.affine)
    u = phase_encoding_vector(pe, data.affine)
    motion, field = remove_shared_field(
        motion, field, np.asarray(data.gradients)[volumes], u, _centre(data)
    )
    removed = np.array(estimates, dtype=float)
    removed[volumes] = field @ motion
    return removed


def _previous(args: argparse.Namespace, name: str, kind: str, k: int, data: Any) -> np.ndarray:
    """The estimates of every volume in round ``k - 1`` of a kind."""
    n_volumes = len(data)
    if kind == "removed" and k == 2:
        stored = np.load(_path(args, name, "round1-i9"))
    else:
        stored = np.load(_path(args, name, f"{kind}{k - 1}-i9"))
    if sorted(int(i) for i in stored["volumes"]) != list(range(n_volumes)):
        raise RuntimeError(f"{name}: round {k - 1} did not register every volume")
    estimates = np.asarray(stored["estimates"], dtype=float)
    if kind == "removed" and k == 2:
        return _removed(estimates, data, data.phase_encoding, list(range(n_volumes)))
    return estimates


def run_round(args: argparse.Namespace, name: str, kind: str, k: str) -> None:
    """Round ``k`` of a kind on a copy: targets from round ``k - 1``, every volume."""
    began = perf_counter()
    reference, _, load, pe = parts._load(args, name)
    n_volumes = len(reference)
    volumes = parts._volumes(args, n_volumes)
    copy = load()
    copy.phase_encoding = pe
    previous = _previous(args, name, kind, int(k), copy)
    training = load()
    training.motion, training.eddy = _split(previous, pe, training.affine)
    targets = held_out_predictions(MODEL, training.realigned(), volumes)
    del training
    targets_seconds = perf_counter() - began
    print(f"{name} {kind} {k}: targets in {targets_seconds:.0f} s", flush=True)

    arm_began = perf_counter()
    parts._SHARED.update(copy=copy, targets=targets, starts=previous, registration="i9")
    estimates = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    seconds = np.full(n_volumes, np.nan)
    failed = np.zeros(n_volumes, dtype=bool)
    iterations = np.zeros((n_volumes, 2), dtype=int)
    with ProcessPoolExecutor(
        args.jobs,
        mp_context=get_context("fork"),
        initializer=parts._warm,
        initargs=(volumes[0],),
    ) as pool:
        for future in as_completed([pool.submit(parts._register, i) for i in volumes]):
            index, estimate, took, broke, stops = future.result()
            estimates[index], seconds[index], failed[index] = estimate, took, broke
            iterations[index, : len(stops)] = stops
    registered = estimates.copy()
    if kind == "removed":
        estimates = _removed(estimates, copy, pe, volumes)
    np.savez_compressed(
        _path(args, name, f"{kind}{k}-i9"),
        estimates=estimates,
        registered=registered,
        starts=previous,
        volumes=np.asarray(volumes),
        seconds=seconds,
        failed=failed,
        iterations=iterations,
        targets_seconds=targets_seconds,
        wall=perf_counter() - arm_began,
    )
    print(f"{name} {kind} {k}: registered in {perf_counter() - arm_began:.0f} s", flush=True)


def _estimator_arrays(data: Any, starts: np.ndarray, began: float) -> dict[str, Any]:
    n_volumes = len(data)
    return {
        "estimates": np.stack([np.asarray(data.get_transform(i)) for i in range(n_volumes)]),
        "motion": np.asarray(data.motion),
        "eddy": np.asarray(data.eddy),
        "starts": starts,
        "seconds": np.full(n_volumes, np.nan),
        "failed": np.zeros(n_volumes, dtype=bool),
        "iterations": np.zeros((n_volumes, 0), dtype=int),
        "wall": perf_counter() - began,
    }


def run_instant(args: argparse.Namespace, name: str) -> None:
    """The 9-parameter pass with the instant update, from the head-motion step."""
    began = perf_counter()
    _, _, load, pe = parts._load(args, name)
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    data = load()
    data.phase_encoding = pe
    data.motion = cascade.copy()
    Estimator(MODEL, registration="itk", estimates="eddy", align_config=AFFINE).run(
        data, omp_nthreads=args.threads, n_jobs=args.threads, seed=SEED, num_threads=1
    )
    np.savez_compressed(_path(args, name, "pass-i9-u"), **_estimator_arrays(data, cascade, began))
    print(f"{name} pass-i9-u: {perf_counter() - began:.0f} s", flush=True)


def run_check(args: argparse.Namespace, name: str) -> None:
    """Round 2 through the Estimator: the delayed update from round 1's stored estimates."""
    began = perf_counter()
    _, _, load, pe = parts._load(args, name)
    data = load()
    data.phase_encoding = pe
    previous = _previous(args, name, "round", 2, data)
    data.motion, data.eddy = _split(previous, pe, data.affine)
    # The delayed update: a model built on the realigned copy keeps reading that view
    model = ModelFactory.init(model=MODEL, dataset=data.realigned())
    Estimator(model, registration="itk", estimates="eddy", align_config=AFFINE).run(
        data, omp_nthreads=args.threads, n_jobs=args.threads, seed=SEED, num_threads=1
    )
    np.savez_compressed(_path(args, name, CHECK), **_estimator_arrays(data, previous, began))
    print(f"{name} {CHECK}: {perf_counter() - began:.0f} s", flush=True)


def _arms(args: argparse.Namespace) -> list[str]:
    """The arms of a copy that are run or read, in the order they run."""
    rounds = [f"{kind}{k}-i9" for kind in args.kinds for k in range(2, args.rounds + 1)] + [
        "round1-i9"
    ]
    extra = (["pass-i9-u"] if args.instant else []) + ([CHECK] if args.check else [])
    return rounds + extra + ["truth-i9-truth", "pass-e12-u"]


def _run_steps(args: argparse.Namespace) -> None:
    """Run each copy's steps not yet written, each in a process of its own: the rounds of
    every copy (the rounds that remove the shared field first), then the instant passes,
    then the check."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out), "--volumes", args.volumes]
    command += ["--test4", str(args.test4), "--jobs", str(args.jobs)]
    command += ["--threads", str(args.threads)]
    steps = [
        (name, f"{kind}{k}-i9", f"round:{name}:{kind}:{k}")
        for kind in args.kinds
        for name in args.copies
        for k in range(2, args.rounds + 1)
    ]
    if args.instant:
        steps += [(name, "pass-i9-u", f"instant:{name}") for name in args.copies]
    if args.check and "m2s0e2" in args.copies:
        steps.append(("m2s0e2", CHECK, "check:m2s0e2"))
    for name, arm, step in steps:
        if not _path(args, name, arm).exists():
            subprocess.run([*command, "--run", step], check=True)


def _derive(stored: dict[str, Any], data: Any, pe: str) -> None:
    """Add the arms that remove the shared field from another arm's estimates."""
    for arm, source in DERIVED.items():
        if source not in stored:
            continue
        values = dict(stored[source])
        volumes = (
            [int(i) for i in values["volumes"]] if "volumes" in values else list(range(len(data)))
        )
        values["estimates"] = _removed(np.asarray(values["estimates"]), data, pe, volumes)
        stored[arm] = values


def _frames(cascade: np.ndarray, stored: dict[str, Any], truth: dict[str, np.ndarray]) -> dict:
    """The frame of each arm's targets (the module's docstring)."""
    frames = {"hm": _common(cascade, truth["motion"]), "truth": np.eye(4)}
    for arm, values in stored.items():
        found = _round(arm)
        if found is not None and found[1] > 1 and arm != CHECK:
            frames[arm] = _common(np.asarray(values["starts"], dtype=float), truth["whole"])
    frames["round1-i9"] = frames["removed1-i9"] = frames["hm"]
    if "round2-i9" in frames:
        frames[CHECK] = frames["round2-i9"]
    return frames


def _arm_rows(name: str, arm: str, values: Any, context: dict, frames: dict) -> list[dict]:
    """An arm's rows, scored as Test 4 scores them, in its targets' frame."""
    found = _round(arm)
    if found is None:
        # Test 4's arms, and those with the shared field removed from them
        proxy = DERIVED.get(arm, arm)
        proxy = READ.get(proxy, proxy)
        rows = parts._rows(name, proxy, values, context | {"offset": frames["hm"]})
        targets = None
    else:
        # A round is scored as Test 4's delayed pass, in its own frame
        rows = parts._rows(name, "hm-i9-hm", values, context | {"offset": frames[arm]})
        targets = f"{found[0]}{found[1]}"
    for row in rows:
        row["arm"] = arm
        row["targets"] = targets or row["targets"]
    return rows


def _frame_drift(frames: dict[str, np.ndarray]) -> dict[str, Any]:
    """Each round's frame against round 1's: its rotation (degrees) and translation (mm)."""
    drift = {}
    for arm, frame in frames.items():
        if _round(arm) is None or arm in ("round1-i9", "removed1-i9", CHECK):
            continue
        change = frame @ np.linalg.inv(frames["hm"])
        drift[arm] = {
            "RotationDeg": float(np.degrees(Rotation.from_matrix(change[:3, :3]).magnitude())),
            "TranslationMm": change[:3, 3].tolist(),
        }
    return drift


def _checks(name: str, stored: dict, args: argparse.Namespace, mask: Any) -> dict[str, Any]:
    """On ``m2s0e2``, the instant pass against Test 3's, and round 2 against the Estimator's."""
    if name != "m2s0e2":
        return {}
    checks: dict[str, Any] = {}
    pairs = []
    if "pass-i9-u" in stored and args.test3 is not None:
        recorded = args.test3 / f"{args.protocol}-pass-i9-u.npz"
        if recorded.exists():
            pairs.append(("Test3/pass-i9-u", stored["pass-i9-u"], np.load(recorded)))
    if CHECK in stored and "round2-i9" in stored:
        pairs.append((f"{CHECK}/round2-i9", stored[CHECK], stored["round2-i9"]))
    for label, ours, theirs in pairs:
        a, b = np.asarray(ours["estimates"]), np.asarray(theirs["estimates"])
        gaps = [float(_distances(mask, x, y).max()) for x, y in zip(a, b, strict=True)]
        checks[label] = {"identical": bool(np.array_equal(a, b)), "max_mm": max(gaps)}
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(EDDY_MM))
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--test4", type=Path, required=True, help="Test 4's arms/ folder")
    parser.add_argument("--test3", type=Path, help="Test 3's arms/ folder: its pass-i9-u")
    parser.add_argument("--copies", nargs="+", default=list(COPIES), choices=list(COPIES))
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="rounds, Test 4's included")
    parser.add_argument("--no-instant", dest="instant", action="store_false")
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--volumes", default="all", choices=("all", "dev", "confirm"))
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the models and passes")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        step, name, *rest = args.run.split(":")
        if step == "round":
            run_round(args, name, *rest)
        elif step == "instant":
            run_instant(args, name)
        else:
            run_check(args, name)
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

    rows, arrays, walls, drifts = [], {}, {}, {}
    checks: dict[str, Any] = {"PhaseEncoding": pe, "PhaseEncodingDirection": u.tolist()}
    arms = _arms(args)
    for name in args.copies:
        perturbation, load = parts._choose(reference, cells, name)
        truth = parts._truth(perturbation, n_volumes)
        cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
        stored = {
            arm: dict(np.load(_path(args, name, arm)))
            for arm in arms
            if _path(args, name, arm).exists()
        }
        data = load()
        _derive(stored, data, pe)
        del data
        frames = _frames(cascade, stored, truth)
        drifts[name] = _frame_drift(frames)
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
        checks |= _checks(name, stored, args, mask)

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
            "Rounds": args.rounds,
            "Kinds": args.kinds,
            "Arms": arms + [arm for arm in DERIVED if arm not in arms],
            "ReadFromTest4": READ,
            "Derived": DERIVED,
            "Volumes": args.volumes,
            "Order": order,
            "FrameDrift": drifts,
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
