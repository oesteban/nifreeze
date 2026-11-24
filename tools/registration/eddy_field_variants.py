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
"""Does a matrix C for each shell, or shifts derived from one zero point, lower the error
of the fit, then the 9-parameter pass? (#735, Test 8)

Test 7's chain on Test 4's four noisy copies of the phantom: the fit
(:class:`~nifreeze.registration.eddy_field.EddyFieldFit`, the ``itk`` section of
:data:`~nifreeze.registration.base.AFFINE`) with Test 4's targets, from the head-motion
step's estimate ``m̂``; then targets retrained on the copy realigned by the fit, and the
9-parameter pass from the fit's estimates; then the shared field removed
(:func:`~nifreeze.registration.utils.remove_shared_field`, per shell, about the brain's
centroid). The fit comes in four variants (:data:`VARIANTS`):

``cp-free``
    One ``C`` and the exponent ``p``, each volume's shift estimated: Test 7's, read from
    its results.
``cp-derived``
    One ``C`` and ``p``, the shifts derived from one zero point of the fields.
``shells-free``
    A ``C`` for each shell (``segments`` halfway between the shells), the shifts
    estimated.
``shells-derived``
    A ``C`` for each shell, the shifts derived from one zero point.

Each variant gives the arms ``fit-<variant>`` (the fit), ``chain-<variant>`` (the pass
after it) and ``chain-<variant>-removed`` (with the shared field removed). Read from Test
4: ``hm-i9-hm`` (the delayed pass), scored with the shared field removed as
``hm-i9-hm-removed``, and ``pass-e12-u`` (today's pass).

Every arm is scored as Test 7 scores it: the fits and the arms with Test 4's targets in
the frame of the head-motion step; a chain in the frame that all of its fit's estimates
share against the truth's whole mapping.

The options this test compares were dropped from #743 after it, so this script runs
only as it ran, at oesteban/nifreeze@4764024ce.

Writes ``<out>/<protocol>.{tsv,json,npz}``, and each step's results to ``<out>/arms/`` (a
step already written is not run again). With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_field_variants.py --dataset <dataset> \\
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
from eddy_field_start import _pass, _targets
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
from nifreeze.registration.eddy_field import EddyFieldFit
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

VARIANTS = {
    "cp-free": {"shells": False, "shifts": "free"},
    "cp-derived": {"shells": False, "shifts": "derived"},
    "shells-free": {"shells": True, "shifts": "free"},
    "shells-derived": {"shells": True, "shifts": "derived"},
}
"""Each variant of the fit: a C for each shell or not, and how the shifts are found."""
FROM_TEST7 = {"fit-cp-free": "hm-field", "chain-cp-free": "field-i9-field"}
"""The arms of ``cp-free``, read from Test 7."""
READ = ("hm-i9-hm", "pass-e12-u")
"""The arms read from Test 4."""
REMOVED = {"hm-i9-hm-removed": "hm-i9-hm"} | {
    f"chain-{variant}-removed": f"chain-{variant}" for variant in VARIANTS
}
"""The arms that are another arm's estimates with the shared field removed."""


def _edges(bvals: np.ndarray) -> tuple[float, ...]:
    """Halfway between the shells (the b-values rounded to 100 s/mm²)."""
    shells = np.unique(np.round(np.asarray(bvals, dtype=float), -2))
    return tuple(float(edge) for edge in (shells[:-1] + shells[1:]) / 2)


def _path(args: argparse.Namespace, name: str, arm: str) -> Path:
    """Where an arm's results are: Test 4's or Test 7's for the arms read from them."""
    if arm in READ or arm == "cascade":
        return args.test4 / f"{args.protocol}-{name}-{arm}.npz"
    if arm in FROM_TEST7:
        return args.test7 / f"{args.protocol}-{name}-{FROM_TEST7[arm]}.npz"
    return args.out / "arms" / f"{args.protocol}-{name}-{arm}.npz"


def _fit() -> dict[str, Any]:
    """One variant's fit on every volume of the copy in ``parts._SHARED``, in a process
    of its own."""
    shared = parts._SHARED
    copy, targets, starts = shared["copy"], shared["targets"], shared["starts"]
    variant = VARIANTS[shared["variant"]]
    gradients = np.asarray(copy.gradients)
    began = perf_counter()
    options: dict[str, Any] = {"shifts": variant["shifts"]}
    if variant["shells"]:
        options["segments"] = _edges(gradients[:, 3])
    fit = EddyFieldFit(
        AFFINE,
        phase_encoding=copy.phase_encoding,
        seed=SEED,
        num_threads=shared["threads"],
        **options,
    )
    for index in sorted(targets):
        fit.add(index, targets[index], copy[index][0], copy.affine, copy.brainmask, starts[index])
    field = fit.fit(gradients)
    n_volumes = len(copy)
    motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    eddy = motion.copy()
    for index, (head_motion, matrix) in fit.mappings().items():
        motion[index], eddy[index] = head_motion, matrix
    pieces = getattr(field, "parts", (field,))
    return {
        "estimates": eddy @ motion,
        "motion": motion,
        "eddy": eddy,
        "matrices": np.stack([part.matrix for part in pieces]),
        "b_refs": np.array([part.b_ref for part in pieces]),
        "exponent": pieces[0].exponent,
        "edges": np.asarray(getattr(field, "edges", ()), dtype=float),
        "zero_point": np.full(3, np.nan) if fit.zero_point is None else fit.zero_point,
        "shifts": np.array([fit.shifts[i] for i in sorted(fit.shifts)]),
        "evaluations": np.asarray(fit.evaluations),
        "messages": np.asarray(fit.messages),
        "fit_seconds": perf_counter() - began,
    }


def run_fits(args: argparse.Namespace, name: str) -> None:
    """On Test 4's targets, each variant's fit from the head-motion step."""
    reference, _, load, pe = parts._load(args, name)
    n_volumes = len(reference)
    volumes = list(range(n_volumes))
    cascade = np.load(_path(args, name, "cascade"))[f"cascade/{CASCADE[-1]}"]
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, cascade, None, volumes)
    print(f"{name} hm: targets in {targets_seconds:.0f} s", flush=True)
    parts._SHARED.update(copy=copy, targets=targets, starts=cascade, threads=args.threads)
    for variant in VARIANTS:
        arm = f"fit-{variant}"
        if _path(args, name, arm).exists():
            continue
        began = perf_counter()
        parts._SHARED.update(variant=variant)
        # A child process, so that no ITK threads exist in this one
        with ProcessPoolExecutor(1, mp_context=get_context("fork")) as pool:
            fitted = pool.submit(_fit).result()
        np.savez_compressed(
            _path(args, name, arm),
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
            f"{name} {arm}: {fitted['evaluations'].tolist()} evaluations,"
            f" {perf_counter() - began:.0f} s",
            flush=True,
        )


def run_chain(args: argparse.Namespace, name: str, variant: str) -> None:
    """The 9-parameter pass after a variant's fit: targets from the copy it realigned."""
    reference, _, load, pe = parts._load(args, name)
    n_volumes = len(reference)
    volumes = list(range(n_volumes))
    fitted = np.load(_path(args, name, f"fit-{variant}"))
    starts = np.asarray(fitted["estimates"], dtype=float)
    copy = load()
    copy.phase_encoding = pe
    targets, targets_seconds = _targets(load, fitted["motion"], fitted["eddy"], volumes)
    print(f"{name} {variant}: targets in {targets_seconds:.0f} s", flush=True)
    began = perf_counter()
    parts._SHARED.update(copy=copy, targets=targets, starts=starts, registration="i9")
    result = _pass(volumes, n_volumes, args.jobs)
    np.savez_compressed(
        _path(args, name, f"chain-{variant}"),
        **result,
        starts=starts,
        volumes=np.asarray(volumes),
        targets_seconds=targets_seconds,
        wall=perf_counter() - began,
    )
    print(f"{name} chain-{variant}: {perf_counter() - began:.0f} s", flush=True)


def _run_steps(args: argparse.Namespace) -> None:
    """Run each copy's steps not yet written, each in a process of its own."""
    command = [sys.executable, __file__, "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out)]
    command += ["--test4", str(args.test4), "--test7", str(args.test7)]
    command += ["--jobs", str(args.jobs), "--threads", str(args.threads)]
    for name in args.copies:
        if not all(_path(args, name, f"fit-{variant}").exists() for variant in VARIANTS):
            subprocess.run([*command, "--run", f"fits:{name}"], check=True)
    for name in args.copies:
        for variant in VARIANTS:
            if not _path(args, name, f"chain-{variant}").exists():
                subprocess.run([*command, "--run", f"chain:{name}:{variant}"], check=True)


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
    """A fit's model, its stages and its time."""
    return {
        "Matrices": np.asarray(values["matrices"]).tolist()
        if "matrices" in values
        else [np.asarray(values["matrix"]).tolist()],
        "BRefs": np.atleast_1d(values["b_refs"] if "b_refs" in values else values["b_ref"])
        .astype(float)
        .tolist(),
        "Exponent": float(values["exponent"]),
        "Edges": np.asarray(values.get("edges", []), dtype=float).tolist(),
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
        if step == "fits":
            run_fits(args, names[0])
        else:
            run_chain(args, *names)
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
    arms = [*READ, *(f"{kind}-{variant}" for variant in VARIANTS for kind in ("fit", "chain"))]

    rows, arrays, walls, fields = [], {}, {}, {}
    checks: dict[str, Any] = {"PhaseEncoding": pe, "PhaseEncodingDirection": u.tolist()}
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
            fields[f"{name}/{variant}"] = _field(fitted)
        context = base | {
            "truth": truth,
            "splits": [split_phase_encoding(whole, u)[0] for whole in truth["whole"]],
        }
        for arm, values in stored.items():
            chained = arm.startswith("chain-")
            frame = frames[arm.split("-", 1)[1].removesuffix("-removed")] if chained else None
            arm_rows = _arm_rows(
                name, arm, values, context, frames["hm"] if frame is None else frame
            )
            rows += arm_rows
            walls[f"{name}/{arm}"] = float(values["wall"])
            if "targets_seconds" in values:
                walls[f"{name}/{arm}/targets"] = float(values["targets_seconds"])
            arrays[f"{name}/{arm}"] = values["estimates"]
            median = np.median([row["whole_frame_mean"] for row in arm_rows])
            print(f"{name} {arm}: median error {median:.3f} mm", flush=True)
        # The rotations each fit keeps, and its head motion across phase encoding
        across = np.eye(3) - np.outer(u, u)
        for variant in VARIANTS:
            change = stored[f"fit-{variant}"]["motion"] @ np.linalg.inv(cascade) - np.eye(4)
            checks[f"{name}/{variant}/RotationKept"] = float(np.abs(change[:, :3, :3]).max())
            checks[f"{name}/{variant}/AcrossPE"] = float(
                np.abs(across @ change[:, :3, 3, None]).max()
            )

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
            "Edges": list(_edges(reference.gradients[:, -1])),
            "Noise": NOISE,
            "Seed": SEED,
            "Model": MODEL,
            "Variants": VARIANTS,
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
