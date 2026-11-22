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
"""The decision and tables of Test 4 on #735: the eddy-current pass on copies of known parts.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_pass_parts.py`` wrote.
Comparisons are paired by volume (:func:`~experiments.paired_comparison`, candidate minus
reference, stratified by shell, 10,000 resamples, seed 0), on the confirmation volumes.
A volume pairs with itself across copies: every copy has the same noise in it. The
margins are 0.01 voxel: on the error, on the field and across phase encoding (mm); on the
rotation, the angle that moves the brain's mean radius by 0.01 voxel; and on the volume
change, Test 2's. Within a question, each interval is widened for the question's number
of comparisons (Bonferroni). The rules (``hm``: the head-motion step):

Q1, invented head motion
    ``hm-i9-hm`` on ``e2``, against itself on ``unmoved`` and against ``hm-e12-hm`` on
    ``e2`` (whose head motion is the head-motion step's): "better" or "not worse" on
    ``motion_rotation_deg`` and on ``motion_across_mean``. If it holds, the 9 parameters
    keep head motion and the field apart.
Q2, the field only
    ``oracle-i3`` against ``oracle-i9`` on ``e2``, and ``hm-i3-hm`` against ``hm-i9-hm``
    on every copy: "better" or "not worse" on the error. If it holds, the field-only pass
    is the candidate; otherwise the 9-parameter pass is.
Q3, invented field
    ``hm-i9-hm`` and ``hm-i3-hm`` on ``m2s0``, against themselves on ``unmoved``: "better"
    or "not worse" on ``field_mean``.
Q5, where the rest comes from (the candidate, on ``m2s0e2``)
    The start: ``hm-<c>-hm`` against ``hm-<c>-truth``. The targets: ``hm-<c>-truth``
    against ``truth-<c>-truth``. Each counts if "worse".

Reported beside: Q4, the floor (each registration on ``unmoved``); Q6, today's pass
(``pass-e12-u``) against ``hm-e12-hm`` on each copy; Q5 on every copy and registration;
the head-motion step's errors; each arm's medians. Writes
``<out>/eddy-parts-<protocol>-decision.json`` and its tables::

    uv run tools/registration/eddy_pass_parts_figures.py --results <results> --out <out>
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments import paired_comparison

MARGIN_VOXELS = 0.01
"""The margins, in voxels."""
PRIMARY = "whole_frame_mean"
"""The error: the whole mapping, in the frame of its targets (mean, mm)."""
N_RESAMPLES = 10_000
"""Resamples of the intervals."""
SUBSET = "confirm"
"""The volumes of the decision."""
COPIES = ("unmoved", "m2s0", "e2", "m2s0e2")
"""The copies, as the run names them."""
TEXT = ("copy", "arm", "targets", "registration", "start", "subset", "iterations", "in_frame")
"""The columns that are not numbers."""


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a results table, with numbers as numbers."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key, value in row.items():
            if key in TEXT:
                continue
            row[key] = value == "True" if value in ("True", "False") else float(value)
    return rows


def _select(rows: list[dict[str, Any]], copy: str, arm: str, subset: str | None) -> dict:
    """An arm's rows on a copy, by volume."""
    return {
        int(row["index"]): row
        for row in rows
        if row["copy"] == copy
        and row["arm"] == arm
        and (subset is None or row["subset"] == subset)
    }


def compare(
    rows: list[dict[str, Any]],
    candidate: tuple[str, str],
    reference: tuple[str, str],
    key: str,
    margin: float,
    *,
    tests: int = 1,
    subset: str | None = SUBSET,
) -> dict[str, Any]:
    """``(copy, arm)`` minus ``(copy, arm)`` on ``key``, paired by volume, and by shell."""
    first, second = _select(rows, *candidate, subset), _select(rows, *reference, subset)
    common = sorted(set(first) & set(second))
    a = np.array([first[i][key] for i in common])
    b = np.array([second[i][key] for i in common])
    shells = np.array([first[i]["shell"] for i in common])
    confidence = 1 - 0.05 / tests
    result = paired_comparison(
        a, b, margin, strata=shells, n_boot=N_RESAMPLES, seed=0, confidence=confidence
    )
    result["shells"] = {
        str(int(s)): paired_comparison(
            a[shells == s], b[shells == s], margin, n_boot=N_RESAMPLES, seed=0
        )
        for s in np.unique(shells)
    }
    return {
        "candidate": "/".join(candidate),
        "reference": "/".join(reference),
        "key": key,
        "confidence": confidence,
    } | result


def _ok(comparison: dict[str, Any]) -> bool:
    return comparison["verdict"] in ("better", "not worse")


def _margins(meta: dict[str, Any]) -> dict[str, float]:
    """The margin of each score (see the module's docstring)."""
    voxel = MARGIN_VOXELS * float(np.mean(meta["VoxelSize"]))
    return {
        PRIMARY: voxel,
        "field_mean": voxel,
        "motion_across_mean": voxel,
        "motion_rotation_deg": float(np.degrees(voxel / meta["MeanRadiusMm"])),
        "volume_change": voxel / meta["MeanRadiusMm"],
    }


def invented_motion(rows: list[dict[str, Any]], margins: dict[str, float]) -> dict[str, Any]:
    """Q1: does the 9-parameter pass report head motion on the eddy-only copy?"""
    pairs = {
        f"{key}/{against}": compare(
            rows, ("e2", "hm-i9-hm"), reference, key, margins[key], tests=4
        )
        for key in ("motion_rotation_deg", "motion_across_mean")
        for against, reference in (
            ("unmoved", ("unmoved", "hm-i9-hm")),
            ("head-motion step", ("e2", "hm-e12-hm")),
        )
    }
    return {"comparisons": pairs, "holds": all(_ok(c) for c in pairs.values())}


def field_only(rows: list[dict[str, Any]], margins: dict[str, float]) -> dict[str, Any]:
    """Q2: does the field-only pass fit the eddy currents, and keep up where there is motion?"""
    margin = margins[PRIMARY]
    pairs = {
        "oracle/e2": compare(
            rows, ("e2", "oracle-i3"), ("e2", "oracle-i9"), PRIMARY, margin, tests=5
        )
    }
    pairs |= {
        f"pass/{copy}": compare(
            rows, (copy, "hm-i3-hm"), (copy, "hm-i9-hm"), PRIMARY, margin, tests=5
        )
        for copy in COPIES
    }
    holds = all(_ok(c) for c in pairs.values())
    return {"comparisons": pairs, "holds": holds, "candidate": "i3" if holds else "i9"}


def invented_field(rows: list[dict[str, Any]], margins: dict[str, float]) -> dict[str, Any]:
    """Q3: does a pass report a field on the copy with head motion only?"""
    pairs = {
        registration: compare(
            rows,
            ("m2s0", f"hm-{registration}-hm"),
            ("unmoved", f"hm-{registration}-hm"),
            "field_mean",
            margins["field_mean"],
            tests=2,
        )
        for registration in ("i9", "i3")
    }
    return {"comparisons": pairs, "holds": all(_ok(c) for c in pairs.values())}


def sources(
    rows: list[dict[str, Any]], margin: float, copy: str, registration: str, tests: int
) -> dict[str, Any]:
    """Q5: the start's effect and the targets' effect, on a copy, for a registration."""
    start = compare(
        rows,
        (copy, f"hm-{registration}-hm"),
        (copy, f"hm-{registration}-truth"),
        PRIMARY,
        margin,
        tests=tests,
    )
    targets = compare(
        rows,
        (copy, f"hm-{registration}-truth"),
        (copy, f"truth-{registration}-truth"),
        PRIMARY,
        margin,
        tests=tests,
    )
    return {
        "start": start,
        "targets": targets,
        "start_counts": start["verdict"] == "worse",
        "targets_counts": targets["verdict"] == "worse",
    }


def _next(effects: dict[str, Any]) -> str:
    """What Q5's outcome points to (the module's docstring of the run, and #735)."""
    if effects["start_counts"] and effects["targets_counts"]:
        return "both: map the cost and try repeated delayed passes"
    if effects["start_counts"]:
        return "the start: map the cost around the truth and the estimates"
    if effects["targets_counts"]:
        return "the targets: repeated delayed passes, the ceiling as their best"
    return "neither: the floor of the model and the noise"


def summary(rows: list[dict[str, Any]], subset: str | None = SUBSET) -> dict[str, Any]:
    """Each copy's and arm's medians, overall and by shell."""
    keys = (
        PRIMARY,
        "motion_rotation_deg",
        "motion_across_mean",
        "motion_along_mean",
        "field_mean",
        "volume_change",
        "strain_pe",
        "strain_across",
        "seconds",
    )
    result: dict[str, Any] = {}
    for copy in COPIES:
        for arm in dict.fromkeys(row["arm"] for row in rows if row["copy"] == copy):
            chosen = list(_select(rows, copy, arm, subset).values())
            shells = np.array([row["shell"] for row in chosen])
            errors = np.array([row[PRIMARY] for row in chosen])
            values = {key: float(np.median([row[key] for row in chosen])) for key in keys}
            for shell in np.unique(shells):
                values[f"{PRIMARY}/{int(shell)}"] = float(np.median(errors[shells == shell]))
            values["failed"] = int(sum(row["failed"] for row in chosen))
            result[f"{copy}/{arm}"] = values
    return result


def cascade_summary(rows: list[dict[str, Any]], subset: str | None = SUBSET) -> dict[str, Any]:
    """The head-motion step's medians on each copy, in the targets' frame."""
    keys = ("motion_mean", "motion_rotation_deg", "motion_across_mean", "motion_along_mean")
    return {
        copy: {
            key: float(
                np.median(
                    [
                        row[key]
                        for row in rows
                        if row["copy"] == copy
                        and row["in_frame"] == "yes"
                        and (subset is None or row["subset"] == subset)
                    ]
                )
            )
            for key in keys
        }
        for copy in COPIES
        if any(row["copy"] == copy for row in rows)
    }


def decide(
    rows: list[dict[str, Any]], cascade: list[dict[str, Any]], meta: dict[str, Any]
) -> dict[str, Any]:
    """Test 4's decision: Q1, Q2, Q3 and Q5, and what is reported beside."""
    margins = _margins(meta)
    margin = margins[PRIMARY]
    q2 = field_only(rows, margins)
    candidate = q2["candidate"]
    q5 = sources(rows, margin, "m2s0e2", candidate, tests=2)
    q5["next"] = _next(q5)
    reported = {
        "q5": {
            f"{copy}/{registration}": sources(rows, margin, copy, registration, tests=1)
            for copy in COPIES
            for registration in ("i9", "i3")
        },
        "q6": {
            copy: compare(rows, (copy, "pass-e12-u"), (copy, "hm-e12-hm"), PRIMARY, margin)
            for copy in COPIES
        },
        "better_targets_only": {
            f"{copy}/{registration}": compare(
                rows,
                (copy, f"truth-{registration}-hm"),
                (copy, f"hm-{registration}-hm"),
                PRIMARY,
                margin,
            )
            for copy in COPIES
            for registration in ("i9", "i3")
        },
        "against_today": {
            f"{copy}/{registration}": compare(
                rows, (copy, f"hm-{registration}-hm"), (copy, "pass-e12-u"), PRIMARY, margin
            )
            for copy in COPIES
            for registration in ("i9", "i3")
        },
    }
    return {
        "protocol": meta["Protocol"],
        "subset": SUBSET,
        "margins": margins,
        "q1": invented_motion(rows, margins),
        "q2": q2,
        "q3": invented_field(rows, margins),
        "q5": q5,
        "reported": reported,
        "summary": summary(rows),
        "cascade": cascade_summary(cascade),
        "checks": meta["Checks"],
    }


def _write(path: Path, header: list[str], table: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(table)


def _interval(c: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{c['median'] * scale:+.3f} [{c['median_low'] * scale:+.3f}, "
        f"{c['median_high'] * scale:+.3f}]"
    )


def comparisons(decision: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Every comparison of the rules, labelled, then those reported."""
    listed = []
    for question in ("q1", "q2", "q3"):
        for label, c in decision[question]["comparisons"].items():
            listed.append((question.upper(), label, c))
    q5 = decision["q5"]
    listed += [("Q5", "start", q5["start"]), ("Q5", "targets", q5["targets"])]
    for label, c in decision["reported"]["q6"].items():
        listed.append(("Q6 (reported)", label, c))
    return listed


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the comparisons, and the head-motion step's errors."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    shells = [key for key in next(iter(arms.values())) if key.startswith(f"{PRIMARY}/")]
    _write(
        out / f"eddy-parts-{protocol}-arms.csv",
        [
            "Copy",
            "Arm",
            "Error (mm)",
            *(f"b={key.split('/')[1]}" for key in shells),
            "Rotation (deg)",
            "Across PE (mm)",
            "Along PE (mm)",
            "Field (mm)",
            "Volume change",
            "Strain along PE",
            "Time (s)",
        ],
        [
            [
                *name.split("/"),
                f"{values[PRIMARY]:.3f}",
                *(f"{values[key]:.3f}" for key in shells),
                f"{values['motion_rotation_deg']:.3f}",
                f"{values['motion_across_mean']:.3f}",
                f"{values['motion_along_mean']:.3f}",
                f"{values['field_mean']:.3f}",
                f"{100 * values['volume_change'] + 0.0:+.3f}%",
                f"{100 * values['strain_pe'] + 0.0:+.3f}%",
                "" if np.isnan(values["seconds"]) else f"{values['seconds']:.2f}",
            ]
            for name, values in arms.items()
        ],
    )
    listed = comparisons(decision)
    shells = list(listed[0][2]["shells"])
    _write(
        out / f"eddy-parts-{protocol}-comparisons.csv",
        ["Question", "Comparison", "Score", "Median difference [interval]", "Verdict"]
        + [f"b={s}" for s in shells],
        [
            [
                question,
                f"{c['candidate']} against {c['reference']} ({label})",
                c["key"],
                _interval(c),
                c["verdict"],
                *(
                    f"{c['shells'][s]['median']:+.3f} ({c['shells'][s]['verdict']})"
                    for s in shells
                ),
            ]
            for question, label, c in listed
        ],
    )
    _write(
        out / f"eddy-parts-{protocol}-cascade.csv",
        ["Copy", "Head motion (mm)", "Rotation (deg)", "Across PE (mm)", "Along PE (mm)"],
        [
            [copy, *(f"{values[key]:.3f}" for key in values)]
            for copy, values in decision["cascade"].items()
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the run's folder")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="where the tables go")
    args = parser.parse_args(argv)
    rows = read(args.results / f"{args.protocol}.tsv")
    cascade = read(args.results / f"{args.protocol}-cascade.tsv")
    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    decision = decide(rows, cascade, meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"eddy-parts-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    print(
        f"Q1 {decision['q1']['holds']}, Q2 {decision['q2']['holds']} "
        f"(candidate {decision['q2']['candidate']}), Q3 {decision['q3']['holds']}, "
        f"Q5 {decision['q5']['next']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
