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
"""The decision, tables and figures of Test 5 on #735: the delayed pass, repeated.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_pass_rounds.py`` wrote.
Comparisons are Test 4's (:func:`eddy_pass_parts_figures.compare`: paired by volume,
candidate minus reference, stratified by shell, on the confirmation volumes), with its
margins. Within a rule, each interval is widened for the rule's number of comparisons
(Bonferroni). The rules judge the rounds that remove the shared field (``x<k>``:
``removed<k>-i9``; ``r1``: ``round1-i9``, Test 4's delayed pass):

Rule 1, the removal
    ``x1`` against ``r1`` on every copy: "better" or "not worse" on the error.
Rule 2, the rounds help
    ``x4`` against ``x1`` on ``m2s0``, ``e2`` and ``m2s0e2``: "better" on the error.
Rule 3, b=1000
    ``x4`` against ``x1`` on every copy, on the b=1000 volumes: "better" or "not worse".
Rule 4, invented head motion
    ``x4`` on ``e2`` against ``x4`` on ``unmoved``: "better" or "not worse" on
    ``motion_rotation_deg`` and on ``motion_across_mean``.
Rule 5, the ceiling
    ``x4`` against ``truth-i9-truth-removed`` on every copy: "better" or "not worse".
Rule 6, the instant update
    ``x4`` against ``pass-i9-u-removed`` on every copy: "better" or "not worse".

Reported beside: each round against the one before, of both kinds, on every copy (95%),
and the rounds the pass needs (the last round that removes the shared field and is
"better" than the one before on some copy); the plain rounds' round 4 against round 1;
``x4``, ``pass-i9-u`` and ``r1`` against today's pass (``pass-e12-u``, Test 4) on every
copy; ``x4`` against ``pass-i9-u``; each arm's medians. Writes
``<out>/eddy-rounds-<protocol>-decision.json``, its tables, and two figures: each arm's
median error by round (``-rounds.png``) and the rules' comparisons on the error
(``-comparisons.png``)::

    uv run tools/registration/eddy_pass_rounds_figures.py --results <results> --out <out>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_pass_parts_figures import (
    COPIES,
    COPY_TITLES,
    PRIMARY,
    SUBSET,
    _interval,
    _interval_row,
    _margins,
    _ok,
    _write,
    compare,
    read,
    summary,
)

MOVED = ("m2s0", "e2", "m2s0e2")
"""The copies with something to correct."""
SHELL = 1000.0
"""Rule 3's shell."""
KINDS = {"round": "Plain rounds", "removed": "Shared field removed"}
"""The kinds of rounds, as the figures name them."""


def _arm(kind: str, k: int) -> str:
    return f"{kind}{k}-i9"


def _rule(comparisons: dict[str, dict[str, Any]], better: bool = False) -> dict[str, Any]:
    """A rule holds if it has comparisons and each has the verdict it needs."""
    holds = bool(comparisons) and all(
        c["verdict"] == "better" if better else _ok(c) for c in comparisons.values()
    )
    return {"holds": holds, "comparisons": comparisons}


def decide(
    rows: list[dict[str, Any]], meta: dict[str, Any], subset: str | None = SUBSET
) -> dict[str, Any]:
    """Test 5's decision: the rules, and what is reported beside."""
    margins = _margins(meta)
    rounds = int(meta["Rounds"])
    first, last = _arm("removed", 1), _arm("removed", rounds)
    copies = [copy for copy in COPIES if copy in meta["Copies"]]
    present = {(row["copy"], row["arm"]) for row in rows}
    kinds = [kind for kind in KINDS if any((c, _arm(kind, 2)) in present for c in copies)]
    shell_rows = [row for row in rows if row["shell"] == SHELL]

    def paired(table, candidate, reference, key=PRIMARY, tests=1):
        return compare(table, candidate, reference, key, margins[key], tests=tests, subset=subset)

    def each(reference: str, tests: int = len(COPIES), table=rows) -> dict[str, Any]:
        return {
            copy: paired(table, (copy, last), (copy, reference), tests=tests)
            for copy in copies
            if (copy, reference) in present
        }

    rule1 = {
        copy: paired(rows, (copy, first), (copy, _arm("round", 1)), tests=len(COPIES))
        for copy in copies
    }
    rule2 = {
        copy: paired(rows, (copy, last), (copy, first), tests=len(MOVED))
        for copy in MOVED
        if copy in copies
    }
    rule4 = {
        key: paired(rows, ("e2", last), ("unmoved", last), key, tests=2)
        for key in ("motion_rotation_deg", "motion_across_mean")
        if {"e2", "unmoved"} <= set(copies)
    }
    # Reported: each round against the one before, and against today's pass
    successive = {
        f"{copy}/{_arm(kind, k)}": paired(rows, (copy, _arm(kind, k)), (copy, _arm(kind, k - 1)))
        for kind in kinds
        for copy in copies
        for k in range(2, rounds + 1)
    }
    better = [
        k
        for k in range(2, rounds + 1)
        if any(
            successive[f"{copy}/{_arm('removed', k)}"]["verdict"] == "better" for copy in copies
        )
    ]
    needed = max(better, default=1)
    plain = {
        copy: paired(rows, (copy, _arm("round", rounds)), (copy, _arm("round", 1)))
        for copy in copies
        if (copy, _arm("round", rounds)) in present
    }
    today = {
        f"{copy}/{arm}": paired(rows, (copy, arm), (copy, "pass-e12-u"))
        for copy in copies
        for arm in (last, "pass-i9-u", _arm("round", 1))
        if (copy, arm) in present
    }
    return {
        "protocol": meta["Protocol"],
        "margins": margins,
        "rule1": _rule(rule1),
        "rule2": _rule(rule2, better=True),
        "rule3": _rule(each(first, table=shell_rows)),
        "rule4": _rule(rule4),
        "rule5": _rule(each("truth-i9-truth-removed")),
        "rule6": _rule(each("pass-i9-u-removed")),
        "reported": {
            "successive": successive,
            "rounds_needed": needed,
            "settled": needed < rounds,
            "plain_rounds": plain,
            "against_today": today,
            "against_instant": each("pass-i9-u", tests=1),
        },
        "rounds": rounds,
        "summary": summary(rows, subset),
        "frame_drift": meta["FrameDrift"],
        "checks": meta["Checks"],
    }


RULES = (
    ("rule1", "Rule 1: the shared field removed, against the delayed pass"),
    ("rule2", "Rule 2: round 4 against round 1"),
    ("rule3", "Rule 3: round 4 against round 1, b=1000"),
    ("rule4", "Rule 4: round 4, eddy-only copy against neither"),
    ("rule5", "Rule 5: round 4 against truth-trained targets"),
    ("rule6", "Rule 6: round 4 against the instant update"),
)
"""The rules, as the tables label them."""


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the rules' comparisons, and those reported."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    shells = [key for key in next(iter(arms.values())) if key.startswith(f"{PRIMARY}/")]
    _write(
        out / f"eddy-rounds-{protocol}-arms.csv",
        [
            "Copy",
            "Arm",
            "Error (mm)",
            *(f"b={key.split('/')[1]}" for key in shells),
            "Rotation (deg)",
            "Across PE (mm)",
            "Field (mm)",
            "Volume change",
            "Strain along PE",
            "Time (s)",
        ],
        [
            [
                *name.split("/"),
                f"{values[PRIMARY]:.3f}",
                *(f"{values.get(key, np.nan):.3f}" for key in shells),
                f"{values['motion_rotation_deg']:.3f}",
                f"{values['motion_across_mean']:.3f}",
                f"{values['field_mean']:.3f}",
                f"{100 * values['volume_change'] + 0.0:+.3f}%",
                f"{100 * values['strain_pe'] + 0.0:+.3f}%",
                "" if np.isnan(values["seconds"]) else f"{values['seconds']:.2f}",
            ]
            for name, values in arms.items()
        ],
    )
    listed = [
        (title, label, c)
        for rule, title in RULES
        for label, c in decision[rule]["comparisons"].items()
    ]
    names = sorted({s for _, _, c in listed for s in c["shells"]}, key=float)
    header = ["Rule", "Comparison", "Score", "Median difference [interval]", "Verdict"]
    header += [f"b={s}" for s in names]
    _write(out / f"eddy-rounds-{protocol}-comparisons.csv", header, _table(listed, names))
    reported = decision["reported"]
    listed = [
        (title, label, c)
        for group, title in (
            ("successive", "Against the round before"),
            ("plain_rounds", "Plain rounds: round 4 against round 1"),
            ("against_today", "Against today's pass"),
            ("against_instant", "Against the instant update, as it is"),
        )
        for label, c in reported[group].items()
    ]
    _write(out / f"eddy-rounds-{protocol}-reported.csv", header, _table(listed, names))


def _table(listed: list[tuple[str, str, dict[str, Any]]], names: list[str]) -> list[list[Any]]:
    """Each comparison's row; a shell it does not have is left blank."""
    return [
        [
            rule,
            f"{c['candidate']} against {c['reference']} ({label})",
            c["key"],
            _interval(c),
            c["verdict"],
            *(
                f"{c['shells'][s]['median']:+.3f} ({c['shells'][s]['verdict']})"
                if s in c["shells"]
                else ""
                for s in names
            ),
        ]
        for rule, label, c in listed
    ]


def rounds_figure(decision: dict[str, Any], out: Path, rounds: int) -> None:
    """Each copy's median error by round, then the instant update and the ceiling, for
    each kind of rounds."""
    from eddy_targets_figures import INK, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    arms = decision["summary"]
    shells = [key for key in arms[f"unmoved/{_arm('round', 1)}"] if key.startswith(f"{PRIMARY}/")]
    kinds = [kind for kind in KINDS if f"unmoved/{_arm(kind, 2)}" in arms]
    copies = [copy for copy in COPIES if f"{copy}/{_arm('round', 1)}" in arms]
    fig, axes = plt.subplots(
        len(kinds),
        len(copies),
        figsize=(9.5, 0.6 + 2.9 * len(kinds)),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    top = 0.0
    for row, kind in enumerate(kinds):
        ending = "-removed" if kind == "removed" else ""
        cells = [(str(k), _arm(kind, k)) for k in range(1, rounds + 1)]
        cells += [
            ("instant\nupdate", f"pass-i9-u{ending}"),
            ("truth-\ntrained", f"truth-i9-truth{ending}"),
        ]
        cells = [(label, arm) for label, arm in cells if f"unmoved/{arm}" in arms]
        connected = rounds
        x = np.array([k if k < connected else k + 0.6 for k in range(len(cells))], dtype=float)
        for col, copy in enumerate(copies):
            ax = axes[row, col]
            values = [arms[f"{copy}/{arm}"] for _, arm in cells]
            series = [(PRIMARY, INK, "D", 2.0, 5.0)] + [
                (shell, SHELL_COLORS[k], SHELL_MARKERS[k], 1.4, 4.5)
                for k, shell in enumerate(shells)
            ]
            for key, color, marker, width, size in series:
                ys = np.array([v.get(key, np.nan) for v in values])
                top = max(top, float(np.nanmax(ys)))
                style = {
                    "color": color,
                    "marker": marker,
                    "ms": size,
                    "markeredgecolor": "white",
                    "markeredgewidth": 0.6,
                }
                ax.plot(x[:connected], ys[:connected], lw=width, **style)
                ax.plot(x[connected:], ys[connected:], lw=0, **style)
            if len(cells) > connected:
                ax.axvline((x[connected - 1] + x[connected]) / 2, color="#d9d7d2", lw=0.8)
            ax.grid(axis="y", color="#ecebe8", linewidth=0.6)
            if row == 0:
                ax.set_title(COPY_TITLES[copy], color=INK, fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel(f"{KINDS[kind]}\nmedian error (mm)")
            ax.set_xticks(x, [label for label, _ in cells], fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("round", fontsize=8)
    axes[0, 0].set_ylim(0, 1.05 * top)
    handles = [
        plt.Line2D([], [], color=INK, marker="D", lw=2, label="All shells"),
        *[
            plt.Line2D(
                [],
                [],
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                lw=1.4,
                label=f"b={key.split('/')[1]}",
            )
            for k, key in enumerate(shells)
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / f"eddy-rounds-{decision['protocol']}-rounds.png", dpi=200)
    plt.close(fig)


def comparisons_figure(decision: dict[str, Any], out: Path) -> None:
    """The rules on the error: each comparison's median difference, with its intervals."""
    from eddy_targets_figures import INK, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    groups = [
        (rule, title) for rule, title in RULES if rule != "rule4" and decision[rule]["comparisons"]
    ]
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(6.4, 1.2 + 1.45 * len(groups)), sharex=True, layout="constrained"
    )
    margin = decision["margins"][PRIMARY]
    for ax, (rule, title) in zip(np.atleast_1d(axes), groups, strict=True):
        listed = list(decision[rule]["comparisons"].items())
        for y, (_, c) in enumerate(listed):
            n = 1 + len(c["shells"])
            offsets = np.linspace(0.27, -0.27, n) if n > 1 else np.zeros(1)
            _interval_row(ax, -y, c, offsets)
        ax.set_yticks(
            [-y for y in range(len(listed))], [COPY_TITLES[copy] for copy, _ in listed], fontsize=8
        )
        ax.axvline(0, color=INK, lw=0.8)
        for edge in (-margin, margin):
            ax.axvline(edge, color=INK, lw=0.8, ls=":")
        verdict = "holds" if decision[rule]["holds"] else "fails"
        ax.set_title(f"{title}: {verdict}", color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", color="#ecebe8", linewidth=0.6)
    np.atleast_1d(axes)[-1].set_xlabel(
        "median difference in error, candidate minus reference (mm)"
    )
    shells = sorted({s for _, c in decision["rule1"]["comparisons"].items() for s in c["shells"]})
    handles = [
        plt.Line2D([], [], color=INK, marker="D", lw=2, label="All shells"),
        *[
            plt.Line2D(
                [], [], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], lw=1.6, label=f"b={s}"
            )
            for k, s in enumerate(sorted(shells, key=float))
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / f"eddy-rounds-{decision['protocol']}-comparisons.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the run's folder")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="where the tables go")
    parser.add_argument("--subset", default=SUBSET, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    rows = read(args.results / f"{args.protocol}.tsv")
    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    decision = decide(rows, meta, args.subset)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"eddy-rounds-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    rounds_figure(decision, args.out, int(meta["Rounds"]))
    comparisons_figure(decision, args.out)
    print(
        ", ".join(f"{rule} {decision[rule]['holds']}" for rule, _ in RULES)
        + f"; rounds needed {decision['reported']['rounds_needed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
