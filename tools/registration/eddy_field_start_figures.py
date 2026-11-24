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
"""The decision, tables and figure of Test 7 on #735: the 9-parameter pass from the fields
of one model for all volumes.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_field_start.py`` wrote.
Comparisons are Test 4's (:func:`eddy_pass_parts_figures.compare`: paired by volume,
candidate minus reference, stratified by shell, on the confirmation volumes), with its
margins. Within a rule, each interval is widened for the rule's number of comparisons
(Bonferroni). The reference of every rule is Test 4's delayed pass, ``hm-i9-hm``:

Rule 1, the start
    ``hm-i9-field`` on ``m2s0e2``: the start counts if "better" on the error.
Rule 2, the field
    ``hm-field`` on ``e2`` and ``m2s0e2``: "better" on ``field_mean``.
Rule 3, invented field
    ``hm-field`` on ``unmoved`` and ``m2s0``: "better" or "not worse" on ``field_mean``.
Rule 4, the two passes in a row
    ``field-i9-field`` on ``m2s0e2``: "better" on the error.

Reported beside (95%): the rules' comparisons on every copy; ``hm-field``'s error against
the delayed pass; ``field-i9-field`` against ``hm-field``, on the error and on the field;
each arm against today's pass (``pass-e12-u``); each arm's medians, and how far it moved
from its start (``moved_mean``); the fit's exponent, matrix and stages on each copy.

Writes ``<out>/eddy-field-start-<protocol>-decision.json``, its tables, and a figure of
the rules' comparisons (``-comparisons.png``). The report's results come from two more
tables, the methods' medians (``-methods.csv``) and their paired differences (95%,
``-differences.csv``), and a figure of each method's error on each copy
(``-methods.png``)::

    uv run tools/registration/eddy_field_start_figures.py --results <results> --out <out>
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
    _select,
    _write,
    compare,
    read,
    summary,
)

DELAYED = "hm-i9-hm"
"""The reference of every rule: Test 4's delayed pass."""
FIELD = "field_mean"
"""The field's error (mm)."""
RULES = {
    "rule1": ("Rule 1: the start, the pass from the fit", "hm-i9-field", PRIMARY, ("m2s0e2",)),
    "rule2": ("Rule 2: the field, the fit", "hm-field", FIELD, ("e2", "m2s0e2")),
    "rule3": ("Rule 3: invented field, the fit", "hm-field", FIELD, ("unmoved", "m2s0")),
    "rule4": ("Rule 4: the two passes in a row", "field-i9-field", PRIMARY, ("m2s0e2",)),
}
"""Each rule's title, candidate, score and copies, against the delayed pass."""
BETTER = ("rule1", "rule2", "rule4")
"""The rules that need "better"; rule 3 needs "better" or "not worse"."""
ARMS = ("hm-field", "hm-i9-field", "field-i9-field", DELAYED, "hm-i9-truth", "pass-e12-u")
"""The arms the tables list, in order."""
METHODS = {
    "pass-e12-u": "Today's pass",
    DELAYED: "Delayed pass",
    "hm-field": "Fit alone",
    "field-i9-field": "Fit, then the pass",
}
"""The methods the results compare, as the report names them."""
ORDER = ("m2s0e2", "e2", "m2s0", "unmoved")
"""The copies, the one with head motion and eddy currents first."""
TITLES = {
    "m2s0e2": "Head motion and eddy currents",
    "e2": "Eddy currents only",
    "m2s0": "Head motion only",
    "unmoved": "Neither",
}
"""The copies, as the report names them."""
DIFFERENCES = (
    ("Fit, then the pass, minus today's pass", "field-i9-field", "pass-e12-u", PRIMARY),
    ("Fit, then the pass, minus the delayed pass", "field-i9-field", DELAYED, PRIMARY),
    ("Fit alone minus today's pass", "hm-field", "pass-e12-u", PRIMARY),
    ("Delayed pass minus today's pass", DELAYED, "pass-e12-u", PRIMARY),
    ("Fit, then the pass, minus the fit alone", "field-i9-field", "hm-field", PRIMARY),
    (
        "The pass from the fit's field minus the pass from the head-motion step",
        "hm-i9-field",
        DELAYED,
        PRIMARY,
    ),
    ("Fit alone minus the delayed pass", "hm-field", DELAYED, FIELD),
    ("Fit, then the pass, minus the fit alone", "field-i9-field", "hm-field", FIELD),
)
"""The paired differences the report lists: label, candidate, reference and score."""


def _rule(comparisons: dict[str, dict[str, Any]], better: bool) -> dict[str, Any]:
    """A rule holds if it has comparisons and each has the verdict it needs."""
    holds = bool(comparisons) and all(
        c["verdict"] == "better" if better else _ok(c) for c in comparisons.values()
    )
    return {"holds": holds, "comparisons": comparisons}


def _follows(decision: dict[str, Any]) -> str:
    """What follows, as the working hypothesis set it out."""
    if decision["rule1"]["holds"] or decision["rule4"]["holds"]:
        return "the fit, then the 9-parameter pass, is the candidate; HCP-YA next"
    if decision["rule2"]["holds"]:
        return "the gain is in the field alone; next, head motion with the fitted field held"
    if decision["rule2"]["comparisons"]:
        return "with these targets, the shared model does not help"
    return "undecided: rule 2 has no comparisons"


def decide(
    rows: list[dict[str, Any]], meta: dict[str, Any], subset: str | None = SUBSET
) -> dict[str, Any]:
    """Test 7's decision: the rules, and what is reported beside."""
    margins = _margins(meta)
    copies = [copy for copy in COPIES if copy in meta["Copies"]]
    present = {(row["copy"], row["arm"]) for row in rows}

    def paired(copy, candidate, reference, key=PRIMARY, tests=1):
        chosen = ((copy, candidate), (copy, reference))
        return compare(rows, *chosen, key, margins[key], tests=tests, subset=subset)

    def every(candidate, reference, key=PRIMARY):
        return {
            copy: paired(copy, candidate, reference, key)
            for copy in copies
            if {(copy, candidate), (copy, reference)} <= present
        }

    decision: dict[str, Any] = {"protocol": meta["Protocol"], "margins": margins}
    for rule, (_, candidate, key, ruled) in RULES.items():
        chosen = [copy for copy in ruled if (copy, candidate) in present]
        decision[rule] = _rule(
            {copy: paired(copy, candidate, DELAYED, key, tests=len(ruled)) for copy in chosen},
            better=rule in BETTER,
        )
    decision["follows"] = _follows(decision)

    moved = {}
    for copy in copies:
        for arm in ARMS:
            chosen = list(_select(rows, copy, arm, subset).values())
            if chosen:
                moved[f"{copy}/{arm}"] = float(np.median([row["moved_mean"] for row in chosen]))
    decision["reported"] = {
        "start": every("hm-i9-field", DELAYED),
        "field": every("hm-field", DELAYED, FIELD),
        "chain": every("field-i9-field", DELAYED),
        "fit_error": every("hm-field", DELAYED),
        "chain_against_fit": every("field-i9-field", "hm-field"),
        "chain_against_fit_field": every("field-i9-field", "hm-field", FIELD),
        "chain_rotation": every("field-i9-field", DELAYED, "motion_rotation_deg"),
        "against_today": {
            f"{copy}/{arm}": c for arm in ARMS[:-1] for copy, c in every(arm, "pass-e12-u").items()
        },
        "moved_mean": moved,
    }
    decision["summary"] = summary(rows, subset)
    decision["fields"] = meta["Fields"]
    decision["checks"] = meta["Checks"]
    return decision


def _table(listed: list[tuple[str, str, dict[str, Any]]], names: list[str]) -> list[list[Any]]:
    """Each comparison's row; a shell it does not have is left blank."""
    return [
        [
            group,
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
        for group, label, c in listed
    ]


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the rules' comparisons, those reported, and the fits."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    shells = sorted(
        {key for values in arms.values() for key in values if key.startswith(f"{PRIMARY}/")},
        key=lambda key: float(key.split("/")[1]),
    )
    _write(
        out / f"eddy-field-start-{protocol}-arms.csv",
        [
            "Copy",
            "Arm",
            "Error (mm)",
            *(f"b={key.split('/')[1]}" for key in shells),
            "Rotation (deg)",
            "Across PE (mm)",
            "Field (mm)",
            "Moved from the start (mm)",
        ],
        [
            [
                *name.split("/"),
                f"{values[PRIMARY]:.3f}",
                *(f"{values.get(key, np.nan):.3f}" for key in shells),
                f"{values['motion_rotation_deg']:.3f}",
                f"{values['motion_across_mean']:.3f}",
                f"{values['field_mean']:.3f}",
                f"{decision['reported']['moved_mean'].get(name, np.nan):.3f}",
            ]
            for name, values in arms.items()
        ],
    )
    listed = [
        (RULES[rule][0], label, c)
        for rule in RULES
        for label, c in decision[rule]["comparisons"].items()
    ]
    names = sorted({s for _, _, c in listed for s in c["shells"]}, key=float)
    header = ["Group", "Comparison", "Score", "Median difference [interval]", "Verdict"]
    header += [f"b={s}" for s in names]
    _write(out / f"eddy-field-start-{protocol}-comparisons.csv", header, _table(listed, names))
    listed = [
        (group, label, c)
        for group, comparisons in decision["reported"].items()
        if group != "moved_mean"
        for label, c in comparisons.items()
    ]
    _write(out / f"eddy-field-start-{protocol}-reported.csv", header, _table(listed, names))
    _write(
        out / f"eddy-field-start-{protocol}-fits.csv",
        ["Copy", "Exponent", "C (row by row)", "Evaluations", "Stages", "Time (s)"],
        [
            [
                copy,
                f"{fit['Exponent']:.3f}",
                " ; ".join(" ".join(f"{v:+.5f}" for v in row) for row in fit["Matrix"]),
                ",".join(str(n) for n in fit["Evaluations"]),
                " | ".join(fit["Messages"]),
                f"{fit['FitSeconds']:.0f}",
            ]
            for copy, fit in decision["fields"].items()
        ],
    )


def comparisons_figure(decision: dict[str, Any], out: Path) -> None:
    """The rules: each comparison's median difference, with its intervals."""
    from eddy_targets_figures import INK, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    groups = [rule for rule in RULES if decision[rule]["comparisons"]]
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(6.4, 1.0 + 1.25 * len(groups)), sharex=True, layout="constrained"
    )
    for ax, rule in zip(np.atleast_1d(axes), groups, strict=True):
        title, _, key, _ = RULES[rule]
        listed = list(decision[rule]["comparisons"].items())
        for y, (_, c) in enumerate(listed):
            n = 1 + len(c["shells"])
            offsets = np.linspace(0.27, -0.27, n) if n > 1 else np.zeros(1)
            _interval_row(ax, -y, c, offsets)
        ax.set_yticks(
            [-y for y in range(len(listed))], [COPY_TITLES[copy] for copy, _ in listed], fontsize=8
        )
        ax.set_ylim(-len(listed) + 0.5, 0.5)
        ax.axvline(0, color=INK, lw=0.8)
        margin = decision["margins"][key]
        for edge in (-margin, margin):
            ax.axvline(edge, color=INK, lw=0.8, ls=":")
        score = "error" if key == PRIMARY else "field"
        verdict = "holds" if decision[rule]["holds"] else "fails"
        ax.set_title(f"{title} ({score}): {verdict}", color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", color="#ecebe8", linewidth=0.6)
    np.atleast_1d(axes)[-1].set_xlabel(
        "median difference, candidate minus the delayed 9-parameter pass (mm)"
    )
    shells = sorted(
        {
            s
            for rule in groups
            for c in decision[rule]["comparisons"].values()
            for s in c["shells"]
        },
        key=float,
    )
    handles = [
        plt.Line2D([], [], color=INK, marker="D", lw=2, label="All shells"),
        *[
            plt.Line2D(
                [], [], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], lw=1.6, label=f"b={s}"
            )
            for k, s in enumerate(shells)
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / f"eddy-field-start-{decision['protocol']}-comparisons.png", dpi=200)
    plt.close(fig)


def results_tables(
    rows: list[dict[str, Any]], decision: dict[str, Any], out: Path, subset: str | None = SUBSET
) -> None:
    """The methods' medians and their paired differences (95%), as the report names them."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    copies = [copy for copy in ORDER if f"{copy}/{DELAYED}" in arms]
    shells = sorted(
        {
            key.split("/")[1]
            for values in arms.values()
            for key in values
            if key.startswith(f"{PRIMARY}/")
        },
        key=float,
    )
    _write(
        out / f"eddy-field-start-{protocol}-methods.csv",
        ["Copy", "Method", "Error (mm)", *(f"b={s}" for s in shells), "Field (mm)"],
        [
            [
                TITLES[copy],
                name,
                f"{values[PRIMARY]:.3f}",
                *(f"{values.get(f'{PRIMARY}/{s}', np.nan):.3f}" for s in shells),
                f"{values[FIELD]:.3f}",
            ]
            for copy in copies
            for arm, name in METHODS.items()
            if (values := arms.get(f"{copy}/{arm}"))
        ],
    )
    margins = decision["margins"]
    listed = []
    for copy in copies:
        for label, candidate, reference, key in DIFFERENCES:
            if f"{copy}/{candidate}" not in arms or f"{copy}/{reference}" not in arms:
                continue
            chosen = ((copy, candidate), (copy, reference))
            c = compare(rows, *chosen, key, margins[key], subset=subset)
            listed.append(
                [
                    TITLES[copy],
                    label,
                    "error" if key == PRIMARY else "field",
                    _interval(c),
                    *(
                        f"{c['shells'][s]['median']:+.3f}" if s in c["shells"] else ""
                        for s in shells
                    ),
                ]
            )
    _write(
        out / f"eddy-field-start-{protocol}-differences.csv",
        ["Copy", "Difference", "Score", "Median [95%] (mm)", *(f"b={s}" for s in shells)],
        listed,
    )


def methods_figure(decision: dict[str, Any], out: Path) -> None:
    """Each method's median error on each copy, overall (bars) and by shell (marks)."""
    from eddy_targets_figures import INK, MUTED, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    arms = decision["summary"]
    copies = [copy for copy in ORDER if f"{copy}/{DELAYED}" in arms]
    shells = sorted(
        {
            key.split("/")[1]
            for values in arms.values()
            for key in values
            if key.startswith(f"{PRIMARY}/")
        },
        key=float,
    )
    fig, axes = plt.subplots(
        1, len(copies), figsize=(9.5, 3.2), sharey=True, layout="constrained", squeeze=False
    )
    top = 0.0
    x = np.arange(len(METHODS))
    offsets = np.linspace(-0.22, 0.22, len(shells))
    for ax, copy in zip(axes[0], copies, strict=True):
        values = [arms[f"{copy}/{arm}"] for arm in METHODS]
        overall = np.array([v[PRIMARY] for v in values])
        colors = ["#d9d8d4"] * (len(METHODS) - 1) + ["#a9a7a2"]
        ax.bar(x, overall, width=0.72, color=colors, edgecolor="white", linewidth=2, zorder=1)
        for xi, value in zip(x, overall, strict=True):
            # Inside the bar's foot, clear of the shells' marks
            ax.annotate(
                f"{value:.2f}",
                (xi, 0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
        for k, s in enumerate(shells):
            ys = np.array([v.get(f"{PRIMARY}/{s}", np.nan) for v in values])
            top = max(top, float(np.nanmax(ys)), float(overall.max()))
            ax.plot(
                x + offsets[k],
                ys,
                lw=0,
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                ms=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=2,
            )
        ax.set_title(TITLES[copy], color=INK, fontsize=9, loc="left")
        ax.set_xticks(
            x, [name.replace(", ", ",\n").replace(" ", "\n", 1) for name in METHODS.values()]
        )
        ax.tick_params(axis="x", labelsize=7, length=0)
        ax.grid(axis="y", color="#ecebe8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("median error (mm)", color=MUTED)
    axes[0, 0].set_ylim(0, 1.1 * top)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d9d8d4", label="All shells"),
        *[
            plt.Line2D(
                [],
                [],
                lw=0,
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                markeredgecolor="white",
                label=f"b={s}",
            )
            for k, s in enumerate(shells)
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / f"eddy-field-start-{decision['protocol']}-methods.png", dpi=200)
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
    (args.out / f"eddy-field-start-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    comparisons_figure(decision, args.out)
    results_tables(rows, decision, args.out, args.subset)
    methods_figure(decision, args.out)
    print(", ".join(f"{rule} {decision[rule]['holds']}" for rule in RULES))
    print(f"follows: {decision['follows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
