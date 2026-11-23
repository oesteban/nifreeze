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
"""The decision, tables and figure of Test 6 on #735: the pass on HCP-YA.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_pass_hcpya.py`` wrote.
Comparisons are Test 4's (:func:`eddy_pass_parts_figures.compare`: paired by volume,
candidate minus reference, stratified by shell), on every volume, with a margin of 0.01
voxel (0.0125 mm on HCP-YA). ``x`` marks an arm with the shared field removed; the
highest shell is b=3000:

Rule 1, the confirmation
    ``hm-i9-hm-x`` against today's pass (``pass-e12-u``), over all shells: "better".
Rule 2, every shell
    The same, at each shell: "better" or "not worse" (each interval widened for the
    3 shells, Bonferroni).
Rule 3, the sampling
    At the highest shell, ``hm-i9s-hm-x`` against ``hm-i9-hm-x``: "better".
Rule 4, the update
    At the highest shell, ``pass-i9-u-x`` against ``hm-i9-hm-x``: "better".

Reported beside (95%): each pass with the shared field removed against as it is;
``hm-i9s-hm-x`` and ``pass-i9-u-x`` against today's pass; the delayed pass as it is
against today's (Test 3's confirmation); the registrations to the noise-free originals
(``oracle-i9s`` against ``oracle-i9``, and each against ``oracle-e12``); each arm's
medians. Writes ``<out>/eddy-hcpya-<protocol>-decision.json``, its tables, and a figure
of the comparisons (``-comparisons.png``)::

    uv run tools/registration/eddy_pass_hcpya_figures.py --results <results> --out <out>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_pass_parts_figures import (
    PRIMARY,
    _interval,
    _interval_row,
    _margins,
    _write,
    compare,
    read,
    summary,
)
from eddy_pass_rounds_figures import _rule

NAME = "m2s0e2"
"""The copy, Test 3's."""
DELAYED, SCALED, INSTANT = "hm-i9-hm-removed", "hm-i9s-hm-removed", "pass-i9-u-removed"
"""The passes with the shared field removed."""
TODAY = "pass-e12-u"
"""Today's pass."""
REPORTED = (
    ("The removal", DELAYED, "hm-i9-hm"),
    ("The removal", SCALED, "hm-i9s-hm"),
    ("The removal", INSTANT, "pass-i9-u"),
    ("Against today's pass", SCALED, TODAY),
    ("Against today's pass", INSTANT, TODAY),
    ("Against today's pass", "hm-i9-hm", TODAY),
    ("Noise-free originals", "oracle-i9s", "oracle-i9"),
    ("Noise-free originals", "oracle-i9s", "oracle-e12"),
    ("Noise-free originals", "oracle-i9", "oracle-e12"),
)
"""The comparisons reported beside the rules: (group, candidate, reference)."""
TITLES = {
    "hm-i9-hm-removed": "delayed pass",
    "hm-i9s-hm-removed": "delayed pass, scaled samples",
    "pass-i9-u-removed": "instant update",
    "hm-i9-hm": "delayed pass, as it is",
    "hm-i9s-hm": "delayed pass, scaled samples, as it is",
    "pass-i9-u": "instant update, as it is",
    "pass-e12-u": "today's pass",
    "oracle-i9s": "9 parameters, scaled samples",
    "oracle-i9": "9 parameters",
    "oracle-e12": "elastix",
}
"""How the figure names the arms (the passes with the shared field removed, unless said)."""


def decide(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Test 6's decision: the rules, and what is reported beside."""
    margins = _margins(meta)
    present = {row["arm"] for row in rows}
    shells = sorted({row["shell"] for row in rows})
    top = shells[-1]

    def paired(candidate: str, reference: str, table=rows, tests: int = 1) -> dict[str, Any]:
        return compare(
            table,
            (NAME, candidate),
            (NAME, reference),
            PRIMARY,
            margins[PRIMARY],
            tests=tests,
            subset=None,
        )

    def at(shell: float) -> list[dict[str, Any]]:
        return [row for row in rows if row["shell"] == shell]

    highest = f"b={int(top)}"
    rule1 = {"all": paired(DELAYED, TODAY)}
    rule2 = {f"b={int(s)}": paired(DELAYED, TODAY, at(s), tests=len(shells)) for s in shells}
    rule3 = {highest: paired(SCALED, DELAYED, at(top))} if SCALED in present else {}
    rule4 = {highest: paired(INSTANT, DELAYED, at(top))} if INSTANT in present else {}
    reported = {
        f"{group}/{candidate}/{reference}": paired(candidate, reference)
        for group, candidate, reference in REPORTED
        if {candidate, reference} <= present
    }
    return {
        "protocol": meta["Protocol"],
        "margins": margins,
        "rule1": _rule(rule1, better=True),
        "rule2": _rule(rule2),
        "rule3": _rule(rule3, better=True),
        "rule4": _rule(rule4, better=True),
        "reported": reported,
        "summary": {arm.split("/", 1)[1]: values for arm, values in summary(rows, None).items()},
        "samples": meta["Samples"],
        "checks": meta["Checks"],
    }


RULES = (
    ("rule1", "Rule 1: the delayed pass against today's"),
    ("rule2", "Rule 2: the delayed pass against today's, by shell"),
    ("rule3", "Rule 3: scaled samples against 17,000, highest shell"),
    ("rule4", "Rule 4: the instant update against the delayed, highest shell"),
)
"""The rules, as the tables and figure label them."""


def _row(title: str, label: str, c: dict[str, Any], names: list[str]) -> list[Any]:
    return [
        title,
        f"{c['candidate']} against {c['reference']} ({label})",
        _interval(c),
        c["verdict"],
        *(
            f"{c['shells'][s]['median']:+.3f} ({c['shells'][s]['verdict']})"
            if s in c["shells"]
            else ""
            for s in names
        ),
    ]


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the rules' comparisons, and those reported."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    shells = [key for key in next(iter(arms.values())) if key.startswith(f"{PRIMARY}/")]
    _write(
        out / f"eddy-hcpya-{protocol}-arms.csv",
        [
            "Arm",
            "Error (mm)",
            *(f"b={key.split('/')[1]}" for key in shells),
            "Rotation (deg)",
            "Field (mm)",
            "Volume change",
            "Strain along PE",
            "Time (s)",
        ],
        [
            [
                arm,
                f"{values[PRIMARY]:.3f}",
                *(f"{values.get(key, np.nan):.3f}" for key in shells),
                f"{values['motion_rotation_deg']:.3f}",
                f"{values['field_mean']:.3f}",
                f"{100 * values['volume_change'] + 0.0:+.3f}%",
                f"{100 * values['strain_pe'] + 0.0:+.3f}%",
                "" if np.isnan(values["seconds"]) else f"{values['seconds']:.2f}",
            ]
            for arm, values in arms.items()
        ],
    )
    names = [key.split("/")[1] for key in shells]
    header = ["Rule", "Comparison", "Median difference [interval]", "Verdict"]
    header += [f"b={s}" for s in names]
    listed = [
        _row(title, label, c, names)
        for rule, title in RULES
        for label, c in decision[rule]["comparisons"].items()
    ]
    _write(out / f"eddy-hcpya-{protocol}-comparisons.csv", header, listed)
    listed = [
        _row(key.split("/")[0], "all shells", c, names) for key, c in decision["reported"].items()
    ]
    _write(out / f"eddy-hcpya-{protocol}-reported.csv", header, listed)


def comparisons_figure(decision: dict[str, Any], out: Path) -> None:
    """The rules, and the registrations to the noise-free originals, on the error."""
    from eddy_targets_figures import INK, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    groups = [
        (f"{title}: {'holds' if decision[rule]['holds'] else 'fails'}", rule)
        for rule, title in RULES
        if rule != "rule2" and decision[rule]["comparisons"]
    ]
    oracle = {key: c for key, c in decision["reported"].items() if key.startswith("Noise-free")}
    panels = [(title, list(decision[rule]["comparisons"].values())) for title, rule in groups]
    if oracle:
        panels.append(("Reported: registered to the noise-free originals", list(oracle.values())))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(6.4, 1.0 + sum(0.55 + 0.55 * len(listed) for _, listed in panels)),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [len(listed) for _, listed in panels]},
    )
    margin = decision["margins"][PRIMARY]
    shells = sorted(
        {s for c in decision["rule1"]["comparisons"].values() for s in c["shells"]}, key=float
    )
    for ax, (title, listed) in zip(np.atleast_1d(axes), panels, strict=True):
        for y, c in enumerate(listed):
            if len(c["shells"]) == 1:
                # A comparison on one shell: its interval, in that shell's colour
                k = shells.index(next(iter(c["shells"])))
                ax.plot([c["median_low"], c["median_high"]], [-y] * 2, color=SHELL_COLORS[k], lw=2)
                ax.scatter(
                    [c["median"]], [-y], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], s=30
                )
                continue
            offsets = np.linspace(0.27, -0.27, 1 + len(c["shells"]))
            _interval_row(ax, -y, c, offsets)
        labels = [
            f"{TITLES[c['candidate'].split('/')[1]]}\nagainst "
            f"{TITLES[c['reference'].split('/')[1]]}"
            for c in listed
        ]
        ax.set_yticks([-y for y in range(len(listed))], labels, fontsize=7)
        ax.set_ylim(-len(listed) + 0.5, 0.5)
        ax.axvline(0, color=INK, lw=0.8)
        for edge in (-margin, margin):
            ax.axvline(edge, color=INK, lw=0.8, ls=":")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", color="#ecebe8", linewidth=0.6)
    np.atleast_1d(axes)[-1].set_xlabel(
        "median difference in error, candidate minus reference (mm)"
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
    fig.savefig(out / f"eddy-hcpya-{decision['protocol']}-comparisons.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the run's folder")
    parser.add_argument("--protocol", default="hcpya")
    parser.add_argument("--out", type=Path, required=True, help="where the tables go")
    args = parser.parse_args(argv)
    rows = read(args.results / f"{args.protocol}.tsv")
    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    decision = decide(rows, meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"eddy-hcpya-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    comparisons_figure(decision, args.out)
    print(", ".join(f"{rule} {decision[rule]['holds']}" for rule, _ in RULES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
