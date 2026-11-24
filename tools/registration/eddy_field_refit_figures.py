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
"""The decision, tables and figure of Test 9 on #735: a second fit in place of the
9-parameter pass.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_field_refit.py`` wrote.
Comparisons are Test 4's (:func:`eddy_pass_parts_figures.compare`: paired by volume,
candidate minus reference, stratified by shell, on the confirmation volumes, 95%), with a
margin of 0.02 mm.

The decision, on the copy with head motion and eddy currents: the second fit with derived
shifts takes the place of the 9-parameter pass when it is "better" or "not worse" than
Test 7's chain with the shared field removed.

Writes ``<out>/eddy-field-refit-<protocol>-{decision.json, methods.csv, differences.csv,
fits.csv, methods.png}``::

    uv run tools/registration/eddy_field_refit_figures.py --results <results> --out <out>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_pass_parts_figures import PRIMARY, SUBSET, _interval, _write, compare, read
from eddy_pass_parts_figures import summary as medians

MARGIN_MM = 0.02
"""The margin: 0.01 of Sherbrooke's 2 mm voxel, as in Test 4."""
MAIN = "m2s0e2"
"""The copy the decision is made on: head motion and eddy currents."""
CANDIDATE, CHAIN = "refit-derived", "chain-removed"
"""The second fit with derived shifts, and Test 7's chain with the shared field removed."""
TODAY, DELAYED = "pass-e12-u", "hm-i9-hm-removed"
"""Today's pass, and the delayed pass with the shared field removed."""
FIELD = "field_mean"
"""The field's error (mm)."""
NAMES = {
    TODAY: "Today's pass",
    DELAYED: "Delayed pass, shared field removed",
    CHAIN: "Fit, then the 9-parameter pass, shared field removed",
    "fit-free": "Fit alone, free shifts",
    "fit-derived": "Fit alone, derived shifts",
    "refit-free": "Fit twice, free shifts",
    "refit-derived": "Fit twice, derived shifts",
}
"""Every arm the tables list, as the report names it."""
SHORT = {
    TODAY: "Today's\npass",
    DELAYED: "Delayed\npass",
    CHAIN: "Fit, then\nthe pass",
    "fit-derived": "Fit alone\nderived",
    "refit-free": "Fit twice\nfree",
    "refit-derived": "Fit twice\nderived",
}
"""The arms of the figure, as it names them."""
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
    ("Fit twice, derived shifts, minus the fit then the pass", CANDIDATE, CHAIN, PRIMARY),
    ("Fit twice, free shifts, minus the fit then the pass", "refit-free", CHAIN, PRIMARY),
    ("Fit twice, derived shifts, minus fit twice, free shifts", CANDIDATE, "refit-free", PRIMARY),
    ("Fit twice minus fit alone, derived shifts", CANDIDATE, "fit-derived", PRIMARY),
    ("Fit twice minus fit alone, free shifts", "refit-free", "fit-free", PRIMARY),
    ("Fit alone, derived shifts, minus free shifts", "fit-derived", "fit-free", PRIMARY),
    ("Fit twice, derived shifts, minus today's pass", CANDIDATE, TODAY, PRIMARY),
    ("Fit twice, derived shifts, minus the delayed pass", CANDIDATE, DELAYED, PRIMARY),
    ("Fit twice, derived shifts, minus the fit then the pass", CANDIDATE, CHAIN, FIELD),
    ("Fit twice minus fit alone, derived shifts", CANDIDATE, "fit-derived", FIELD),
)
"""The paired differences reported: label, candidate, reference and score."""


def decide(rows: list[dict[str, Any]], meta: dict[str, Any], subset: str | None = SUBSET):
    """Test 9's decision, and every comparison reported."""
    copies = [copy for copy in ORDER if copy in meta["Copies"]]

    def paired(copy, candidate, reference, key=PRIMARY):
        return compare(rows, (copy, candidate), (copy, reference), key, MARGIN_MM, subset=subset)

    main = paired(MAIN, CANDIDATE, CHAIN)
    decision: dict[str, Any] = {
        "protocol": meta["Protocol"],
        "margin": MARGIN_MM,
        "main": main,
        "replaces_the_pass": main["verdict"] in ("better", "not worse"),
        "reported": [
            {"copy": copy, "label": label, "comparison": paired(copy, cand, ref, key)}
            for copy in copies
            for label, cand, ref, key in DIFFERENCES
        ],
        "summary": medians(rows, subset),
        "fields": meta["Fields"],
        "checks": meta["Checks"],
    }
    return decision


def _shells(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            key.split("/")[1]
            for values in summary.values()
            for key in values
            if key.startswith(f"{PRIMARY}/")
        },
        key=float,
    )


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the paired differences, and the fits."""
    protocol, summary = decision["protocol"], decision["summary"]
    shells = _shells(summary)
    _write(
        out / f"eddy-field-refit-{protocol}-methods.csv",
        ["Copy", "Method", "Error (mm)", *(f"b={s}" for s in shells), "Field (mm)"],
        [
            [
                TITLES[copy],
                name,
                f"{values[PRIMARY]:.3f}",
                *(f"{values.get(f'{PRIMARY}/{s}', np.nan):.3f}" for s in shells),
                f"{values[FIELD]:.3f}",
            ]
            for copy in ORDER
            for arm, name in NAMES.items()
            if (values := summary.get(f"{copy}/{arm}"))
        ],
    )
    _write(
        out / f"eddy-field-refit-{protocol}-differences.csv",
        ["Copy", "Difference", "Score", "Median [95%] (mm)", *(f"b={s}" for s in shells)],
        [
            [
                TITLES[item["copy"]],
                item["label"],
                "error" if item["comparison"]["key"] == PRIMARY else "field",
                _interval(item["comparison"]),
                *(
                    f"{item['comparison']['shells'][s]['median']:+.3f}"
                    if s in item["comparison"]["shells"]
                    else ""
                    for s in shells
                ),
            ]
            for item in decision["reported"]
        ],
    )
    rows = []
    for key, fit in decision["fields"].items():
        copy, arm = key.split("/")
        rows.append(
            [
                TITLES[copy],
                NAMES[arm],
                f"{fit['Exponent']:.3f}",
                ", ".join(f"{v:+.4f}" for v in np.diag(np.asarray(fit["Matrix"]))),
                ", ".join(str(n) for n in fit["Evaluations"]),
                "yes" if all("CONVERGENCE" in m for m in fit["Messages"]) else "no",
                f"{fit['FitSeconds']:.0f}",
            ]
        )
    _write(
        out / f"eddy-field-refit-{protocol}-fits.csv",
        ["Copy", "Fit", "Exponent", "Diagonal of C", "Evaluations", "Converged", "Time (s)"],
        rows,
    )


def figure(decision: dict[str, Any], out: Path) -> None:
    """Each arm's median error on each copy, overall (bars) and by shell (marks)."""
    from eddy_targets_figures import INK, MUTED, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    summary = decision["summary"]
    copies = [copy for copy in ORDER if f"{copy}/{CANDIDATE}" in summary]
    shells = _shells(summary)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharey=True, layout="constrained")
    top = 0.0
    x = np.arange(len(SHORT))
    offsets = np.linspace(-0.22, 0.22, len(shells))
    chosen = list(SHORT).index(CANDIDATE)
    for ax, copy in zip(axes.ravel(), copies, strict=False):
        values = [summary[f"{copy}/{arm}"] for arm in SHORT]
        overall = np.array([v[PRIMARY] for v in values])
        colors = ["#d9d8d4"] * len(SHORT)
        colors[chosen] = "#6f6d68"
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
                color="white" if xi == chosen else INK,
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
        ax.set_xticks(x, list(SHORT.values()))
        ax.tick_params(axis="x", labelsize=7, length=0)
        ax.grid(axis="y", color="#ecebe8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("median error (mm)", color=MUTED)
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
    fig.savefig(out / f"eddy-field-refit-{decision['protocol']}-methods.png", dpi=200)
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
    (args.out / f"eddy-field-refit-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    figure(decision, args.out)
    main = decision["main"]
    print(
        f"fit twice (derived) minus the fit then the pass: {_interval(main)} ({main['verdict']})"
    )
    print(f"replaces the pass: {decision['replaces_the_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
