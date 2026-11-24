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
"""The decision, tables and figure of Test 8 on #735: a matrix C for each shell, or shifts
derived from one zero point.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_field_variants.py``
wrote. Comparisons are Test 4's (:func:`eddy_pass_parts_figures.compare`: paired by
volume, candidate minus reference, stratified by shell, on the confirmation volumes, 95%),
with its margin of 0.02 mm.

The decision, on the copy with head motion and eddy currents: a variant goes forward when
its chain (the fit, then the pass, with the shared field removed) is "better" than Test
7's (``cp-free``); of several, the one with the lowest median error. When none is, Test
7's goes forward.

Writes ``<out>/eddy-field-variants-<protocol>-{decision.json, methods.csv,
differences.csv, fits.csv, methods.png}``::

    uv run tools/registration/eddy_field_variants_figures.py --results <results> --out <out>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_pass_parts_figures import PRIMARY, SUBSET, _interval, _write, compare, read
from eddy_pass_parts_figures import summary as medians

MARGIN_MM = 0.02
"""The working hypothesis's margin: 0.01 of Sherbrooke's 2 mm voxel, as in Test 4."""
MAIN = "m2s0e2"
"""The copy the decision is made on: head motion and eddy currents."""
REFERENCE = "cp-free"
"""Test 7's fit: one C and p, each volume's shift estimated."""
FIELD = "field_mean"
"""The field's error (mm)."""
VARIANTS = {
    "cp-free": "One C and p, free shifts",
    "cp-derived": "One C and p, derived shifts",
    "shells-free": "A C per shell, free shifts",
    "shells-derived": "A C per shell, derived shifts",
}
"""The variants of the fit, as the report names them."""
SHORT = {
    "cp-free": "C, p\nfree",
    "cp-derived": "C, p\nderived",
    "shells-free": "C per shell\nfree",
    "shells-derived": "C per shell\nderived",
}
"""The variants, as the figure names them."""
ORDER = ("m2s0e2", "e2", "m2s0", "unmoved")
"""The copies, the one with head motion and eddy currents first."""
TITLES = {
    "m2s0e2": "Head motion and eddy currents",
    "e2": "Eddy currents only",
    "m2s0": "Head motion only",
    "unmoved": "Neither",
}
"""The copies, as the report names them."""
TODAY, DELAYED = "pass-e12-u", "hm-i9-hm-removed"
"""Today's pass, and the delayed pass with the shared field removed."""


def _chain(variant: str) -> str:
    return f"chain-{variant}-removed"


def methods() -> dict[str, str]:
    """Every arm the tables list, and its name."""
    named = {TODAY: "Today's pass", DELAYED: "Delayed pass, shared field removed"}
    named |= {f"fit-{v}": f"Fit alone: {name}" for v, name in VARIANTS.items()}
    named |= {f"chain-{v}": f"Fit, then the pass: {name}" for v, name in VARIANTS.items()}
    named |= {
        _chain(v): f"Fit, then the pass, shared field removed: {name}"
        for v, name in VARIANTS.items()
    }
    return named


def decide(rows: list[dict[str, Any]], meta: dict[str, Any], subset: str | None = SUBSET):
    """Test 8's decision, and every comparison reported."""
    margin = MARGIN_MM
    copies = [copy for copy in ORDER if copy in meta["Copies"]]

    def paired(copy, candidate, reference, key=PRIMARY):
        return compare(rows, (copy, candidate), (copy, reference), key, margin, subset=subset)

    decision: dict[str, Any] = {"protocol": meta["Protocol"], "margin": margin}
    against = {v: paired(MAIN, _chain(v), _chain(REFERENCE)) for v in VARIANTS if v != REFERENCE}
    better = [v for v, c in against.items() if c["verdict"] == "better"]
    summary = medians(rows, subset)
    chosen = (
        min(better, key=lambda v: summary[f"{MAIN}/{_chain(v)}"][PRIMARY]) if better else REFERENCE
    )
    decision |= {"against_test7": against, "better": better, "chosen": chosen}

    listed = []
    for copy in copies:
        for v, name in VARIANTS.items():
            if v != REFERENCE:
                listed.append(
                    (
                        copy,
                        f"{name}, minus Test 7's (chain)",
                        _chain(v),
                        _chain(REFERENCE),
                        PRIMARY,
                    )
                )
                listed.append(
                    (
                        copy,
                        f"{name}, minus Test 7's (fit alone)",
                        f"fit-{v}",
                        f"fit-{REFERENCE}",
                        PRIMARY,
                    )
                )
            listed.append(
                (copy, f"{name}, minus the delayed pass (chain)", _chain(v), DELAYED, PRIMARY)
            )
            listed.append((copy, f"{name}, minus today's pass (chain)", _chain(v), TODAY, PRIMARY))
            listed.append(
                (copy, f"{name}, minus the delayed pass (chain)", _chain(v), DELAYED, FIELD)
            )
    decision["reported"] = [
        {"copy": copy, "label": label, "comparison": paired(copy, candidate, reference, key)}
        for copy, label, candidate, reference, key in listed
    ]
    decision["summary"] = summary
    decision["fields"] = meta["Fields"]
    decision["checks"] = meta["Checks"]
    return decision


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each method's medians, the paired differences, and the fits."""
    protocol, summary = decision["protocol"], decision["summary"]
    shells = sorted(
        {key.split("/")[1] for values in summary.values() for key in values if "/" in key},
        key=float,
    )
    named = methods()
    _write(
        out / f"eddy-field-variants-{protocol}-methods.csv",
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
            for arm, name in named.items()
            if (values := summary.get(f"{copy}/{arm}"))
        ],
    )
    _write(
        out / f"eddy-field-variants-{protocol}-differences.csv",
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
        copy, variant = key.split("/")
        diagonals = "; ".join(
            ", ".join(f"{v:+.4f}" for v in np.diag(np.asarray(matrix)))
            for matrix in fit["Matrices"]
        )
        zero = np.asarray(fit["ZeroPoint"], dtype=float)
        rows.append(
            [
                TITLES[copy],
                VARIANTS[variant],
                f"{fit['Exponent']:.3f}",
                diagonals,
                "" if np.isnan(zero).any() else ", ".join(f"{v:+.1f}" for v in zero),
                ", ".join(str(n) for n in fit["Evaluations"]),
                f"{fit['FitSeconds']:.0f}",
            ]
        )
    _write(
        out / f"eddy-field-variants-{protocol}-fits.csv",
        [
            "Copy",
            "Variant",
            "Exponent",
            "Diagonal of each C",
            "Zero point (mm)",
            "Evaluations",
            "Time (s)",
        ],
        rows,
    )


def figure(decision: dict[str, Any], out: Path) -> None:
    """Each method's median error on each copy, overall (bars) and by shell (marks)."""
    from eddy_targets_figures import INK, MUTED, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    summary = decision["summary"]
    shown = {TODAY: "Today's\npass", DELAYED: "Delayed\npass"}
    shown |= {_chain(v): SHORT[v] for v in VARIANTS}
    copies = [copy for copy in ORDER if f"{copy}/{DELAYED}" in summary]
    shells = sorted(
        {key.split("/")[1] for values in summary.values() for key in values if "/" in key},
        key=float,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharey=True, layout="constrained")
    top = 0.0
    x = np.arange(len(shown))
    offsets = np.linspace(-0.22, 0.22, len(shells))
    for ax, copy in zip(axes.ravel(), copies, strict=False):
        values = [summary[f"{copy}/{arm}"] for arm in shown]
        overall = np.array([v[PRIMARY] for v in values])
        colors = ["#d9d8d4"] * 2 + ["#a9a7a2"] * len(VARIANTS)
        chosen = list(shown).index(_chain(decision["chosen"]))
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
        ax.set_xticks(x, list(shown.values()))
        ax.tick_params(axis="x", labelsize=7, length=0)
        ax.grid(axis="y", color="#ecebe8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("median error (mm)", color=MUTED)
    axes[0, 0].set_ylim(0, 1.1 * top)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#a9a7a2", label="All shells"),
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
    fig.savefig(out / f"eddy-field-variants-{decision['protocol']}-methods.png", dpi=200)
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
    (args.out / f"eddy-field-variants-{args.protocol}-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    tables(decision, args.out)
    figure(decision, args.out)
    for variant, c in decision["against_test7"].items():
        print(f"{variant} against Test 7's: {_interval(c)} ({c['verdict']})")
    print(f"goes forward: {decision['chosen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
