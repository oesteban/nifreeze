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
"""Post hoc on Test 5 of #735: how each round's errors follow the b-vector.

For every volume of a round, the error of its estimate (in the frame of the round's
targets, as ``eddy_pass_rounds.py`` scores it) is split as the pass stores it: the head
motion's rotation (a rotation vector, degrees) and the field's three slopes (1e-3 per mm).
Within each shell, the table gives:

- the root mean square of the error;
- the size of the part that every volume of the shell shares (its mean);
- the share of the rest's variance that a linear fit on the b-vector explains (eddy
  currents follow the gradient linearly, and change sign with it);
- the share that a quadratic fit explains (the same for ``g`` and ``-g``, as the diffusion
  signal is).

By chance alone, with 64 volumes a shell, the fits would explain about 3/63 (linear) and
6/63 (quadratic) of the variance. Round 1 is Test 4's delayed pass, with the shared field
removed for the rounds that remove it::

    uv run tools/registration/eddy_pass_rounds_shares.py --dataset <phantoms> \\
        --results <results> --test4 <eddy-parts>/arms --out <out>
"""

import argparse
import json
from pathlib import Path

import eddy_pass_parts as parts
import eddy_pass_rounds as rounds
import numpy as np
from eddy_pass_parts_figures import _write
from eddy_targets import NOISE, _common
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

SCORES = ("rotation", "field")
"""The parts of each error, as the table names them."""


def shares(values: np.ndarray, directions: np.ndarray) -> tuple[float, float, float, float]:
    """The RMS, the shared part's size, and the linear and quadratic fits' shares."""
    ones = np.ones((len(values), 1))
    linear = np.column_stack([ones, directions])
    x, y, z = directions.T
    quadratic = np.column_stack([ones, x * x, y * y, z * z, x * y, x * z, y * z])
    mean = values.mean(axis=0)
    total = ((values - mean) ** 2).sum()
    found = []
    for design in (linear, quadratic):
        fit = design @ np.linalg.lstsq(design, values, rcond=None)[0]
        found.append(float(((fit - mean) ** 2).sum() / total))
    rms = float(np.sqrt((values**2).sum(axis=1).mean()))
    return rms, float(np.linalg.norm(mean)), found[0], found[1]


def errors(
    estimates: np.ndarray, frame: np.ndarray, whole: np.ndarray, u: np.ndarray
) -> dict[str, np.ndarray]:
    """Each volume's error, split as the pass stores it."""
    rotation, slopes = [], []
    for estimate, truth in zip(estimates, whole, strict=True):
        offset = estimate @ np.linalg.inv(frame) @ np.linalg.inv(truth)
        motion, field = split_phase_encoding(offset, u)
        rotation.append(Rotation.from_matrix(motion[:3, :3]).as_rotvec(degrees=True))
        slopes.append(1e3 * (u @ (np.eye(3) - field[:3, :3])))
    return {"rotation": np.asarray(rotation), "field": np.asarray(slopes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--results", type=Path, required=True, help="Test 5's results folder")
    parser.add_argument("--test4", type=Path, required=True, help="Test 4's arms/ folder")
    parser.add_argument("--out", type=Path, required=True, help="where the table goes")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    # ``eddy_pass_rounds._path`` reads Test 5's arms from ``out``
    paths = argparse.Namespace(protocol=args.protocol, test4=args.test4, out=args.results)
    meta = json.loads((args.results / f"{args.protocol}.json").read_text())

    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(reference)
    distorted, _ = parts._choose(reference, cells, "e2")
    pe = "ijk"[distorted.distortion.axis]
    u = phase_encoding_vector(pe, reference.affine)
    gradients = np.asarray(reference.gradients)
    directions = gradients[:, :3] / np.linalg.norm(gradients[:, :3], axis=1, keepdims=True)
    shells = np.round(gradients[:, 3], -2)

    rows = []
    for name in meta["Copies"]:
        perturbation, load = parts._choose(reference, cells, name)
        truth = parts._truth(perturbation, n_volumes)
        cascade = np.load(rounds._path(paths, name, "cascade"))[f"cascade/{rounds.CASCADE[-1]}"]
        first = np.asarray(np.load(rounds._path(paths, name, "round1-i9"))["estimates"])
        data = load()
        for kind in meta["Kinds"]:
            found = {
                1: rounds._removed(first, data, pe, list(range(n_volumes)))
                if kind == "removed"
                else first
            }
            for k in range(2, int(meta["Rounds"]) + 1):
                arm = rounds._path(paths, name, f"{kind}{k}-i9")
                found[k] = np.asarray(np.load(arm)["estimates"])
            for k, estimates in found.items():
                frame = (
                    _common(cascade, truth["motion"])
                    if k == 1
                    else _common(found[k - 1], truth["whole"])
                )
                split = errors(estimates, frame, truth["whole"], u)
                for shell in np.unique(shells[shells > 50]):
                    chosen = shells == shell
                    for score in SCORES:
                        rms, shared, linear, quadratic = shares(
                            split[score][chosen], directions[chosen]
                        )
                        rows.append(
                            [
                                name,
                                kind,
                                k,
                                int(shell),
                                score,
                                f"{rms:.3f}",
                                f"{shared:.3f}",
                                f"{100 * linear:.0f}%",
                                f"{100 * quadratic:.0f}%",
                            ]
                        )
        del data
    _write(
        args.out / f"eddy-rounds-{args.protocol}-shares.csv",
        ["Copy", "Kind", "Round", "Shell", "Score", "RMS", "Shared", "Linear", "Quadratic"],
        rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
