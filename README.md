# Experiment record: TRXScan spike and the Sherbrooke sensitivity runs

This branch is a record, not code to merge. It keeps the scripts and results of
two experiments behind nipreps/nifreeze#548 (synthetic reference data) and #545
(inherited distortion under motion), which were first run in a temporary folder.
Paths inside the scripts point to that folder (`/private/tmp/...`), and they run
with the NiFreeze checkout of their time; they are kept as run, for the record.

## `trxscan/`: the TRXScan spike (2026-10-22)

Checks that PennLINC's TRXScan (main `680fc4f4`) can simulate NiFreeze's
reference data and its perturbations. Results: `spike-report.md` (also posted on
#548); findings for TRXScan's developers: `upstream-findings.md`.

- `pyproject.toml`, `uv.lock`: the environment (`uv sync --python 3.12`): TRXScan
  built from that commit, and NiFreeze from a local checkout (edit the path).
- `common.py`: the clean protocol and small gradient tables shared by the checks.
- `s2_s4.py`: determinism, the zero-pose motion path, and gradient-table subsets
  (results in `out/s2_s4.json`).
- `s5*.py`, `epi.py`: the eddy-current geometry. `s5c.py`/`s5e.py` reproduce
  TRXScan's eddy-current images from the clean ones with the closed-form field,
  `shift_j = -TRT * eddy * b * (bvec . c)` (max relative error 1.6e-7);
  `epi.py` and `s5h.py` fit its coefficients with that operator (within 1e-5).
  `s5.py`, `s5b.py` are the first, flawed attempts (centroids and half-maximum
  edges of sub-voxel Fourier shifts), kept to show why.
- `s6*.py`, `repro_motion_translation.py`: the pose convention, and the defect of
  TRXScan's motion path (sub-voxel translations move CSF and GM by
  `2 * round(t / v) * v - t`); `s6e.py` checks the `Phantom.moved` workaround.
- `s7.py`, `s7.log`: runtime and memory at 2 and 1.25 mm.
- `s8.py`, `s9.py`: noise equivalence; phase encoding along LR, the frame of
  written b-vectors, and T2 contrast through per-tissue S0.

## `sherbrooke/`: sensitivity runs on DIPY's Sherbrooke data (2026-10-22)

- `eddy_real.py`: measures the real eddy currents of Sherbrooke against the shell
  median (in-plane shear of 0.65 mm at the brain's edge per unit readout
  gradient), the finding that made it unfit as an aligned reference (#548).
- `check_inherit.py`, `summarize_inherit.py` (`out-inherit/`): the inherited
  share of a moved and distorted copy, split into what the motion alone
  projects onto the distortion's terms and the rest (#545).
- `run_inherit_fixed.py`, `summarize_inherit_fixed.py` (`out-inherit-fixed/`):
  the share measured with the fix of #547, on every count and magnitude of rung 3.
- `out-rung3-k{16,96,192}/`: rung 3 cells (rigid motion of K volumes with eddy
  currents) for three of the four counts.

The scripts of rungs 1 to 3 themselves (2026-10-21), and the K=64 cells, were
lost when the temporary folder was purged; their results survive in the bodies
of #542, #544, #545 and #547. Rungs 1 to 3 are to be rerun on the synthetic data
(#549).
