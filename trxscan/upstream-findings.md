# TRXScan findings from the NiFreeze spike

TRXScan main `680fc4f4` (2026-10-19), mrsim-acq `63fb68a`, built from source on macOS arm64,
Python 3.12. TRXScan's own suite passes on this build (72 passed, 1 skipped).

## 1. Motion path: sub-voxel translations move CSF and GM the wrong way

`Artifacts(motion=Motion.from_arrays(...))`, translation-only poses, b=0 volumes. The centre
of mass of a CSF-only b=0 (`tissue_s0=(0, 0, 1)`) moves by `2·round(t/v)·v − t` instead of `t`
(`v` = voxel size): the fraction of a voxel is applied with the wrong sign. Whole-voxel
translations are right, which is why 2 mm runs with 2/4/8 mm translations look fine.

| voxel | translation (mm) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| 3 mm | CSF-only b=0, COM shift | 0 | −2.90 | 5.94 | 3.00 | 0.10 | 8.94 | 6.00 |
| 3 mm | GM-only b=0 | 0 | 0.73 | 2.26 | 3.00 | 3.73 | 5.26 | 6.00 |
| 3 mm | WM-only b=0 (streamlines) | 0 | 0.99 | 2.01 | 3.00 | 3.99 | 5.01 | 6.00 |
| 3 mm | `Phantom.moved(T)` + `grid(like=)`, CSF only | 0 | 1.04 | 1.85 | 3.00 | 4.04 | 4.85 | 6.00 |

The same rule holds at 2.5 and 4 mm. Rotations about the FOV centre are close to the expected
centre-of-mass shifts. `motion::resample_by_pose` and `trilinear` read correctly in isolation, so
the error is probably in how the moved GM/CSF maps (or the moved mask) reach the signal stage of
`generate_compartments_moving`. Reproduction: `repro_motion_translation.py` (attached).

## 2. Noise seeds 2k and 2k+1 give identical noise

Already open as mrsim-acq #12; confirmed on this build (see S8).

## 3. Eddy currents: frame, centre and scaling (documentation)

Verified by reproducing the k-space output exactly (max relative error 1.6e-7) with
`shift_j = −TRT · eddy · b · (bvec_x·(i − nx/2) + bvec_y·(j − ny/2) + bvec_z·(k − nz/2))`
(`eddy_tau_ms` very large):
- The b-vector components are applied to the grid axes as given, whatever the affine (RAS,
  LPS and oblique grids give identical shifts). For a grid that is not RAS-aligned, the field
  no longer follows the gradient's direction in space.
- The slice term is centred at `nz/2` in practice, although the comment near the eddy
  object-phase ramp says `zc` is "slice-index-from-start, not centred".
- The amplitude scales with `b` (`g = bvec·bval`), whereas a gradient's eddy field scales with
  its amplitude (√b at fixed timing).
- No eddy ground truth is written; a per-volume field or displacement output would help.

## 4. Release request

The motion fixes on main (chunked slabs rotating about their own centre in 0.3.0) are not on
PyPI; a 0.4 release would let downstream projects pin a wheel instead of building from source.
