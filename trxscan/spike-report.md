## TRXScan spike: results

TRXScan main `680fc4f4` (mrsim-acq `63fb68a`), built from source with maturin through uv on
macOS arm64 (Apple M4, 24 GB), Python 3.12; NiFreeze as a path source. Phantom sub-60501,
subsampled to 100k (checks) or 400k (timing) streamlines.

| # | Check | Result |
|---|---|---|
| S0 | Build | 34 s; TRXScan's suite passes (72 passed, 1 skipped) |
| S1 | Clean configuration | phase exactly 0: magnitude storage, noise added by us |
| S2 | Repeats, 1 vs 10 threads, chunked vs whole (static and motion) | bit-identical |
| S3 | Motion path with zero poses vs static path | differ by 9 to 13% in DW volumes (GM one ball vs two); unmoved volumes of a motion run equal the zero-pose run bit for bit |
| S4 | Gradient-table subsets (static, motion, eddy; with or without max b) | bit-identical to the full run |
| S5 | Eddy geometry | TRXScan's output reproduced exactly (max rel. error 1.6e-7) by `shift_j = −TRT·eddy·b·(bvec·c)`, c centred at (nx/2, ny/2, nz/2), b-vector components applied to the grid axes as given (RAS, LPS, oblique identical) |
| S5 | Coefficient recovery | NiFreeze's first-order spline fit is biased 3 to 70% (11 to 57% at 1 voxel); an exact EPI operator with Gauss-Newton recovers them within 1e-5 up to 2 voxels |
| S6 | Pose convention | world RAS, rotation Rz·Ry·Rx about the FOV centre ((n−1)/2); **defect:** the motion path moves CSF and GM by `2·round(t/v)·v − t` for sub-voxel translations. `Phantom.moved(T)` + `grid(like=)` moves everything correctly |
| S7 | Runtime, 400k streamlines, chunked | a fixed 8 to 13 s per call, about 0.05 s per extra volume; one moved volume 12 s at 2 mm, 14 s at 1.25 mm; peak RSS under 2 GB |
| S8 | Noise | TRXScan's variance within 0.1% of target; ours indistinguishable (KS p = 0.17); seeds 4 and 5 identical (mrsim-acq #12) |
| S9 | PE along LR, b-vector frame, T2 | oblique (0, 0, 90)° puts j along LR; written bvecs are the prescribed ones in the grid frame; per-tissue S0 gives WM 26.7, GM 31.2, CSF 94.7 at TE 90 ms |

### Consequences for the plan

1. **Motion through `Phantom.moved`, not the motion path.** The reference comes from the static
   clean path (two-ball GM, truth peaks available); each moved volume is simulated from the
   moved phantom gridded `like=` the reference, and spliced. Splicing is bit-exact.
2. **Eddy ground truth is analytic** (exact). B2's inherited share must fit with the EPI
   operator (Gauss-Newton), not the first-order spline fit.
3. **Noise on load.** Only noise-free volumes are stored; noise keyed on (seed, volume) is added
   when a cell is loaded (one noisy reference per protocol is stored for browsing).
4. **Gradients** for NiFreeze come from the prescribed world table, not the written bvec.
5. **Budget** (4 motion magnitudes × N moved volumes, each with 0 + 4 eddy magnitudes, plus
   4 eddy-only runs; seed 0): HCP-YA 17 h, Sherbrooke 9 h, Stanford 7 h, HBN 6 h, ABCD 5 h,
   AOMIC 2 h; about 47 h sequential. Storage of noise-free changed volumes, gzipped float32:
   about 40 GB in all, about half of it HCP-YA.
