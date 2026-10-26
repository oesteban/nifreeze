"""S2 determinism; S3 zero-pose motion path vs clean path; S4 gradient-table subsets."""
import numpy as np
import trxscan as ts
from common import *

g = table()
n = len(g.bvals)
obj = phantom(100_000)
P, A = clean_protocol(4.0), clean_artifacts()
res = {}

def diff(a, b):
    a, b = np.abs(a), np.abs(b)
    return {"identical": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a - b).max()),
            "max_rel": float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))}

base, _, sec, _ = run(obj, g, P, A)
res["seconds_clean"] = sec
again, *_ = run(obj, g, P, A)
res["S2_repeat"] = diff(again, base)
ts.set_threads(1); one, _, sec1, _ = run(obj, g, P, A); ts.set_threads(10)
res["S2_threads_1_vs_10"] = diff(one, base); res["seconds_1thread"] = sec1
chunked, *_ = run(obj, g, P, A, chunk=8)
res["S2_chunked_vs_whole"] = diff(chunked, base)

zero = ts.Motion.from_arrays(np.zeros((n, 3)), np.zeros((n, 3)))
zpath, _, secm, _ = run(obj, g, P, A.replace(motion=zero), context=8)
res["seconds_motion_path"] = secm
res["S3_zero_pose_vs_clean"] = diff(zpath, base)
res["S3_zero_pose_vs_clean_per_volume_max_rel"] = [diff(zpath[..., i], base[..., i])["max_rel"] for i in range(n)]

j = 3
poses = np.zeros((n, 6)); poses[j] = [2.0, -1.0, 1.5, 2.0, -1.0, 1.5]
moved, *_ = run(obj, g, P, A.replace(motion=ts.Motion.from_arrays(poses, None)), context=8)
others = [i for i in range(n) if i != j]
res["S3_unmoved_volumes_vs_zero_pose"] = diff(moved[..., others], zpath[..., others])
res["S3_moved_volume_changed_rel"] = diff(moved[..., j], zpath[..., j])["max_rel"]

chunk_moved, *_ = run(obj, g, P, A.replace(motion=ts.Motion.from_arrays(poses, None)), chunk=8, context=8)
res["S2_chunked_motion_vs_whole"] = diff(chunk_moved, moved)

idx = [0, j, n - 1]  # b=0, the moved volume, a max-b volume
sub, *_ = run(obj, subset(g, idx), P, A.replace(motion=ts.Motion.from_arrays(poses[idx], None)), context=8)
res["S4_subset_motion_vs_full"] = diff(sub, moved[..., idx])
sub_clean, *_ = run(obj, subset(g, idx), P, A.replace(motion=ts.Motion.from_arrays(np.zeros((3, 6)), None)), context=8)
res["S4_subset_zero_pose_vs_full"] = diff(sub_clean, zpath[..., idx])
eddy = A.replace(eddy=2e-4, eddy_tau_ms=1e12)
full_e, *_ = run(obj, g, P, eddy)
sub_e, *_ = run(obj, subset(g, idx), P, eddy)
res["S4_subset_eddy_vs_full"] = diff(sub_e, full_e[..., idx])
sub_e2, *_ = run(obj, subset(g, [0, j]), P, eddy)
res["S4_subset_eddy_without_max_b"] = diff(sub_e2, full_e[..., [0, j]])
res["peak_rss_gb"] = peak_rss_gb()
record("s2_s4", res)
