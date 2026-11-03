"""Stage 2: regions and array shapes for all 3 age groups (exemplar-level, correlation distance)."""
import pickle, glob, numpy as np, warnings
warnings.filterwarnings("ignore")
for g in ["adult", "ninemonth", "twomonth"]:
    p = glob.glob(f"data/brain_rdms/*{g}_exemplar*.pickle")[0]
    d = pickle.load(open(p, "rb"))
    print(f"== {g}: {len(d)} regions: {list(d.keys())}")
    k = list(d.keys())[0]
    print("   shapes in", k, {kk: v.shape for kk, v in d[k].items()})
    a = d[k]["acrosssub"]
    print("   acrosssub: diag mean %.3f, offdiag mean %.3f, nan count %d, symmetric=%s" % (
        np.nanmean(np.diagonal(a, axis1=1, axis2=2)), np.nanmean(a[:, ~np.eye(36, dtype=bool)]),
        int(np.isnan(a).sum()), bool(np.allclose(a, a.transpose(0, 2, 1), equal_nan=True))))
