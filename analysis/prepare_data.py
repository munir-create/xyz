"""
Step 0 - data preparation.

* reads the corrected 7/28-day data set (the controlling data set),
* writes the long (60-row) format used by mixed-model software,
* documents every difference from the run sheet used in the earlier 7-day model,
* writes descriptive statistics (replicate / pure error, specimen scatter).

Run:  python analysis/prepare_data.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import design as dz

OUT_T = os.path.join(dz.ROOT, "results", "tables")
os.makedirs(OUT_T, exist_ok=True)


def main():
    d = dz.load()

    # ---- long format
    long = []
    for _, r in d.iterrows():
        for age in (7, 28):
            long.append({"mix": r.mix, "age_d": age, "carbonation": r.carbonation, "carb_h": r.carb_h,
                         "RCF_pct": r.RCF_pct, "SS_pct": r.SS_pct, "AB": r.AB,
                         "strength_MPa": r[f"f{age}_mean"], "sd_MPa": r[f"f{age}_sd"],
                         "n_specimens": r[f"f{age}_n"], "replicate_group": r.replicate_group})
    long = pd.DataFrame(long)
    long.to_csv(os.path.join(dz.ROOT, "data", "strength_long.csv"), index=False)

    # ---- differences from the previous 7-day run sheet
    old = pd.read_csv(os.path.join(dz.ROOT, "data", "previous_7d_run_sheet.csv"))
    diffs = []
    for _, o in old.iterrows():
        n = d[d.mix == o.mix].iloc[0]
        for col_old, col_new, nm in [("carb_h", "carb_h", "carbonation (h)"), ("RCF_pct", "RCF_pct", "RCF (%)"),
                                     ("SS_pct", "SS_pct", "SS (%)"), ("AB", "AB", "A/B"),
                                     ("f7_mean", "f7_mean", "7-day strength (MPa)")]:
            if not np.isclose(o[col_old], n[col_new], atol=1e-9):
                diffs.append({"mix": int(o.mix), "variable": nm, "previous": o[col_old], "corrected": n[col_new],
                              "change": n[col_new] - o[col_old]})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(os.path.join(OUT_T, "T00_changes_vs_previous_7d_data.csv"), index=False)

    # ---- pure error (replicate batches of the same nominal design point)
    g = d.dropna(subset=["replicate_group"])
    pe = {}
    for c, lab in [("f7_mean", "7 d MPa"), ("f28_mean", "28 d MPa"), ("l7", "7 d ln"), ("l28", "28 d ln"),
                   ("M", "level M"), ("G", "gain G")]:
        ss = sum(((gg[c] - gg[c].mean()) ** 2).sum() for _, gg in g.groupby("replicate_group"))
        dfp = sum(len(gg) - 1 for _, gg in g.groupby("replicate_group"))
        pe[lab] = {"sd": float(np.sqrt(ss / dfp)), "df": int(dfp)}
    r = pd.concat([gg[["l7", "l28"]] - gg[["l7", "l28"]].mean() for _, gg in g.groupby("replicate_group")])
    S = (r.T @ r / pe["7 d ln"]["df"]).values
    pe["ln covariance 7/28"] = {"var7": S[0, 0], "var28": S[1, 1], "cov": S[0, 1],
                                "rho": S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])}

    # ---- specimen (within-batch) scatter, n assumed 3 unless stated n = 1
    spec = {}
    for a in ("7", "28"):
        m, s, n = d[f"f{a}_mean"], d[f"f{a}_sd"], d[f"f{a}_n"]
        ok = s.notna() & (n > 1)
        cv = (s / m)[ok]
        spec[a] = {"pooled_cv": float(np.sqrt(np.sum((n[ok] - 1) * cv ** 2) / np.sum(n[ok] - 1))),
                   "pooled_sd_MPa": float(np.sqrt(np.sum((n[ok] - 1) * s[ok] ** 2) / np.sum(n[ok] - 1))),
                   "n_single": int((n == 1).sum())}

    desc = d[["f7_mean", "f28_mean"]].describe().T
    desc["gain_ratio_mean"] = np.nan
    gain = d.f28_mean / d.f7_mean
    summary = {
        "n_mixes": int(len(d)), "n_obs": int(2 * len(d)),
        "n_design_points": int(d.group.nunique()),
        "replicate_groups": {k: list(map(int, v)) for k, v in g.groupby("replicate_group").mix.apply(list).items()},
        "carbonation_counts": d.carbonation.value_counts().reindex(dz.CARB_LEVELS).astype(int).to_dict(),
        "carbonation_distinct_points": d.groupby("carbonation", observed=True).group.nunique().reindex(dz.CARB_LEVELS).astype(int).to_dict(),
        "f7": {"min": d.f7_mean.min(), "max": d.f7_mean.max(), "mean": d.f7_mean.mean(), "sd": d.f7_mean.std()},
        "f28": {"min": d.f28_mean.min(), "max": d.f28_mean.max(), "mean": d.f28_mean.mean(), "sd": d.f28_mean.std()},
        "gain_ratio": {"min": gain.min(), "max": gain.max(), "median": gain.median(), "geo_mean": float(np.exp(np.log(gain).mean()))},
        "corr_ln7_ln28": float(np.corrcoef(d.l7, d.l28)[0, 1]),
        "pure_error": pe, "specimen_scatter": spec,
        "n_changes": int(len(diffs)),
    }
    with open(os.path.join(dz.ROOT, "results", "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"{len(d)} mixes, {len(diffs)} differences from the previous 7-day run sheet")
    print(diffs.to_string(index=False))
    print(json.dumps(pe, indent=1, default=float))


if __name__ == "__main__":
    main()
