"""
Comparison of the paired model fitted to the current corrected data set with the same
analysis fitted to the previous corrected data set (v1).

  python analysis/compare_previous.py --extract OLD_model_results.json
        one-off: stores the key quantities of the earlier run in results/previous_v1/key_results.json
  python analysis/compare_previous.py
        writes results/tables/T24_comparison_with_previous_data.csv and results/comparison_previous.json

Both runs use the identical protocol (analysis/run_analysis.py); only the data differ
(see results/tables/T00b_changes_vs_previous_corrected_data.csv).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import design as dz

RES = os.path.join(dz.ROOT, "results")
PREV = os.path.join(RES, "previous_v1", "key_results.json")


def key_quantities(R):
    """Key numbers of one run of run_analysis.py (works on any model_results.json)."""
    fs, val, vc = R["final"]["fit_stats"], R["validation"], R["final"]["variance_components"]
    gc = pd.DataFrame(R["gain_curve"]).set_index("SS")
    ncv = pd.DataFrame(R["nested_cv"])
    ncm = ncv[ncv.cv == "leave-one-mixture-out"].set_index("procedure")
    q = {"final_terms": R["final_terms"], "final_struct": R["final_struct"],
         "AICc": R["final"]["AICc"], "AICc_best_CS": R["search"]["CS"]["top"][0]["AICc"],
         "sd7": vc["sd_7"], "sd28": vc["sd_28"], "rho": vc["rho"],
         "pure_error_sd_ln_7": R["data"]["pure_error"]["7 d ln"]["sd"], "pure_error_sd_ln_28": R["data"]["pure_error"]["28 d ln"]["sd"],
         "pure_error_sd_MPa_7": R["data"]["pure_error"]["7 d MPa"]["sd"], "pure_error_sd_MPa_28": R["data"]["pure_error"]["28 d MPa"]["sd"],
         "R2_7": fs["7"]["R2"], "R2_28": fs["28"]["R2"],
         "predR2_7": val["mix"]["predR2_7"], "predR2_28": val["mix"]["predR2_28"],
         "rmse_pred_MPa_7": val["mix"]["rmse_MPa_7"], "rmse_pred_MPa_28": val["mix"]["rmse_MPa_28"],
         "cv_resid_pct_7": fs["7"]["cv_pct"], "cv_resid_pct_28": fs["28"]["cv_pct"],
         "lof_p_7": R["final"]["lack_of_fit"]["7 d (per-age OLS, same terms)"]["p"],
         "lof_p_28": R["final"]["lack_of_fit"]["28 d (per-age OLS, same terms)"]["p"],
         "ncv_primary_predR2_7": float(ncm.loc["AICc overall (primary)", "predR2_7"]),
         "ncv_primary_predR2_28": float(ncm.loc["AICc overall (primary)", "predR2_28"]),
         "boxcox_lambda": R["boxcox"]["final_UN"]["lambda"], "boxcox_ci": R["boxcox"]["final_UN"]["ci"],
         "gain_SS0": float(gc.loc[0.0, "ratio_est"]), "gain_SS37.5": float(gc.loc[37.5, "ratio_est"]), "gain_SS75": float(gc.loc[75.0, "ratio_est"]),
         "gain_min_SS": R["contrasts"].get("gain_min_SS", {}).get("SS"),
         "p_duration_levels_equal": R["coding"]["duration_equal_test"]["p"],
         "resid_boot_exact_final": R["bootstrap"]["resid"]["exact_final"],
         "term_p": {r["term"]: r["p"] for r in R["final"]["terms_F"]},
         "coef": {r["column"]: {"est": r["coef"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]} for r in R["final"]["coef"]},
         "optimum": {f"{o['age']} d|{o['carbonation']}": {"RCF": o["RCF"], "SS": o["SS"], "median": o["median"]} for o in R["optimum"]},
         "age_change_p": {r["column"]: r["p_change"] for r in R["age_specific"]["table"]}}
    ce = pd.DataFrame(R["carb_effect"])
    r1 = ce[(ce.carbonation == "1 h") & (ce.RCF == 50) & (ce.SS == 0) & (ce.age == 7)].iloc[0]
    r2 = ce[(ce.carbonation == "1 h") & (ce.RCF == 10) & (ce.SS == 75) & (ce.age == 7)].iloc[0]
    q["carb1h_vs_NC_RCF50_SS0"] = float(r1.ratio); q["carb1h_vs_NC_RCF10_SS75"] = float(r2.ratio)
    return q


ROWS = [  # (label, key, format)
    ("Selected terms", "final_terms", "terms"), ("Within-mixture covariance", "final_struct", "s"),
    ("Box–Cox λ (selected model)", "boxcox_lambda", "2"),
    ("AICc of the final model", "AICc", "2"), ("AICc of the best CS model", "AICc_best_CS", "2"),
    ("Replicate (pure-error) SD, 7 d (MPa)", "pure_error_sd_MPa_7", "1"), ("Replicate (pure-error) SD, 28 d (MPa)", "pure_error_sd_MPa_28", "1"),
    ("Residual SD 7 d (ln)", "sd7", "3"), ("Residual SD 28 d (ln)", "sd28", "3"), ("7/28-day correlation within mixture", "rho", "2"),
    ("R² 7 d", "R2_7", "2"), ("R² 28 d", "R2_28", "2"),
    ("Predicted R², leave one mixture out, 7 d", "predR2_7", "2"), ("Predicted R², leave one mixture out, 28 d", "predR2_28", "2"),
    ("RMSE of a left-out mixture, 7 d (MPa)", "rmse_pred_MPa_7", "1"), ("RMSE of a left-out mixture, 28 d (MPa)", "rmse_pred_MPa_28", "1"),
    ("Nested CV pred. R² of the selection procedure, 7 d", "ncv_primary_predR2_7", "2"),
    ("Nested CV pred. R² of the selection procedure, 28 d", "ncv_primary_predR2_28", "2"),
    ("Lack of fit p, 7 d", "lof_p_7", "p"), ("Lack of fit p, 28 d", "lof_p_28", "p"),
    ("Gain f28/f7 at SS 0 %", "gain_SS0", "2"), ("Gain f28/f7 at SS 37.5 %", "gain_SS37.5", "2"), ("Gain f28/f7 at SS 75 %", "gain_SS75", "2"),
    ("SS of minimum gain (%)", "gain_min_SS", "0"),
    ("1 h / NC at RCF 50 %, SS 0 %", "carb1h_vs_NC_RCF50_SS0", "2"), ("1 h / NC at RCF 10 %, SS 75 %", "carb1h_vs_NC_RCF10_SS75", "2"),
    ("p, carbonation levels 0.5 / 1 / 5 h equal", "p_duration_levels_equal", "p"),
    ("Exact final model re-selected, residual bootstrap", "resid_boot_exact_final", "pct"),
]


def fmt(v, f):
    if v is None:
        return "–"
    if f == "terms":
        return " ".join(v)
    if f == "s":
        return str(v)
    if f == "p":
        return "< 0.001" if v < 0.001 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
    if f == "pct":
        return f"{100 * v:.0f} %"
    return f"{v:.{int(f)}f}"


def compare(prev, cur):
    rows = []
    for lab, k, f in ROWS:
        rows.append({"quantity": lab, "previous": fmt(prev.get(k), f), "current": fmt(cur.get(k), f), "group": "model"})
    for t in cur["term_p"]:
        rows.append({"quantity": f"p, {dz.label(t)}", "previous": fmt(prev["term_p"].get(t), "p"), "current": fmt(cur["term_p"][t], "p"),
                     "group": "term"})
    for c, v in cur["coef"].items():
        pv = prev["coef"].get(c)
        rows.append({"quantity": f"coefficient {c}", "previous": "–" if pv is None else f"{pv['est']:.4f} [{pv['lo']:.3f}, {pv['hi']:.3f}]",
                     "current": f"{v['est']:.4f} [{v['lo']:.3f}, {v['hi']:.3f}]", "group": "coef"})
    for k, v in cur["optimum"].items():
        pv = prev["optimum"].get(k)
        rows.append({"quantity": f"optimum {k}", "previous": "–" if pv is None else f"{pv['median']:.1f} MPa at RCF {pv['RCF']:.0f}, SS {pv['SS']:.1f}",
                     "current": f"{v['median']:.1f} MPa at RCF {v['RCF']:.0f}, SS {v['SS']:.1f}", "group": "optimum"})
    for c, v in cur["age_change_p"].items():
        rows.append({"quantity": f"p, change 7→28 d of {c}", "previous": fmt(prev["age_change_p"].get(c), "p"), "current": fmt(v, "p"),
                     "group": "age"})
    return pd.DataFrame(rows)


def main(extract=None):
    if extract:
        with open(extract) as f:
            q = key_quantities(json.load(f))
        os.makedirs(os.path.dirname(PREV), exist_ok=True)
        with open(PREV, "w") as f:
            json.dump(q, f, indent=1)
        print("wrote", PREV)
        return
    with open(PREV) as f:
        prev = json.load(f)
    with open(os.path.join(RES, "model_results.json")) as f:
        cur = key_quantities(json.load(f))
    tab = compare(prev, cur)
    tab.to_csv(os.path.join(RES, "tables", "T24_comparison_with_previous_data.csv"), index=False)
    with open(os.path.join(RES, "comparison_previous.json"), "w") as f:
        json.dump({"previous": prev, "current": cur, "same_terms": set(prev["final_terms"]) == set(cur["final_terms"])
                   and prev["final_struct"] == cur["final_struct"]}, f, indent=1)
    print(tab[tab.group == "model"].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default=None)
    main(ap.parse_args().extract)
