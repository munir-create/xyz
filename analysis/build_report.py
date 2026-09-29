"""
Builds the reports from results/model_results.json:

  report/index.html          interactive report (published as the Artifact)
  report/summary.html        printable summary -> report/Model_Summary.pdf (headless Chromium)
  report/MODEL_REPORT.md     manuscript-style methods and results

Run:  python analysis/build_report.py
"""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

import design as dz
import report_text as rt
from report_text import fp, f1, f2, f3, minus

REP = os.path.join(dz.ROOT, "report")
os.makedirs(REP, exist_ok=True)
E = html.escape

TERM_TXT = {"A": "RCF", "C": "A/B", "D": "SS", "Carb": "Carbonation (4 levels)", "AD": "RCF × SS", "D2": "SS²",
            "KA": "Carbonated × RCF", "KD": "Carbonated × SS", "Age": "Age (28 vs 7 d)", "D.Age": "SS × Age",
            "D2.Age": "SS² × Age", "A2": "RCF²", "C2": "(A/B)²", "AC": "RCF × A/B", "CD": "A/B × SS", "KC": "Carbonated × A/B"}
COL_TXT = {"Intercept": "Intercept", "A": "RCF (A)", "D": "SS (D)", "Carb[0.5 h]": "Carb 0.5 h", "Carb[1 h]": "Carb 1 h",
           "Carb[5 h]": "Carb 5 h", "AD": "RCF·SS (AD)", "D2": "SS² (D²)", "KA": "K·RCF (KA)", "KD": "K·SS (KD)",
           "Age": "Age", "D.Age": "SS × Age", "D2.Age": "SS² × Age"}


def tlabel(t):
    if t.endswith(".Age") and t[:-4] in TERM_TXT:
        return TERM_TXT[t[:-4]] + " × Age"
    return TERM_TXT.get(t, t)


def table(df, cols, heads, fmt=None, cls="", num=None, caption=None):
    fmt = fmt or {}
    num = set(num or [])
    out = [f'<div class="tw"><table class="{cls}">']
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead><tr>" + "".join(f'<th class="{"n" if c in num else ""}">{h}</th>' for c, h in zip(cols, heads)) + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            s = fmt[c](v) if c in fmt else ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
            tds.append(f'<td class="{"n" if c in num else ""}">{s}</td>')
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def md_table(df, cols, heads, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def eq_html(coefs, lhs):
    parts = []
    for c in coefs:
        v = c["coef"]
        term = "" if c["term"] == "1" else "·" + c["term"]
        if abs(v) >= 1e-3:
            mag = f"{abs(v):.6g}"
        else:
            mant, ex_ = f"{abs(v):.5e}".split("e")
            mag = f"{float(mant):.5g}×10<sup>−{int(ex_[1:])}</sup>"
        s = f"{mag}{term}"
        parts.append(("−" if v < 0 else "") + s if not parts else (" − " if v < 0 else " + ") + s)
    return f"{lhs} = " + "".join(parts)


def main():
    R, d, final, m = rt.load()
    ex = rt.extras(R, d, final, m)
    K = rt.key_numbers(R, ex)
    D = R["data"]
    ev = ex["evidence"]
    names = R["final"]["names"]

    # ------------------------------------------------------------ data for the page scripts
    diag = pd.DataFrame(R["diagnostics"]["table"])
    runs = [{"mix": int(r.mix), "carb": r.carbonation, "RCF": r.RCF, "SS": r.SS, "AB": r.AB, "f7": r.obs7, "f28": r.obs28,
             "fit7": r.fit7, "fit28": r.fit28, "cv7": r.cv7, "cv28": r.cv28} for _, r in diag.iterrows()]
    gcurve = pd.DataFrame(R["gain_curve"])
    page_data = {"names": names, "beta": R["final"]["beta"], "V": R["final"]["cov_beta"],
                 "s2": [R["final"]["Sigma"][0][0], R["final"]["Sigma"][1][1]],
                 "t": {"7": ex["t_age"][7], "28": ex["t_age"][28], "gain": ex["t_gain"]},
                 "runs": runs,
                 "gain": {"SS": gcurve.SS.tolist(), "est": gcurve.ratio_est.tolist(), "lo": gcurve.ratio_lo.tolist(),
                          "hi": gcurve.ratio_hi.tolist(), "blo": gcurve.boot_case_lo.tolist(), "bhi": gcurve.boot_case_hi.tolist()},
                 "optimum": [{k: o[k] for k in ("age", "carbonation", "RCF", "SS", "median", "ci", "pi")} for o in R["optimum"]],
                 "val": {k: R["validation"][k] for k in ("fit", "mix")}}
    page_json = json.dumps(page_data, separators=(",", ":"))

    # ------------------------------------------------------------ tables
    coef = pd.DataFrame(R["final"]["coef"])
    ct = pd.read_csv(os.path.join(dz.ROOT, "results", "tables", "T18_final_coefficients_with_bootstrap_CI.csv"))
    coef["bres"] = [f"{a:.3f} to {b:.3f}" for a, b in zip(ct.boot_resid_lo, ct.boot_resid_hi)]
    coef["bcase"] = [f"{a:.3f} to {b:.3f}" for a, b in zip(ct.boot_case_lo, ct.boot_case_hi)]
    coef["lab"] = coef.column.map(lambda c: COL_TXT.get(c, c))
    coef["ci"] = [f"{a:.3f} to {b:.3f}" for a, b in zip(coef.lo, coef.hi)]
    coef_tab = table(coef, ["lab", "coef", "se", "df", "ci", "p", "bres", "bcase"],
                     ["Term (coded)", "Estimate", "SE", "df", "95 % CI (Wald)", "p", "95 % residual bootstrap", "95 % case bootstrap"],
                     {"coef": lambda v: minus(f"{v:.4f}"), "se": lambda v: f"{v:.4f}", "df": lambda v: f"{v:.1f}",
                      "p": lambda v: rt.pnum(v), "ci": minus, "bres": minus, "bcase": minus},
                     num=["coef", "se", "df", "ci", "p", "bres", "bcase"])
    tf = pd.DataFrame(R["final"]["terms_F"])
    tf["lab"] = tf.term.map(tlabel)
    tf["rb"] = tf.term.map(lambda t: rt.pct(R["bootstrap"]["resid"]["inclusion"].get(t, 1.0)) if t != "Age" else "–")
    tf["sb"] = tf.term.map(lambda t: rt.pct(R["bootstrap"]["subsample"]["inclusion"].get(t, 1.0)) if t != "Age" else "–")
    tf["ev"] = tf.term.map(lambda t: ev.get(t, "–") if t != "Age" else "Design factor")
    tf["dfs"] = [f"{int(a)}, {b:.1f}" for a, b in zip(tf.df1, tf.df2)]
    tf_tab = table(tf, ["lab", "dfs", "F", "p", "rb", "sb", "ev"],
                   ["Term", "df", "F", "p", "Residual bootstrap", "Subsamples 24/30", "Evidence"],
                   {"F": lambda v: f"{v:.2f}", "p": rt.pnum,
                    "ev": lambda v: f'<span class="chip {v.split()[0].lower()}">{v}</span>'},
                   num=["dfs", "F", "p", "rb", "sb"])
    # age-specific
    ag = pd.DataFrame(R["age_specific"]["table"])
    ag["lab"] = ag.column.map(lambda c: COL_TXT.get(c, c).split(" (")[0])
    for a in ("7", "28"):
        ag["e" + a] = [minus(f"{e:.3f} [{lo:.3f}, {hi:.3f}]") for e, lo, hi in zip(ag["eff" + a], ag["lo" + a], ag["hi" + a])]
    ag["chg"] = [minus(f"{e:.3f} [{lo:.3f}, {hi:.3f}]") for e, lo, hi in zip(ag.change, ag.lo_change, ag.hi_change)]
    ag["inf"] = ag.in_final_as_age_term.map(lambda b: "yes" if b else "no (common effect)")
    ag_tab = table(ag, ["lab", "e7", "e28", "chg", "p_change", "inf"],
                   ["Term (coded)", "Effect at 7 d [95 % CI]", "Effect at 28 d [95 % CI]", "Change 7→28 d", "p (change)", "Age term in final model"],
                   {"p_change": rt.pnum}, num=["e7", "e28", "chg", "p_change"])
    # top models
    top = pd.DataFrame(R["search"]["UN"]["top"][:10])
    top["lev"] = top.level_terms.map(lambda s: ", ".join(TERM_TXT.get(t, t) for t in s.split()))
    top["gn"] = top.gain_terms.map(lambda s: ", ".join(TERM_TXT.get(t, t) for t in s.split()) or "none")
    top_tab = table(top, ["rank", "lev", "gn", "k", "AICc", "dAICc", "weight"],
                    ["#", "Terms common to both ages (level)", "Age interactions (gain)", "k", "AICc", "ΔAICc", "Akaike weight"],
                    {"AICc": f2, "dAICc": f2, "weight": f3}, num=["rank", "k", "AICc", "dAICc", "weight"])
    # nested cv
    ncv = pd.DataFrame(R["nested_cv"])
    ncvm = ncv[ncv.cv == "leave-one-mixture-out"].copy()
    ncvg = ncv[ncv.cv != "leave-one-mixture-out"].set_index("procedure")
    ncvm["g7"] = ncvm.procedure.map(lambda p: ncvg.loc[p, "predR2_7"]); ncvm["g28"] = ncvm.procedure.map(lambda p: ncvg.loc[p, "predR2_28"])
    ncvm["stab"] = [("–" if not np.isfinite(a) else f"{int(a)} / {int(b)}") for a, b in zip(ncvm.distinct_models, ncvm.n_folds)]
    ncv_tab = table(ncvm, ["procedure", "predR2_7", "predR2_28", "rmse_MPa_7", "rmse_MPa_28", "g7", "g28", "stab"],
                    ["Procedure (re-run inside every fold)", "Pred R² 7 d", "Pred R² 28 d", "RMSE 7 d (MPa)", "RMSE 28 d (MPa)",
                     "Pred R² 7 d (design point out)", "Pred R² 28 d (design point out)", "Distinct models / folds"],
                    {"predR2_7": lambda v: minus(f2(v)), "predR2_28": lambda v: minus(f2(v)), "rmse_MPa_7": f1, "rmse_MPa_28": f1,
                     "g7": lambda v: minus(f2(v)), "g28": lambda v: minus(f2(v))},
                    num=["predR2_7", "predR2_28", "rmse_MPa_7", "rmse_MPa_28", "g7", "g28", "stab"])
    # robustness
    rb = pd.DataFrame(R["robustness"])
    rb_tab = table(rb, ["analysis", "gain_SS0", "gain_SS75", "1h/NC ratio RCF50 SS0 28d", "1h/NC ratio RCF10 SS75 28d",
                        "RCF50/10 ratio 1h SS0 28d", "predR2_7", "predR2_28"],
                   ["Analysis", "Gain SS 0", "Gain SS 75", "1 h/NC RCF 50 SS 0", "1 h/NC RCF 10 SS 75", "RCF 50/10 (1 h, SS 0)",
                    "Pred R² 7 d", "Pred R² 28 d"],
                   {"analysis": E, "gain_SS0": f2, "gain_SS75": f2, "1h/NC ratio RCF50 SS0 28d": f2, "1h/NC ratio RCF10 SS75 28d": f2,
                    "RCF50/10 ratio 1h SS0 28d": f2,
                    "predR2_7": lambda v: "–" if v is None or not np.isfinite(v) else minus(f2(v)),
                    "predR2_28": lambda v: "–" if v is None or not np.isfinite(v) else minus(f2(v))},
                   num=["gain_SS0", "gain_SS75", "1h/NC ratio RCF50 SS0 28d", "1h/NC ratio RCF10 SS75 28d",
                        "RCF50/10 ratio 1h SS0 28d", "predR2_7", "predR2_28"], cls="small")
    # optimum
    opt = pd.DataFrame(R["optimum"])
    opt["where"] = [f"RCF {a:.0f} %, SS {b:.1f} %" for a, b in zip(opt.RCF, opt.SS)]
    opt["cis"] = [f"{c[0]:.1f}–{c[1]:.1f}" for c in opt.ci]
    opt["pis"] = [f"{c[0]:.1f}–{c[1]:.1f}" for c in opt.pi]
    opt["box"] = [f"RCF {a[0]:.0f}–{a[1]:.0f}, SS {b[0]:.0f}–{b[1]:.0f}" for a, b in zip(opt.near95_RCF, opt.near95_SS)]
    opt["lev"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(opt.spv, opt.spv_design_max)]
    opt["age_s"] = opt.age.map(lambda a: f"{a} d")
    opt_tab = table(opt, ["age_s", "carbonation", "where", "median", "cis", "pis", "box", "lev"],
                    ["Age", "Carbonation", "Maximum at", "Median (MPa)", "95 % CI", "95 % PI (one batch)", "Within 5 % of maximum",
                     "Scaled pred. variance / design max"],
                    {"median": f1}, num=["median", "cis", "pis", "lev"])
    # changes
    ch = pd.read_csv(os.path.join(dz.ROOT, "results", "tables", "T00_changes_vs_previous_7d_data.csv"))
    big = ch[(ch.variable == "7-day strength (MPa)") & (ch.change.abs() > 0.05) | (ch.variable == "SS (%)") & (ch.change.abs() > 0.5) |
             (ch.variable == "A/B") & (ch.change.abs() > 0.005)]
    ch_tab = table(ch, ["mix", "variable", "previous", "corrected", "change"], ["Mix", "Variable", "Previous 7-day file", "Corrected data set", "Change"],
                   {"previous": lambda v: f"{v:g}", "corrected": lambda v: f"{v:g}", "change": lambda v: minus(f"{v:+g}")},
                   num=["mix", "previous", "corrected", "change"], cls="small")
    cmpd = pd.read_csv(os.path.join(dz.ROOT, "results", "tables", "T24_comparison_with_previous_data.csv"))
    cmp_m = cmpd[cmpd.group.isin(["model", "term"])]
    cmp_tab = table(cmp_m, ["quantity", "previous", "current"], ["Quantity", "Previous corrected data", "Current corrected data"],
                    {"previous": minus, "current": minus}, num=["previous", "current"], cls="small")
    cmp_o = cmpd[cmpd.group.isin(["optimum", "age"])]
    cmp_tab2 = table(cmp_o, ["quantity", "previous", "current"], ["Quantity", "Previous corrected data", "Current corrected data"],
                     {"previous": minus, "current": minus}, num=["previous", "current"], cls="small")
    v1 = pd.read_csv(os.path.join(dz.ROOT, "results", "tables", "T00b_changes_vs_previous_corrected_data.csv"))
    v1_tab = table(v1, ["mix", "variable", "previous", "corrected", "change"],
                   ["Mix", "Variable", "Previous corrected data", "Current corrected data", "Change"],
                   {"previous": lambda v: "–" if not np.isfinite(v) else f"{v:.2f}", "corrected": lambda v: "–" if not np.isfinite(v) else f"{v:.2f}",
                    "change": lambda v: "–" if not np.isfinite(v) else minus(f"{v:+.2f}")},
                   num=["mix", "previous", "corrected", "change"], cls="small")
    # bootstrap frequencies
    bf = pd.DataFrame([{"term": t, "lab": tlabel(t), "fin": "●" if t in final else "", "imp": R["search"]["UN"]["importance"][t],
                        "rb": R["bootstrap"]["resid"]["inclusion"][t], "sb": R["bootstrap"]["subsample"]["inclusion"][t],
                        "ev": ev[t]} for t in dz.ALL_TERMS]).sort_values("imp", ascending=False)
    bf_tab = table(bf, ["lab", "fin", "imp", "rb", "sb", "ev"],
                   ["Term", "In final", "Akaike weight", "Residual bootstrap", "Subsamples 24/30", "Evidence"],
                   {"imp": f2, "rb": f2, "sb": f2, "ev": lambda v: f'<span class="chip {v.split()[0].lower()}">{v}</span>'},
                   num=["imp", "rb", "sb"], cls="small")
    # added terms
    at = pd.DataFrame(R["added_terms"])
    at["dfs"] = [f"{int(a)}, {b:.1f}" for a, b in zip(at.df1, at.df2)]
    at_tab = table(at, ["added", "dfs", "F", "p", "dAICc"], ["Term(s) added to the final model", "df", "F", "p", "ΔAICc"],
                   {"F": f2, "p": rt.pnum, "dAICc": lambda v: minus(f"{v:+.2f}")}, num=["dfs", "F", "p", "dAICc"], cls="small")
    # covariance table
    cvt = pd.DataFrame(R["covariance_full"]["table"])
    cvt["lab"] = cvt.structure.map({"IND": "Independent, equal variance (ignores pairing)", "INDH": "Independent, age-specific variance (ignores pairing)",
                                    "CS": "CS: random mixture intercept, equal variance", "UN": "UN: random intercept + age-specific variance"})
    cv_tab = table(cvt, ["lab", "n_cov_par", "reml_loglik", "AIC_REML", "sd7", "sd28", "rho"],
                   ["Within-mixture covariance", "Parameters", "REML log-lik", "REML AIC", "SD 7 d (ln)", "SD 28 d (ln)", "ρ(7 d, 28 d)"],
                   {"reml_loglik": lambda v: minus(f2(v)), "AIC_REML": f2, "sd7": f3, "sd28": f3, "rho": f2},
                   num=["n_cov_par", "reml_loglik", "AIC_REML", "sd7", "sd28", "rho"], cls="small")
    code = pd.DataFrame(R["coding"]["table"])
    code["lab"] = code.coding.map({"cat4": "4-level categorical (primary)", "onoff": "Carbonated yes / no", "numeric": "Duration, numeric (h)",
                                   "log": "Duration, log₂(1 + h)"})
    code["bt"] = code.best_terms.map(lambda s: ", ".join(TERM_TXT.get(t, t) if not t.endswith(".Age") else tlabel(t) for t in s.split() if t != "Age"))
    code_tab = table(code, ["lab", "AICc_final_structure", "final_predR2_7", "final_predR2_28", "best_AICc", "dAICc_vs_cat4", "bt"],
                     ["Carbonation coding", "AICc, final terms", "Pred R² 7 d", "Pred R² 28 d", "AICc, best model", "ΔAICc vs 4-level", "Best model under this coding"],
                     {"AICc_final_structure": f2, "final_predR2_7": f2, "final_predR2_28": f2, "best_AICc": f2,
                      "dAICc_vs_cat4": lambda v: minus(f"{v:+.2f}")},
                     num=["AICc_final_structure", "final_predR2_7", "final_predR2_28", "best_AICc", "dAICc_vs_cat4"], cls="small")

    eqs = R["final"]["actual"]
    eq7 = eq_html(eqs["7 d"], "ln(f<sub>7</sub>)")
    eq28 = eq_html(eqs["28 d"], "ln(f<sub>28</sub>)")
    eqg = eq_html(eqs["gain"], "ln(f<sub>28</sub>/f<sub>7</sub>)")
    b = dict(zip(names, R["final"]["beta"]))
    coded_eq = ("ln f = {Intercept:.4f} + {A:.4f}·A + {D:.4f}·D + {c05:.4f}·C<sub>0.5h</sub> + {c1:.4f}·C<sub>1h</sub> + {c5:.4f}·C<sub>5h</sub>"
                " + {AD:.4f}·AD + {D2:.4f}·D² + {KA:.4f}·K·A + {KD:.4f}·K·D + g·({Age:.4f} + {DAge:.4f}·D + {D2Age:.4f}·D²)").format(
        Intercept=b["Intercept"], A=b["A"], D=b["D"], c05=b["Carb[0.5 h]"], c1=b["Carb[1 h]"], c5=b["Carb[5 h]"], AD=b["AD"], D2=b["D2"],
        KA=b["KA"], KD=b["KD"], Age=b["Age"], DAge=b["D.Age"], D2Age=b["D2.Age"]).replace("+ -", "− ")
    coded_eq = minus(coded_eq).replace("·C<sub>0.5h</sub>", "·C<sub>0.5h</sub>")

    sso = K["sso"]
    ob = K["opt_boot"]

    def ssopt(age, carb, rcf):
        r = sso[(age, carb, rcf)]
        return f"{r.SS_opt:.0f} %"

    def obci(key):
        o = ob[key]
        return f"{o['lo']:.0f}–{o['hi']:.0f} %"

    def lvl(*terms):
        order = ["Robust", "Moderate", "Suggestive", "Not detected", "Retained for hierarchy"]
        labs = [ev[t] for t in terms]
        return max(labs, key=order.index)   # the weakest label among the terms a finding rests on

    findings = [
        (lvl("D", "D2"), "Strength rises with SS to a peak at mid-range SS, at both ages",
         f"SS and SS² are the largest terms ({K['p_D']}, {K['p_D2']}) and are kept in at least {K['min_incl_SS']} of re-selections. The peak lies at SS "
         f"{ssopt(7, 'NC', 30)} (7 d) and {ssopt(28, 'NC', 30)} (28 d) for uncarbonated mixes at RCF 30 %, and lower "
         f"(SS {ssopt(7, '1 h', 50)} at 7 d, {ssopt(28, '1 h', 50)} at 28 d) for carbonated mixes at RCF 50 %."),
        (lvl("D.Age"), "The 7 → 28-day gain is set by SS",
         f"f<sub>28</sub>/f<sub>7</sub> = {K['g0']} (95 % CI {K['g0_ci']}) at SS 0 %, falls to a minimum of {K['gmin']} near SS {K['gmin_ss']} % "
         f"and is {K['g75']} ({K['g75_ci']}) at SS 75 % (SS × Age {K['p_D.Age']}). NaOH-only mixes gain most. "
         f"SS is the only age-dependent variable in {K['resid_gainSS']} of residual-bootstrap and {K['subsample_gainSS']} of subsample re-selections. "
         f"The curvature of the gain (SS² × Age, {K['p_D2.Age']}) is {ev['D2.Age'].lower()}."),
        (lvl("KD"), "Carbonated RCF helps in NaOH-rich mixes and hurts in silicate-rich ones",
         f"Carbonated × SS {K['p_KD']}. At RCF 50 % and SS 0 % a 1 h carbonated mix is {K['carb_50_0']} the uncarbonated one; at RCF 10 % and "
         f"SS 75 % it is {K['carb_10_75']}. The crossover SS rises with RCF, from {K['cross_1h'][10]} at RCF 10 % to {K['cross_1h'][50]} at RCF 50 % (1 h). "
         f"Kept in {rt.pct(R['bootstrap']['resid']['inclusion']['KD'])} of residual-bootstrap but only {rt.pct(R['bootstrap']['subsample']['inclusion']['KD'])} of subsample re-selections: it rests on the 8 uncarbonated mixes."),
        (lvl("KA", "AD"), "RCF raises strength when it is carbonated and when SS is low",
         f"Carbonated × RCF {K['p_KA']}, RCF × SS {K['p_AD']}. Going from 10 to 50 % RCF multiplies strength by {K['rcf_c0']} for carbonated "
         f"RCF at SS 0 %, and by {K['rcf_nc75']} for uncarbonated RCF at SS 75 %." + (" RCF never lowers strength significantly." if K["rcf_never_lowers"] else "")
         + (f" The case-bootstrap intervals of {K['case_zero']} include zero, so these two terms depend on a few mixtures." if K["case_zero"] else "")),
        (lvl("Carb.Age"), "Carbonation duration (0.5, 1, 5 h): no difference in level; a smaller 28-day gain after 5 h is possible",
         f"The three carbonated levels do not differ in level ({K['p_dur']}); on/off coding fits about equally well "
         f"(ΔAICc {K['dAICc_onoff']}). Carbonation × age is {'suggestive but not selected' if K['add_Carb.Age_val'] < 0.10 else 'not supported'} "
         f"(added-term {K['add_Carb.Age']}, ΔAICc {K['add_Carb.Age_dAICc']}). In the age-specific fit the 5 h level gains "
         f"less than NC (×{K['c5_change']}, 95 % CI {K['c5_change_ci']}; {K['chg_Carb[5 h]']})"
         + (", a single contrast that the joint test does not support." if K["add_Carb.Age_val"] >= 0.05 else ".")
         + (f" The only mixes that do not gain strength from 7 to 28 days are 5 h mixes ({K['nogain_txt']})." if K["nogain_5h"] else "")),
        (lvl("C", "C2", "C.Age"), "A/B ratio (0.42–0.48): no effect detected",
         f"No A/B term improves the model at either age (A/B {K['add_C']}, (A/B)² {K['add_C2']}, A/B × Age {K['add_C.Age']}). "
         f"A/B curvature is selected only if the 28-day variance is forced equal to the 7-day variance (CS); under nested cross-validation "
         f"that search predicts 7-day strength clearly worse (R² {K['ncv_cs7']} vs {K['ncv_primary7']}) for a small gain at 28 days "
         f"({K['ncv_cs28']} vs {K['ncv_primary28']})."),
    ]
    card_html = "\n".join(
        f'<article class="card"><span class="chip {lvl.split()[0].lower()}">{lvl}</span><h3>{t}</h3><p>{body}</p></article>'
        for lvl, t, body in findings)

    fsd = R["final"]["fit_stats"]
    stat_strip = f"""
<div class="strip">
  <div><span class="lab">7 d · R²</span><span class="val">{K['R2_7']}</span><span class="sub">pred. R² {K['pr7']} · RMSE {K['rm7']} MPa</span></div>
  <div><span class="lab">28 d · R²</span><span class="val">{K['R2_28']}</span><span class="sub">pred. R² {K['pr28']} · RMSE {K['rm28']} MPa</span></div>
  <div><span class="lab">Residual scatter</span><span class="val">{K['cv7']} / {K['cv28']}</span><span class="sub">7 d / 28 d (replicate {100*float(K['pe7']):.0f} % / {100*float(K['pe28']):.0f} %)</span></div>
  <div><span class="lab">7 ↔ 28 d correlation</span><span class="val">{K['rho']}</span><span class="sub">within mixture (paired)</span></div>
</div>"""

    protocol = [
        "Response scale by Box–Cox profile likelihood on both ages jointly: ln is used if λ = 0 lies in the 95 % CI of the selected model and λ = 1 lies outside.",
        "The 7- and 28-day results of a mixture are modelled as a correlated pair (random mixture intercept); the within-mixture covariance is either compound symmetric (CS) or has an age-specific residual variance (UN).",
        "Carbonation enters as a 4-level categorical factor. Its interaction with RCF, SS and A/B goes through the carbonated contrast K, because the 0.5 h level contains only two distinct compositions. An alternative coding replaces it only if its best model has AICc lower by ≥ 2 and better cross-validated prediction at both ages.",
        f"Candidate terms: quadratic in RCF, A/B and SS, carbonation, K × (RCF, A/B, SS), and every one of these × curing age (26 terms, 35 parameters). All {K['n_models']} models that respect hierarchy were fitted under CS and UN; the lowest AICc is the final model.",
        "Cross-checks: backward elimination (α = 0.10 and 0.05, hierarchy kept), Akaike weights, nested cross-validation of each selection procedure, re-selection with each mixture deleted, 2,000 residual-bootstrap and 2,000 subsample (24 of 30) re-selections.",
        "Validation: lack of fit against replicate batches, residual diagnostics, leave-one-mixture-out and leave-one-design-point-out prediction, 2,000 case-bootstrap refits.",
        "Robustness: other scales, covariance, codings, weights, a robust fit, influential mixtures excluded, age-specific and per-age models.",
        "Interpretation last: contrasts with 95 % CIs, gain ratios and optima, reported only where the model and the checks support them.",
    ]
    proto_html = "<ol class='proto'>" + "".join(f"<li>{p}</li>" for p in protocol) + "</ol>"

    optrow = {(o["age"], o["carbonation"]): o for o in R["optimum"]}
    o7, o28 = max((o for o in R["optimum"] if o["age"] == 7), key=lambda o: o["median"]), max((o for o in R["optimum"] if o["age"] == 28), key=lambda o: o["median"])

    body = f"""
<header class="top">
  <p class="eyebrow">Compressive strength · 30 mixtures tested at 7 and 28 days · linear mixed model</p>
  <h1>Paired 7/28-Day Strength Model</h1>
  <p class="lede">One model for both curing ages, fitted to the updated corrected results of the 30 mixes (see <a href="#changes">What changed</a>). Curing age is a two-level factor, carbonation a four-level factor, and the 7- and 28-day results of each mixture are treated as a correlated pair. The terms were chosen by an exhaustive search of {K['n_models']} hierarchical models and then validated before any effect was interpreted.</p>
  <p class="meta">Data: <code>data/strength_7d_28d_corrected.csv</code> · Analysis: <code>analysis/run_analysis.py</code> · {R['meta']['n_boot']:,} resamples per resampling scheme · built {R['meta']['date']}</p>
  <nav class="toc" aria-label="Sections">
    <a href="#findings">Findings</a><a href="#changes">What changed</a><a href="#model">Model</a><a href="#age">Age effect</a><a href="#surfaces">Surfaces &amp; predictor</a>
    <a href="#selection">Selection</a><a href="#validation">Validation</a><a href="#robustness">Robustness</a><a href="#data">Data</a><a href="#repro">Reproduce</a>
  </nav>
</header>

<main>
<section id="findings">
  <h2>What the data support</h2>
  <p class="note">Evidence labels follow one rule for every term. <b>Robust</b>: in the final model with p &lt; 0.01, kept in ≥ 90 % of residual-bootstrap and ≥ 80 % of subsample re-selections. <b>Moderate</b>: in the final model with p &lt; 0.05 and kept in ≥ 60 % of residual-bootstrap re-selections. <b>Suggestive</b>: not in the final model but added-term p &lt; 0.10 or kept in ≥ 50 % of residual-bootstrap re-selections. <b>Not detected</b>: none of these. A finding carries the weakest label of the terms it rests on.</p>
  <div class="cards">{card_html}</div>
  <div class="callout">
    <h3>How far to trust 28-day predictions</h3>
    <p>At 7 days the model predicts a left-out mixture with R² {K['pr7']} and an error of {K['rm7']} MPa, close to the {K['pe7M']} MPa scatter between replicate batches. At 28 days the residual scatter is {K['cv28']} (replicate batches {100*float(K['pe28']):.0f} %) and a left-out mixture is predicted with R² {K['pr28']} (RMSE {K['rm28']} MPa). Under nested cross-validation, where the selection is repeated in every fold, no procedure predicted 28-day strength of a new mixture with R² above {K['ncv_best28']} (primary procedure {K['ncv_primary28']}); the procedure that reached it ({K['ncv_best28_proc']}) predicted 7-day strength with R² {K['ncv_best28_7']} against {K['ncv_primary7']}. Weighting each mean by its specimen scatter does not help (see Robustness). The 28-day equation gives the expected trend, with its terms estimated jointly from both ages, but the strength of an individual new mixture at 28 days is uncertain by a factor of about {K['pi28']} either way (95 % prediction interval; {K['pi7']} at 7 days).</p>
  </div>
</section>

<section id="changes">
  <h2>What changed with the corrected data</h2>
  <p>{K['chg_intro']}</p>
  {cmp_tab}
  <details><summary>Optima and 7 → 28-day changes of each effect, previous vs current data</summary>{cmp_tab2}</details>
  <p class="note">Previous: the same protocol (<code>analysis/run_analysis.py</code>) run on the previous corrected data set (<code>data/previous_corrected_v1.csv</code>); key results kept in <code>results/previous_v1/key_results.json</code>. Full table: <code>results/tables/T24_comparison_with_previous_data.csv</code>.</p>
</section>

<section id="model">
  <h2>The final model</h2>
  <p>ln-strength of mixture <i>i</i> at age <i>a</i> is modelled as fixed mixture and age effects plus a correlated pair of errors (UN covariance: SD {K['sd7']} at 7 d, {K['sd28']} at 28 d, correlation {K['rho']}). In coded units (A = (RCF − 30)/20, D = (SS − 37.5)/37.5, K = 1 if the RCF is carbonated, C<sub>x</sub> = carbonation-level indicators, g = −½ at 7 d and +½ at 28 d):</p>
  <p class="eq">{coded_eq}</p>
  <p>The terms outside the bracket are common to both ages. The bracket is the ln gain ln(f<sub>28</sub>/f<sub>7</sub>), which depends on SS only. Medians in MPa are exp(ln f); means are about {100*(np.exp(float(K['sd7'])**2/2)-1):.0f} % (7 d) and {100*(np.exp(float(K['sd28'])**2/2)-1):.0f} % (28 d) higher.</p>
  <h3>Equations in actual units</h3>
  <p class="note">RCF and SS in %, K = 1 for carbonated RCF (0.5, 1 or 5 h), C[·] = 1 for that carbonation level. A/B does not appear.</p>
  <p class="eq">{eq7}</p>
  <p class="eq">{eq28}</p>
  <p class="eq">{eqg}</p>
  {stat_strip}
  <h3>Terms</h3>
  {tf_tab}
  <p class="note">Wald F tests with Satterthwaite denominator df from the REML fit. RCF and the carbonation main effect are kept because terms containing them are in the model (hierarchy); on their own they describe the uncarbonated mix at SS 37.5 %. Lack of fit against replicate batches: gain {K['lof_gain']}, level given gain {K['lof_level']}; per age, 7 d {K['lof7']} and 28 d {K['lof28']}.</p>
  <h3>Coefficients (coded units) with three kinds of interval</h3>
  {coef_tab}
  <p class="note">Treating the 60 results as independent would change the standard errors by a factor of {K['se_ind_level']} for the SS level term and {K['se_ind_gain']} for the SS × Age term, which is why the pairing is modelled.</p>
</section>

<section id="age">
  <h2>How the response changes from 7 to 28 days</h2>
  <p>In the selected model only SS changes its effect with age. Silicate-free (NaOH-only) mixes are weak at 7 days and nearly triple by 28 days (×{K['g0']}); silicate-rich mixes are relatively further along at 7 days and rise by about half as much (×{K['g75']} at SS 75 %). The gain at SS 0 % is {K['g0v75']} times the gain at SS 75 % (95 % CI {K['g0v75_ci']}; {K['g0v75_p']}). The apparent rise of the gain above SS {K['gmin_ss']} % is not significant ({K['g75vmin_p']}).</p>
  <figure class="chart"><div id="gainChart" class="plot" role="img" aria-label="Strength gain ratio f28 over f7 against SS with model curve and observed mixtures"></div>
  <figcaption>Gain ratio f<sub>28</sub>/f<sub>7</sub> against SS. Line: model with 95 % CI (darker band) and 95 % case-bootstrap band (lighter band). Markers: observed mixtures by carbonation level. Hover a marker for its data.</figcaption></figure>

  <h3>Strength against SS at both ages</h3>
  <div class="controls" role="group" aria-label="Curve settings">
    <label for="rcfSlider">RCF <output id="rcfOut">30</output> %</label>
    <input type="range" id="rcfSlider" min="10" max="50" step="1" value="30">
    <div class="seg" id="carbSeg" role="radiogroup" aria-label="Carbonation"></div>
  </div>
  <figure class="chart"><div id="curveChart" class="plot" role="img" aria-label="Predicted strength at 7 and 28 days against SS"></div>
  <figcaption>Median predicted strength with 95 % CI. Markers are runs at the selected carbonation level within ±10 % RCF of the slider value. Open circles mark each curve's peak.</figcaption></figure>

  <h3>Does any other effect change with age?</h3>
  <p>The same terms were refitted with every term interacting with age. RCF × SS, carbonated × RCF and carbonated × SS keep their sign from 7 to 28 days; their 28-day estimates are smaller and less precise, and none of the changes is significant (RCF × SS, carbonated × RCF, carbonated × SS: {K['chg_range_int']}; all omitted age terms jointly {K['omit_joint']}). The data are therefore consistent with the same proportional effects at both ages but cannot exclude a moderate weakening at 28 days.{f" On its own, the 5 h carbonation level gains less than NC (×{K['c5_change']}, {K['chg_Carb[5 h]']}); the joint carbonation × age test gives {K['add_Carb.Age']}." if K['chg_c5_val'] < 0.05 else ""}</p>
  {ag_tab}
  <figure class="plate"><img src="figures/F04_age_specific_effects.png" alt="Forest plot of age-specific effects at 7 and 28 days" loading="lazy"></figure>
  <details><summary>Added-term tests for every omitted term</summary>{at_tab}</details>
</section>

<section id="surfaces">
  <h2>Response surfaces and predictor</h2>
  <div class="controls" role="group" aria-label="Map settings">
    <div class="seg" id="ageSeg" role="radiogroup" aria-label="Age"></div>
    <div class="seg" id="mapCarbSeg" role="radiogroup" aria-label="Carbonation"></div>
  </div>
  <div class="split">
    <figure class="chart"><div id="mapChart" class="plot" role="img" aria-label="Predicted strength over RCF and SS"></div>
    <figcaption>Median predicted strength over RCF × SS. Dots: runs at this carbonation level. Star: maximum. Hover for values; click to load a point into the predictor.</figcaption></figure>
    <div class="predictor" id="predictor">
      <h3>Predictor</h3>
      <div class="field"><label for="pRCF">RCF (%)</label><input id="pRCF" type="number" min="10" max="50" step="0.5" value="50"></div>
      <div class="field"><label for="pSS">SS (%)</label><input id="pSS" type="number" min="0" max="75" step="0.5" value="45"></div>
      <div class="field"><label for="pCarb">Carbonation</label><select id="pCarb"><option>NC</option><option>0.5 h</option><option selected>1 h</option><option>5 h</option></select></div>
      <p class="note">A/B is not in the model, so it does not change the prediction.</p>
      <div id="pOut" class="pout" aria-live="polite"></div>
    </div>
  </div>
  <h3>Best combination by age and carbonation level</h3>
  {opt_tab}
  <p class="note">Maximum of the median prediction over RCF 10–50 % and SS 0–75 %. The CI covers the mean of many batches, the PI one new batch. No optimum needs extrapolation: its scaled prediction variance is below the largest value at a design point. The strongest carbonated optimum and the uncarbonated optimum do not differ at either age: 7 d {K['optc7']}; 28 d {K['optc28']}.</p>
  <figure class="plate"><img src="figures/F07_carbonation_and_RCF_effects.png" alt="Carbonation and RCF effect ratios against SS" loading="lazy"></figure>
</section>

<section id="selection">
  <h2>Scale, pairing, coding and term selection</h2>
  <h3>Scale</h3>
  <p>For the selected model Box–Cox gives {K['bc_final']}: ln is inside the interval and no transformation is rejected (LR {K['bc_LR1']}). The saturated candidate model gives {K['bc_full']}{f", just excluding 0; that interval includes 0 when any one of several mixtures is removed ({K['bc_del_inc0']} of 30 deletions)" if K['bc_full_excl0'] else f", which includes 0 ({K['bc_del_inc0']} of 30 single-mixture deletions keep 0 inside)"}. The analysis was repeated on the λ = −0.5 scale (see Robustness).</p>
  <h3>Pairing</h3>
  {cv_tab}
  <p class="note">Full candidate model, REML. UN vs CS: LR {K['cov_LR']}, {K['cov_p']}. Both structures were carried into the model search; the selected combination (UN) beats the best CS model by ΔAICc = {K['daicc_struct']}.</p>
  <h3>Carbonation coding</h3>
  {code_tab}
  <p class="note">The on/off and log-duration codings fit marginally better (ΔAICc {K['dAICc_onoff']} and {K['dAICc_log']}), which is short of the pre-set threshold of 2. The log-duration model's better 28-day prediction disappears under nested cross-validation (7 d {K['ncv_log7']}, 28 d {K['ncv_log28']}). The 4-level coding is kept, as requested unless the data clearly favour another.</p>
  <h3>Term selection</h3>
  {top_tab}
  <p class="note">Exhaustive search, UN covariance, top 10 of {K['n_models']}. The selected model has {int(R['search']['UN']['top'][0]['k'])} parameters and is {K['un2']} AICc units ahead of the next model. Backward elimination at α = 0.10 {'keeps more age interactions' if K['be10_more_age'] else 'returns a different model'}; at α = 0.05 it {'returns the final model' if K['be05_same'] else ('returns a subset of the final model' if K['be05_subset'] else 'returns a different model')} (table T07). With each mixture deleted in turn the search returns the same model {K['del_same']} times out of 30.</p>
  <figure class="plate"><img src="figures/F02_scale_and_selection.png" alt="Box–Cox profile, term support and nested cross-validation" loading="lazy"></figure>
  <h3>Nested cross-validation of the selection procedures</h3>
  {ncv_tab}
  <p class="note">Each procedure is re-run on the training mixtures of every fold, so the numbers include the cost of choosing the model. The primary procedure selects the final model in {K['ncv_primary_modal']} of 30 folds. Backward elimination and the CS-only search predict 7-day strength clearly worse; no procedure predicts 28-day strength of a new mixture well.</p>
  <details><summary>Term support: Akaike weights and re-selection frequencies</summary>{bf_tab}</details>
</section>

<section id="validation">
  <h2>Validation and diagnostics</h2>
  <div class="controls"><div class="seg" id="opSeg" role="radiogroup" aria-label="Prediction type"></div></div>
  <figure class="chart"><div id="opChart" class="plot" role="img" aria-label="Observed against predicted strength"></div>
  <figcaption>Observed against model strength (log axes). Hover a point for the mixture.</figcaption></figure>
  <p>Whitened residuals are consistent with normality (Shapiro–Wilk {K['sw_p']}) and constant variance (against fitted value and age {K['bp_p']}; against the factors {K['bp2_p']}), with no trend over mix number (ρ = {K['run_rho']}, {K['run_p']}). The largest deleted studentized residual is {K['max_t']} (mix {K['max_t_mix']} at {K['max_t_age']} days: {K['max_t_obs']} MPa observed against {K['max_t_fit']} MPa fitted and {K['max_t_cv']} MPa predicted with the mixture left out), which is {'not an outlier' if K['max_t_bonf_val'] >= 0.05 else 'an outlier'} after Bonferroni correction ({K['max_t_bonf']}). Mix {K['cook_mix']} has the largest influence (Cook's D {K['cook_max']}). {K['excl_txt']}</p>
  <figure class="plate"><img src="figures/F09_residual_diagnostics.png" alt="Residual diagnostics" loading="lazy"></figure>
</section>

<section id="robustness">
  <h2>Robustness</h2>
  <p>Key quantities recomputed under alternative analyses. The SS dependence of the gain holds in every analysis: gain at SS 0 % between {rb.gain_SS0.min():.1f} and {rb.gain_SS0.max():.1f}, at SS 75 % between {rb.gain_SS75.min():.2f} and {rb.gain_SS75.max():.2f}. The 28-day carbonation and RCF ratios are much less stable. They depend on whether these effects are assumed proportional across ages, which is the least certain part of the model.</p>
  <p>Several 28-day means in the corrected data carry large specimen SDs. Weighting each mean by its own specimen scatter (CV²/n added to its variance) was tested as an alternative and not adopted: the model residuals are unrelated to specimen scatter (Spearman ρ between |residual| and specimen CV: 7 d {K['spec_rho7']}, {K['spec_p7']}; 28 d {K['spec_rho28']}, {K['spec_p28']}), the weighted fit has a worse REML likelihood (ΔAIC {K['spec_dAIC']}) and it predicts left-out mixtures no better (pred. R² 7 d {K['spec_cv7']}, 28 d {K['spec_cv28']}). The 28-day scatter lies between batches, not between specimens of a batch.</p>
  {rb_tab}
  <figure class="plate"><img src="figures/F10_robustness.png" alt="Robustness of key quantities" loading="lazy"></figure>
  <figure class="plate"><img src="figures/F11_coefficient_intervals.png" alt="Wald and bootstrap intervals for the coefficients" loading="lazy"></figure>
  <p class="note">Residual bootstrap: {K['resid_n']} resamples of the whole selection; the exact final model is re-selected in {K['resid_exact']}, the same terms in {K['resid_same']}. Subsamples of 24 mixtures: {K['subsample_n']} re-selections, exact model {K['subsample_exact']}. Case bootstrap: {K['case_n']} refits of the final terms. The case bootstrap is not used for selection because duplicated mixtures make AICc favour near-saturated models.</p>
</section>

<section id="data">
  <h2>Data</h2>
  <p>{D['n_mixes']} mixtures, each tested at 7 and 28 days ({D['n_obs']} means of typically three specimens; single specimens for mix 29 at both ages and mixes 11 and 30 at 28 days). Four nominal design points were batched more than once (mixes 1/12/19, 10/14, 6/17/28, 8/23), giving 6 df of replicate (pure) error per age: SD {K['pe7M']} MPa at 7 d and {K['pe28M']} MPa at 28 d, or {K['pe7']} and {K['pe28']} on the ln scale. The ln scale makes the replicate scatter equal at the two ages. The 0.5 h level has only two distinct compositions (mixes 1/12/19 and 10/14).</p>
  <p>The data are the corrected results of the 30 mixes (<code>data/source/30_mixes_7_and_28_days_strength_results.pdf</code>), which replace the previous corrected sheet. Compositions are unchanged. {K['v1_n']} entries differ from the previous sheet: the 28-day strength of {K['v1_28_n']} mixes (mixes {K['v1_28_mixes']}; changes {K['v1_28_range']}, mean {K['v1_28_mean']} MPa), the 7-day strength of {K['v1_7_txt']}, {K['v1_round_n']} strengths re-rounded by 0.01 MPa, and {K['v1_sd28_n']} 28-day specimen SDs ({K['v1_sd28_up']} of them larger). Regressions use the values as given; replicate groups are defined by nominal design point.</p>
  <details><summary>All {K['v1_n']} differences from the previous corrected data set</summary>{v1_tab}</details>
  <p>Against the run sheet behind the earlier 7-day model the data differ in {D['n_changes']} entries. {K['t00_big']}; others add a third decimal to A/B, set SS of mix 24 to 0 % (was 0.88 %) and A/B of mix 9 to 0.442 (was 0.43).</p>
  <details><summary>All {D['n_changes']} differences from the previous 7-day data</summary>{ch_tab}</details>
  <figure class="plate"><img src="figures/F01_data_overview.png" alt="Paired strengths and observed gain against SS" loading="lazy"></figure>
</section>

<section id="protocol">
  <h2>Protocol</h2>
  <p>Fixed before the final model was chosen, in this order:</p>
  {proto_html}
</section>

<section id="repro">
  <h2>Reproduce</h2>
  <pre class="code">pip install -r requirements.txt
python analysis/prepare_data.py     # long format, change log, replicate error
python analysis/run_analysis.py     # selection, validation, resampling, robustness (about 10 min on 4 cores)
python analysis/compare_previous.py # previous vs current data (T24)
python analysis/make_figures.py     # results/figures/*.png|pdf
python analysis/build_report.py     # report/index.html, report/Model_Summary.pdf, report/MODEL_REPORT.md</pre>
  <p class="note">Tables T00–T24 in <code>results/tables</code> hold every number on this page. The mixed model is implemented in <code>analysis/lmm.py</code> (REML/GLS for the bivariate 7/28-day response, Satterthwaite df). Under compound symmetry it reproduces ordinary least squares on the mixture mean and gain exactly, and under UN the exhaustive search uses the exact factorisation of the likelihood (<code>analysis/selection.py</code>). Seed {R['meta']['seed']}; NumPy {R['meta']['numpy']}, pandas {R['meta']['pandas']}.</p>
</section>
</main>
"""

    css = CSS
    js = JS.replace("__DATA__", page_json)
    page = f"""<title>Paired 7/28-Day Strength Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<div class="wrap">{body}</div>
<div id="tip" class="tip" hidden></div>
<script>{js}</script>
"""
    with open(os.path.join(REP, "index.html"), "w") as f:
        f.write(page)
    # figures next to the page (for local viewing and for publishing)
    os.makedirs(os.path.join(REP, "figures"), exist_ok=True)
    for fn in ("F01_data_overview", "F02_scale_and_selection", "F04_age_specific_effects", "F07_carbonation_and_RCF_effects",
               "F09_residual_diagnostics", "F10_robustness", "F11_coefficient_intervals"):
        shutil.copy(os.path.join(dz.ROOT, "results", "figures", fn + ".png"), os.path.join(REP, "figures", fn + ".png"))
    print("wrote report/index.html")

    build_markdown(R, K, ex, final, tf, coef, ag, top, ncvm, rb, opt, code, cvt, at, bf, eqs, coded_eq)
    build_readme(R, K)
    build_summary(R, K, findings, coded_eq, eq7, eq28, eqg, tf_tab, ag_tab, opt_tab, stat_strip)


# ============================================================== Markdown report
def build_markdown(R, K, ex, final, tf, coef, ag, top, ncvm, rb, opt, code, cvt, at, bf, eqs, coded_eq):
    import re

    def plain(s):
        s = re.sub(r"<sub>(.*?)</sub>", r"\1", s)
        s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
        s = re.sub(r"<[^>]+>", "", s)
        return html.unescape(s)

    eqtxt = {k: plain(eq_html(v, {"7 d": "ln(f7)", "28 d": "ln(f28)", "gain": "ln(f28/f7)"}[k])) for k, v in eqs.items()}
    oc_ = [o for o in R["optimum"] if o["carbonation"] != "NC"]
    on_ = [o for o in R["optimum"] if o["carbonation"] == "NC"]
    opt_c_rcf = f"{min(o['RCF'] for o in oc_):.0f} %" if len({o['RCF'] for o in oc_}) == 1 else f"{min(o['RCF'] for o in oc_):.0f}–{max(o['RCF'] for o in oc_):.0f} %"
    opt_c_ss = f"{min(o['SS'] for o in oc_):.0f}–{max(o['SS'] for o in oc_):.0f} %"
    opt_nc_ss = f"{min(o['SS'] for o in on_):.0f}–{max(o['SS'] for o in on_):.0f} %"
    D = R["data"]
    md = f"""# Paired 7/28-day compressive-strength model

*Generated by `analysis/build_report.py` from `results/model_results.json`. Every number below comes from the analysis files; do not edit by hand.*

## Summary

A single linear mixed model describes the 7- and 28-day compressive strength of the 30 mixtures ({D['n_obs']} results), using the updated corrected results (`data/source/30_mixes_7_and_28_days_strength_results.pdf`). Curing age is a two-level factor, carbonation a four-level factor (NC, 0.5 h, 1 h, 5 h), and the two results of each mixture are a correlated pair. The model was chosen by an exhaustive AICc search over {K['n_models']} hierarchical models and two within-mixture covariance structures. It was checked with nested cross-validation, bootstrap and subsample re-selection, lack-of-fit tests, residual diagnostics and {len(rb) - 1} alternative analyses.

* **SS** is the dominant factor at both ages, with a peak at mid-range SS (SS {K['p_D']}; SS² {K['p_D2']}).
* **In the selected model, curing age interacts with SS only.** The strength gain f28/f7 is {K['g0']} (95 % CI {K['g0_ci']}) at SS 0 %, reaches a minimum of {K['gmin']} near SS {K['gmin_ss']} % and is {K['g75']} ({K['g75_ci']}) at SS 75 % (SS × Age {K['p_D.Age']}, SS² × Age {K['p_D2.Age']}).
* **Carbonated RCF** raises strength in NaOH-rich mixes and lowers it in silicate-rich mixes (K × SS {K['p_KD']}). RCF raises strength more when it is carbonated (K × RCF {K['p_KA']}) and when SS is low (RCF × SS {K['p_AD']}).
* **Carbonation duration**: the 0.5, 1 and 5 h levels do not differ ({K['p_dur']}).
* **A/B** (0.42–0.48) has no detectable effect (every A/B term added to the final model: p ≥ {K['add_C_min']}).
* **Reliability**: 7-day strength of a new mixture is predicted with R² {K['pr7']} (RMSE {K['rm7']} MPa, replicate scatter {K['pe7M']} MPa). 28-day strength is predicted poorly (R² {K['pr28']}, RMSE {K['rm28']} MPa): the 28-day residual scatter ({K['cv28']}) exceeds the replicate scatter ({100*float(K['pe28']):.0f} %).

## What changed with the corrected data

{K['chg_intro']}

{md_table(pd.read_csv(os.path.join(dz.ROOT, 'results', 'tables', 'T24_comparison_with_previous_data.csv')).query('group in ["model", "term"]'), ['quantity', 'previous', 'current'], ['Quantity', 'Previous corrected data', 'Current corrected data'])}

Full comparison, including optima and the 7 → 28-day change of each effect: `results/tables/T24_comparison_with_previous_data.csv`.

## 1 Data

* Corrected data set (controlling): `data/strength_7d_28d_corrected.csv`; long format `data/strength_long.csv`.
* {D['n_mixes']} mixtures, 24 distinct design points; replicate batches: mixes 1/12/19, 10/14, 6/17/28, 8/23 (6 df pure error per age).
* Pure-error SD: {K['pe7M']} MPa (7 d), {K['pe28M']} MPa (28 d); on the ln scale {K['pe7']} and {K['pe28']}, i.e. equal once the scale is logarithmic.
* Source: `data/source/30_mixes_7_and_28_days_strength_results.pdf` (corrected results of the 30 mixes); it replaces the previous corrected sheet (kept as `data/previous_corrected_v1.csv`). Compositions are unchanged.
* {K['v1_n']} differences from the previous corrected data (`results/tables/T00b_changes_vs_previous_corrected_data.csv`): the 28-day strength of {K['v1_28_n']} mixes (mixes {K['v1_28_mixes']}; {K['v1_28_range']}, mean {K['v1_28_mean']} MPa), the 7-day strength of {K['v1_7_txt']}, {K['v1_round_n']} strengths re-rounded by 0.01 MPa and {K['v1_sd28_n']} 28-day specimen SDs ({K['v1_sd28_up']} larger).
* {D['n_changes']} differences from the run sheet of the earlier 7-day model (`results/tables/T00_changes_vs_previous_7d_data.csv`). {K['t00_big']}.

## 2 Methods

**Model.** For mixture *i* and age *a* ∈ {{7, 28}}, ln f<sub>ia</sub> = x<sub>ia</sub>′β + e<sub>ia</sub>, with (e<sub>i7</sub>, e<sub>i28</sub>) bivariate normal with covariance Σ, independent across mixtures. Σ is either compound symmetric (CS: random mixture intercept, common residual variance) or unstructured (UN: random intercept plus age-specific residual variance; with two occasions the two coincide). Estimation is by REML/GLS; tests are Wald F tests with Satterthwaite degrees of freedom (as in lmerTest). The implementation (`analysis/lmm.py`) reproduces ordinary least squares on the mixture mean M = (ln f7 + ln f28)/2 and the gain G = ln f28 − ln f7 exactly under CS, which served as a numerical check.

**Coding.** A = (RCF − 30)/20, C = (A/B − 0.45)/0.03, D = (SS − 37.5)/37.5; age g = −½ (7 d), +½ (28 d). Carbonation is a 4-level factor; its interactions with the mixture variables use the carbonated contrast K because the 0.5 h level contains only two distinct compositions.

**Candidate set.** Quadratic in A, C, D; carbonation; K·A, K·C, K·D; each of these × age (26 terms, 35 parameters with the intercept and age).

**Selection.** Exhaustive AICc search of all {K['n_models']} models respecting strong hierarchy, under CS and UN. Under CS the likelihood factorises into OLS fits of M and G. Under UN, because hierarchy makes the gain terms a subset of the level terms, it factorises exactly into OLS(G | X<sub>G</sub>) and OLS(M | X<sub>M</sub>, G). The overall AICc minimum is the final model. Cross-checks: backward elimination (α = 0.10, 0.05), Akaike weights, nested leave-one-mixture-out and leave-one-design-point-out cross-validation of each selection procedure, single-mixture deletion, {K['resid_n']} residual-bootstrap and {K['subsample_n']} subsample (24 of 30 mixtures) re-selections.

**Scale.** Box–Cox profile likelihood for both ages jointly: {K['bc_final']} for the selected model (λ = 1 rejected, LR {K['bc_LR1']}); {K['bc_full']} for the saturated candidate model.

**Carbonation coding check.** Pre-set rule: replace the 4-level coding only if an alternative's best model is ≥ 2 AICc units better and predicts both ages better in cross-validation. On/off ΔAICc {K['dAICc_onoff']}, log duration {K['dAICc_log']}, numeric duration {K['dAICc_num']}: the 4-level coding is kept.

## 3 Final model

Coded form (g = −½ at 7 d, +½ at 28 d):

    {plain(coded_eq)}

Actual units (RCF, SS in %; K = 1 for carbonated RCF; C[·] carbonation-level indicators):

    {eqtxt['7 d']}
    {eqtxt['28 d']}
    {eqtxt['gain']}

Covariance (UN): SD {K['sd7']} (7 d), {K['sd28']} (28 d), correlation {K['rho']}. AICc {K['aicc_un']} (best CS model {K['aicc_cs']}).

### Term tests

{md_table(tf, ['lab', 'dfs', 'F', 'p', 'rb', 'sb', 'ev'], ['Term', 'df', 'F', 'p', 'Residual bootstrap', 'Subsamples 24/30', 'Evidence'], {'F': f2, 'p': rt.pnum})}

### Coefficients (coded)

{md_table(coef, ['lab', 'coef', 'se', 'df', 'ci', 'p', 'bres', 'bcase'], ['Term', 'Estimate', 'SE', 'df', '95 % CI', 'p', 'Residual bootstrap 95 %', 'Case bootstrap 95 %'], {'coef': lambda v: f'{v:.4f}', 'se': lambda v: f'{v:.4f}', 'df': lambda v: f'{v:.1f}', 'p': rt.pnum})}

### Fit statistics

| | 7 d | 28 d |
|---|---|---|
| R² (ln) | {K['R2_7']} | {K['R2_28']} |
| Adjusted R² | {K['adj7']} | {K['adj28']} |
| Predicted R², leave one mixture out | {K['pr7']} | {K['pr28']} |
| Predicted R², leave one design point out | {K['prg7']} | {K['prg28']} |
| RMSE fitted / predicted (MPa) | {K['fit_rm7']} / {K['rm7']} | {K['fit_rm28']} / {K['rm28']} |
| Residual scatter (CV) | {K['cv7']} | {K['cv28']} |
| Adequate precision | {K['adeq7']} | {K['adeq28']} |
| Lack of fit (per age) | {K['lof7']} | {K['lof28']} |

Joint lack of fit (exact, against replicate batches): gain {K['lof_gain']}; level given gain {K['lof_level']}.

## 4 How curing age changes the response

* Gain ratio f28/f7: SS 0 % {K['g0']} ({K['g0_ci']}); SS 37.5 % {K['g37']} ({K['g37_ci']}); SS 75 % {K['g75']} ({K['g75_ci']}); minimum {K['gmin']} at SS {K['gmin_ss']} % (95 % CI of location {K['gmin_ci']} %). Gain at SS 0 % / gain at SS 75 % = {K['g0v75']} ({K['g0v75_ci']}; {K['g0v75_p']}). The rise above the minimum is not significant ({K['g75vmin_p']}).
* No other age interaction is selected (smallest added-term p = {rt.pnum(pd.DataFrame(R['added_terms']).query('terms.str.contains(".Age")', engine='python').p.min())}{', and every addition raises AICc' if (pd.DataFrame(R['added_terms']).dAICc > 0).all() else ''}). In the age-specific refit the interaction effects keep their sign but are smaller and less precise at 28 d; the changes are not significant (RCF × SS {K['chg_AD']}, K × RCF {K['chg_KA']}, K × SS {K['chg_KD']}). The 5 h level shows a smaller gain than NC (×{K['c5_change']}, {K['c5_change_ci']}; {K['chg_Carb[5 h]']}), which the selection does not retain.
* SS at maximum strength: NC, RCF 30 % — {rt_ss(K, 7, 'NC', 30)} at 7 d, {rt_ss(K, 28, 'NC', 30)} at 28 d; carbonated (1 h), RCF 50 % — {rt_ss(K, 7, '1 h', 50)} at 7 d, {rt_ss(K, 28, '1 h', 50)} at 28 d.

{md_table(ag, ['lab', 'e7', 'e28', 'chg', 'p_change'], ['Term', '7 d [95 % CI]', '28 d [95 % CI]', 'Change', 'p (change)'], {'p_change': rt.pnum})}

## 5 Mixture effects (common to both ages on the ln scale)

* Carbonated (1 h) / NC at RCF 50 %, SS 0 %: {K['carb_50_0']}; at RCF 30 %, SS 0 %: {K['carb_30_0']}; at RCF 10 %, SS 75 %: {K['carb_10_75']}. Crossover SS (ratio = 1) for 1 h: RCF 10 % {K['cross_1h'][10]}, 30 % {K['cross_1h'][30]}, 50 % {K['cross_1h'][50]}.
* RCF 50 % / 10 %: carbonated, SS 0 % {K['rcf_c0']}; carbonated, SS 75 % {K['rcf_c75']}; NC, SS 0 % {K['rcf_nc0']}; NC, SS 75 % {K['rcf_nc75']}.
* Carbonation levels 0.5 / 1 / 5 h: not different ({K['p_dur']}).
* A/B: no effect detected (A/B {K['add_C']}; (A/B)² {K['add_C2']}; A/B × Age {K['add_C.Age']}).

## 6 Optimum

{md_table(opt, ['age_s', 'carbonation', 'where', 'median', 'cis', 'pis', 'box'], ['Age', 'Carbonation', 'Maximum at', 'Median (MPa)', '95 % CI', '95 % PI', 'Within 5 %'], {'median': f1})}

Best carbonated vs uncarbonated optimum: 7 d {K['optc7']}; 28 d {K['optc28']}. No optimum requires extrapolation.

## 7 Validation

* Whitened residuals: Shapiro–Wilk {K['sw_p']}; Breusch–Pagan (fitted, age) {K['bp_p']}, (factors) {K['bp2_p']}; mix-order trend ρ = {K['run_rho']} ({K['run_p']}).
* Largest deleted studentized residual {K['max_t']} (mix {K['max_t_mix']}, {K['max_t_age']} d; Bonferroni {K['max_t_bonf']}). Largest Cook's D {K['cook_max']} (mix {K['cook_mix']}). {K['excl_txt']} With any single mixture removed, {K['del_same']} of 30 re-selections give the same model.
* Nested cross-validation (selection repeated in every fold), leave one mixture out:

{md_table(ncvm, ['procedure', 'predR2_7', 'predR2_28', 'rmse_MPa_7', 'rmse_MPa_28', 'stab'], ['Procedure', 'Pred R² 7 d', 'Pred R² 28 d', 'RMSE 7 d', 'RMSE 28 d', 'Distinct models / folds'], {'predR2_7': f2, 'predR2_28': f2, 'rmse_MPa_7': f1, 'rmse_MPa_28': f1})}

## 8 Robustness

{md_table(rb, ['analysis', 'gain_SS0', 'gain_SS75', '1h/NC ratio RCF50 SS0 28d', '1h/NC ratio RCF10 SS75 28d', 'RCF50/10 ratio 1h SS0 28d', 'predR2_7', 'predR2_28'], ['Analysis', 'Gain SS 0', 'Gain SS 75', '1 h/NC RCF 50 SS 0 (28 d)', '1 h/NC RCF 10 SS 75 (28 d)', 'RCF 50/10 (28 d)', 'Pred R² 7 d', 'Pred R² 28 d'], {'gain_SS0': f2, 'gain_SS75': f2, '1h/NC ratio RCF50 SS0 28d': f2, '1h/NC ratio RCF10 SS75 28d': f2, 'RCF50/10 ratio 1h SS0 28d': f2, 'predR2_7': lambda v: '–' if v is None or not np.isfinite(v) else f2(v), 'predR2_28': lambda v: '–' if v is None or not np.isfinite(v) else f2(v)})}

Residual bootstrap re-selection: exact model {K['resid_exact']}, same terms {K['resid_same']}, SS the only age-dependent variable {K['resid_gainSS']}. Subsamples (24 of 30): exact model {K['subsample_exact']}, SS the only age-dependent variable {K['subsample_gainSS']}.

## 9 Interpretation and limits

1. The 7-day strength is described to within replicate precision by SS (quadratic), RCF, carbonation, and the interactions K × SS, K × RCF and RCF × SS.
2. From 7 to 28 days the model predicts an increase for every mixture; observed, {K['n_gain']} of 30 mixes gain strength{f" and mixes {K['nogain_mixes']} do not ({K['nogain_txt']})" if K['nogain_txt'] else ""}. The size of the increase is governed by SS: NaOH-only mixes roughly triple, silicate-rich mixes gain about half as much. No part of the gain depending on RCF or A/B is detected. {f"A smaller gain after 5 h carbonation is possible (5 h vs NC ×{K['c5_change']}, {K['chg_Carb[5 h]']}; joint carbonation × age {K['add_Carb.Age']}) but is not selected." if K['chg_c5_val'] < 0.05 else "No part of the gain depending on carbonation is detected."}
3. Because the gain does not depend on them, the model carries the 7-day RCF and carbonation effects to 28 days as the same percentage effects (larger in MPa, because 28-day strength is higher). {'The 28-day data agree with this in sign for RCF × SS, K × RCF and K × SS' if K['int_same_sign'] else 'The 28-day data do not agree in sign for every interaction'}{' (the 5 h level is the exception: at 28 d it lies below NC)' if K['c5_sign_flip'] else ''}, but they are too scattered to confirm the size of these effects at 28 days or to rule out moderate weakening. This is the main limitation.
4. 28-day strength of a new mixture is uncertain (95 % prediction interval about ×/÷ {K['pi28']}, against ×/÷ {K['pi7']} at 7 days) because 28-day results scatter more between mixtures than replicate batches do. Unrecorded differences between batches during curing, or data issues (for example mix {K['big28'][0]} at 28 d), are possible reasons that the data cannot settle. The 28-day residuals are not related to the specimen scatter of each mean (Spearman ρ = {K['spec_rho28']}, {K['spec_p28']}), so weighting the means by their SDs does not help (robustness table).
5. Suggested confirmation: triplicate batches tested at both ages at the predicted optima (carbonated RCF {opt_c_rcf}, SS ≈ {opt_c_ss}; uncarbonated SS ≈ {opt_nc_ss}), repeats of {K['big28_txt']} (the two largest 28-day residuals), and a 5 h vs NC comparison at equal composition to test the smaller gain seen after long carbonation.

## 10 Files

* `analysis/` — `prepare_data.py`, `run_analysis.py`, `compare_previous.py`, `make_figures.py`, `build_report.py`; library modules `design.py`, `lmm.py`, `selection.py`, `model_tools.py`, `report_text.py`.
* `results/model_results.json` — every result; `results/tables/T00–T24` — CSV tables; `results/figures/F01–F11` — PNG (300 dpi) and vector PDF.
* `report/index.html` — interactive report; `report/Model_Summary.pdf` — printable summary; this file.
"""
    with open(os.path.join(REP, "MODEL_REPORT.md"), "w") as f:
        f.write(md)
    print("wrote report/MODEL_REPORT.md")


def build_readme(R, K):
    txt = f"""# Paired 7/28-day compressive-strength model (RCF · SS · A/B · carbonation)

Reproducible analysis of the corrected 7- and 28-day compressive strengths of 30 mixtures. One linear mixed
model covers both curing ages: curing age is a two-level factor, carbonation a four-level factor (NC, 0.5, 1, 5 h),
and the paired 7/28-day results of each mixture share a random mixture effect with age-specific residual variance.

**Data update.** The analysis now uses the updated corrected results
(`data/source/30_mixes_7_and_28_days_strength_results.pdf`), which change the 28-day strength of {K['v1_28_n']} mixes and the
7-day strength of {K['v1_7_txt']} ({K['v1_n']} entries in all, `results/tables/T00b_changes_vs_previous_corrected_data.csv`).
The whole protocol was re-run unchanged. {'It selects the same model as before.' if K['same_model'] else 'It selects a different model (see the report).'}
Previous and current results side by side: `results/tables/T24_comparison_with_previous_data.csv` and the "What changed" section of the report.

## Headline results

* Final model (exhaustive AICc search over {K['n_models']} hierarchical models, both covariance structures):
  ln f = SS + SS² + RCF + carbonation (4 levels) + RCF·SS + K·RCF + K·SS **+ Age × (1 + SS + SS²)**,
  where K = carbonated RCF. A/B has no detectable effect.
* **In the selected model, curing age interacts with SS only.** f28/f7 = {K['g0']} (95 % CI {K['g0_ci']}) at SS 0 %, minimum {K['gmin']} near SS {K['gmin_ss']} %,
  {K['g75']} ({K['g75_ci']}) at SS 75 %.
* Carbonated RCF raises strength in NaOH-rich mixes and lowers it in silicate-rich mixes; RCF raises strength more when carbonated
  and when SS is low. The three carbonation durations are not distinguishable ({K['p_dur']}).
* 7-day strength of a left-out mixture: predicted R² {K['pr7']}, RMSE {K['rm7']} MPa (replicate scatter {K['pe7M']} MPa).
  28-day: predicted R² {K['pr28']}, RMSE {K['rm28']} MPa. 28-day results scatter more than the mixture variables explain.

Full write-up: [`report/MODEL_REPORT.md`](report/MODEL_REPORT.md) · printable summary: [`report/Model_Summary.pdf`](report/Model_Summary.pdf) ·
interactive report: [`report/index.html`](report/index.html) (open in a browser; it loads D3 from cdnjs).

## Layout

| Path | Content |
|---|---|
| `data/strength_7d_28d_corrected.csv` | corrected data set (controlling), one row per mixture |
| `data/strength_long.csv` | long format, one row per mixture and age |
| `data/previous_corrected_v1.csv` | previous corrected data set (superseded; for the change log and the comparison only) |
| `data/previous_7d_run_sheet.csv` | run sheet used by the earlier 7-day model (for the change log only) |
| `data/source/` | corrected data as supplied (PDF); `strength_7d_28d_corrected.csv` is transcribed from `30_mixes_7_and_28_days_strength_results.pdf`; `superseded/` holds the previous sheet |
| `reference/previous_7day_model/` | earlier 7-day model report (HTML, PDF), used as a methodological reference |
| `analysis/prepare_data.py` | long format, change logs (T00, T00b), replicate error |
| `analysis/compare_previous.py` | previous vs current results of the same protocol (T24); `results/previous_v1/` holds the previous key results |
| `analysis/run_analysis.py` | the protocol: scale, covariance, coding, exhaustive selection, validation, resampling, robustness, contrasts |
| `analysis/lmm.py` | bivariate REML/GLS mixed model with Satterthwaite df |
| `analysis/selection.py` | hierarchy bookkeeping, exhaustive AICc search (exact CS/UN factorisation), backward elimination |
| `analysis/design.py`, `analysis/model_tools.py` | factor coding, design matrices, predictions, contrasts, actual-unit equations |
| `analysis/make_figures.py` | figures F01–F11 (PNG 300 dpi + PDF) |
| `analysis/build_report.py`, `analysis/report_text.py` | reports; every number is read from `results/model_results.json` |
| `results/model_results.json` | all results |
| `results/tables/` | T00–T24 CSV tables |
| `results/figures/` | F01–F11 |
| `report/` | `index.html`, `Model_Summary.pdf`, `MODEL_REPORT.md` |

## Reproduce

```bash
pip install -r requirements.txt
python analysis/prepare_data.py
python analysis/run_analysis.py          # about {R['runtime_s'] / 60:.0f} min on 4 cores ({R['meta']['n_boot']:,} resamples per scheme); --quick for a test run
python analysis/compare_previous.py      # previous vs current data (T24)
python analysis/make_figures.py
python analysis/build_report.py          # needs Chromium for the PDF
```

Seed {R['meta']['seed']}. Tested with Python 3.11, NumPy {R['meta']['numpy']}, pandas {R['meta']['pandas']}.

## Separate 7-day and 28-day models

`analysis/run_separate.py` builds two independent models, one per curing age, each from its own 30 results with the same protocol
(Box–Cox scale, the same 16-parameter candidate set, exhaustive AICc over 716 hierarchical models, identical validation and resampling).
Outputs: `results/separate/` (JSON, tables `S7_*`, `S28_*`, figures `H01–H10`) and `report/separate/`
(`index.html`, `Separate_Models_Summary.pdf`, `SEPARATE_MODELS.md`). Run `python analysis/run_separate.py`,
`python analysis/make_figures_separate.py`, `python analysis/report_separate.py`.

## SHAP explanations

`analysis/run_shap.py` computes exact interventional Shapley (SHAP) values and SHAP interaction values for the separate
7-day and 28-day models and the paired model, over all coalitions of the actual model inputs (RCF, SS, A/B, carbonation,
plus curing age in the paired model), with the 30 mixtures as background. Carbonation is one categorical input with
four levels (NC, 0.5 h, 1 h, 5 h). Importance intervals come from 1,000 stratified case-bootstrap refits.
Outputs: `results/shap/` (JSON, tables, figures `Z01–Z08b`) and `report/shap/` (`index.html`, `SHAP_Figures.pdf`,
`SHAP_NOTES.md`). Run `python analysis/run_shap.py`, `python analysis/make_figures_shap.py`,
`python analysis/report_shap.py`. The optional cross-check against the `shap` package (0.51.0) runs only when it is installed.

## Relation to the earlier 7-day model

The earlier 7-day model (numeric carbonation, 7-day data only) was used as a methodological and presentation reference.
The new model was selected independently from the combined data. It arrives at the same pattern of 7-day terms
(RCF, SS, SS², carbonation, RCF·SS, carbonation·RCF, carbonation·SS) with categorical carbonation, and adds the SS-dependent age effect.
The corrected data set differs from the earlier run sheet in {R['data']['n_changes']} entries (`results/tables/T00_changes_vs_previous_7d_data.csv`).
"""
    with open(os.path.join(dz.ROOT, "README.md"), "w") as f:
        f.write(txt)
    print("wrote README.md")


def rt_ss(K, age, carb, rcf):
    return f"SS {K['sso'][(age, carb, rcf)].SS_opt:.0f} %"


# ============================================================== printable summary (PDF)
def build_summary(R, K, findings, coded_eq, eq7, eq28, eqg, tf_tab, ag_tab, opt_tab, stat_strip):
    figdir = "../results/figures"   # relative to report/summary.html
    cards = "".join(f'<div class="fcard"><b>{E(l)}</b> · <b>{t}</b><br>{b}</div>' for l, t, b in findings)
    html_ = f"""<!doctype html><html><head><meta charset="utf-8"><title>Paired 7/28-Day Strength Model</title>
<style>{SUMMARY_CSS}</style></head><body>
<h1>Paired 7/28-day compressive-strength model</h1>
<p class="sub">30 mixtures × 2 curing ages · linear mixed model (random mixture intercept, age-specific residual variance) · corrected data set · generated {R['meta']['date']}</p>
<h2>Findings</h2>{cards}
<h2>Final model</h2>
<p class="eq">{coded_eq}</p>
<p class="eq">{eq7}</p><p class="eq">{eq28}</p><p class="eq">{eqg}</p>
{stat_strip}
{tf_tab}
<p class="small">Wald F tests, Satterthwaite df. Covariance UN: SD {K['sd7']} (7 d), {K['sd28']} (28 d), ρ = {K['rho']}. Lack of fit: gain {K['lof_gain']}, level|gain {K['lof_level']}. Box–Cox (final): {K['bc_final']}. Exhaustive search of {K['n_models']} models; ΔAICc to best CS model {K['daicc_struct']}.</p>
<div class="fig"><img src="{figdir}/F03_gain_ratio_vs_SS.png"></div>
<div class="fig"><img src="{figdir}/F05_strength_vs_SS_by_age.png"></div>
<h2>Age-specific effects</h2>{ag_tab}
<div class="fig"><img src="{figdir}/F04_age_specific_effects.png"></div>
<h2>Optima</h2>{opt_tab}
<div class="fig"><img src="{figdir}/F06_response_surfaces.png"></div>
<h2>Validation and robustness</h2>
<p>Leave-one-mixture-out predicted R²: {K['pr7']} (7 d), {K['pr28']} (28 d); RMSE {K['rm7']} / {K['rm28']} MPa. Nested CV of the selection procedure: {K['ncv_primary7']} / {K['ncv_primary28']}. Residual-bootstrap re-selection keeps SS as the only age-dependent variable in {K['resid_gainSS']} of resamples. Shapiro–Wilk {K['sw_p']}; Breusch–Pagan {K['bp_p']}; largest |t| {K['max_t']} (mix {K['max_t_mix']}, {K['max_t_age']} d; Bonferroni {K['max_t_bonf']}).</p>
<div class="fig"><img src="{figdir}/F08_observed_vs_predicted.png"></div>
<div class="fig"><img src="{figdir}/F09_residual_diagnostics.png"></div>
<div class="fig"><img src="{figdir}/F02_scale_and_selection.png"></div>
<div class="fig"><img src="{figdir}/F10_robustness.png"></div>
</body></html>"""
    p = os.path.join(REP, "summary.html")
    with open(p, "w") as f:
        f.write(html_)
    chrome = None
    for c in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome", shutil.which("chromium") or "", shutil.which("chromium-browser") or ""):
        if c and os.path.exists(c):
            chrome = c
            break
    if chrome is None:
        import glob
        g = glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome")
        chrome = g[0] if g else None
    if chrome:
        out = os.path.join(REP, "Model_Summary.pdf")
        subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", "file://" + p], check=False, capture_output=True, timeout=180)
        print("wrote report/Model_Summary.pdf" if os.path.exists(out) else "PDF not written")
    else:
        print("Chromium not found; summary.html written, PDF skipped")


SUMMARY_CSS = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 8.6pt; color: #16191d; line-height: 1.38; }
h1 { font-size: 16pt; margin: 0 0 2px; } h2 { font-size: 11pt; margin: 14px 0 5px; border-bottom: 1px solid #c3c8cf; padding-bottom: 2px; }
.sub { color: #4b5058; margin: 0 0 8px; } .small { font-size: 7.6pt; color: #4b5058; }
.fcard { border-left: 3px solid #2a78d6; padding: 3px 8px; margin: 4px 0; background: #f4f6f8; break-inside: avoid; }
.eq { font-family: "DejaVu Sans Mono", monospace; font-size: 7.6pt; background: #f4f6f8; padding: 4px 6px; margin: 3px 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.3pt; margin: 6px 0; break-inside: avoid; }
th, td { border-bottom: 1px solid #dde1e6; padding: 2px 4px; text-align: left; } th { background: #eef1f4; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.fig { break-inside: avoid; margin: 6px 0; } .fig img { width: 100%; }
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 6px 0; }
.strip div { background: #f4f6f8; padding: 4px 6px; } .strip .lab { display: block; font-size: 7pt; color: #4b5058; }
.strip .val { display: block; font-size: 11pt; font-weight: bold; } .strip .sub { font-size: 6.8pt; margin: 0; }
.chip { font-size: 7pt; } .tw { overflow: visible; }
caption { text-align: left; }
"""

CSS = r"""
/* Layout: one reading column (~72ch) with wider figure/table blocks; lab-report tone, cool concrete neutrals. */
:root {
  --bg: #f4f5f6; --surface: #ffffff; --ink: #15181c; --ink2: #474d55; --muted: #767d86; --rule: #d9dde2; --grid: #e4e7eb; --axis: #b9bfc7;
  --accent: #1c5cab; --accent-soft: #e3edf9; --code: #eceff2;
  --age7: #2a78d6; --age28: #eb6834; --nc: #3d3c39; --c05: #008300; --c1: #4a3aa7; --c5: #e87ba4; --star: #eda100;
  --good: #0c7a0c; --good-bg: #e2f3e2; --mod: #8a5a00; --mod-bg: #fbefd5; --sugg: #5b4bb8; --sugg-bg: #ebe8fa; --none: #5d636b; --none-bg: #eceef0;
  --seq0: #cde2fb; --seq1: #0d366b;
  --f-display: "Archivo", "Arial Narrow", system-ui, sans-serif; --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #101214; --surface: #181b1e; --ink: #edeff1; --ink2: #b8bdc4; --muted: #8b929b; --rule: #2b2f34; --grid: #262a2e; --axis: #3b4047;
  --accent: #86b6ef; --accent-soft: #16273d; --code: #20242a;
  --age7: #3987e5; --age28: #d95926; --nc: #c3c2b7; --c05: #008300; --c1: #9085e9; --c5: #d55181; --star: #eda100;
  --good: #5fd35f; --good-bg: #173017; --mod: #f0b64a; --mod-bg: #33280f; --sugg: #b3a9f5; --sugg-bg: #25213f; --none: #a4aab2; --none-bg: #24282c;
  --seq0: #0f2340; --seq1: #cde2fb; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #101214; --surface: #181b1e; --ink: #edeff1; --ink2: #b8bdc4; --muted: #8b929b; --rule: #2b2f34; --grid: #262a2e; --axis: #3b4047;
  --accent: #86b6ef; --accent-soft: #16273d; --code: #20242a;
  --age7: #3987e5; --age28: #d95926; --nc: #c3c2b7; --c05: #008300; --c1: #9085e9; --c5: #d55181; --star: #eda100;
  --good: #5fd35f; --good-bg: #173017; --mod: #f0b64a; --mod-bg: #33280f; --sugg: #b3a9f5; --sugg-bg: #25213f; --none: #a4aab2; --none-bg: #24282c;
  --seq0: #0f2340; --seq1: #cde2fb; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--f-body); }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; }
header.top { max-width: 820px; padding-block: 12px 18px; }
.eyebrow { font: 500 12px/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
h1 { font: 800 clamp(30px, 5vw, 46px)/1.05 var(--f-display); font-stretch: 85%; letter-spacing: -.01em; margin: 0 0 14px; text-wrap: balance; }
h2 { font: 700 26px/1.2 var(--f-display); font-stretch: 90%; margin: 0 0 12px; text-wrap: balance; }
h3 { font: 650 17px/1.3 var(--f-display); margin: 26px 0 8px; text-wrap: balance; }
.lede { font-size: 17.5px; color: var(--ink2); max-width: 68ch; margin: 0 0 12px; }
.meta { font: 12.5px/1.5 var(--f-mono); color: var(--muted); margin: 0 0 16px; }
code, .eq, pre { font-family: var(--f-mono); }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: .88em; }
.toc { display: flex; flex-wrap: wrap; gap: 6px 16px; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); padding-block: 9px; }
.toc a { color: var(--ink2); text-decoration: none; font: 500 13.5px var(--f-body); }
.toc a:hover, .toc a:focus-visible { color: var(--accent); text-decoration: underline; }
main section { padding-block: 34px 8px; border-bottom: 1px solid var(--rule); }
main p { max-width: 74ch; }
.note { font-size: 13.5px; color: var(--ink2); }
a { color: var(--accent); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; margin: 16px 0 20px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; min-width: 0; }
.card h3 { margin: 10px 0 6px; font-size: 16.5px; }
.card p { font-size: 14px; color: var(--ink2); margin: 0; }
.chip { display: inline-block; font: 600 11.5px/1 var(--f-mono); letter-spacing: .04em; text-transform: uppercase; padding: 5px 8px; border-radius: 999px; white-space: nowrap; }
.chip.robust { color: var(--good); background: var(--good-bg); }
.chip.moderate { color: var(--mod); background: var(--mod-bg); }
.chip.suggestive { color: var(--sugg); background: var(--sugg-bg); }
.chip.not, .chip.retained, .chip.design { color: var(--none); background: var(--none-bg); }
.callout { background: var(--accent-soft); border-radius: 10px; padding: 14px 18px; margin: 8px 0 16px; }
.callout h3 { margin-top: 2px; } .callout p { margin: 0; font-size: 14.5px; }
.eq { background: var(--code); border-radius: 8px; padding: 10px 14px; font-size: 13px; line-height: 1.7; overflow-x: auto; max-width: 100%; }
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 18px 0; }
.strip > div { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.strip .lab { font: 500 12px var(--f-mono); color: var(--muted); letter-spacing: .03em; }
.strip .val { font: 700 26px/1.1 var(--f-display); }
.strip .sub { font-size: 12.5px; color: var(--ink2); }
.tw { overflow-x: auto; margin: 10px 0 8px; max-width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--surface); }
table.small { font-size: 12.5px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 12px/1.3 var(--f-body); color: var(--ink2); background: var(--surface); position: sticky; top: 0; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--f-mono); font-size: .92em; white-space: nowrap; }
details { margin: 10px 0; } summary { cursor: pointer; color: var(--accent); font-weight: 500; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
figure { margin: 18px 0; }
figure.plate { background: #ffffff; border: 1px solid var(--rule); border-radius: 10px; padding: 10px; }
figure.plate img { display: block; width: 100%; height: auto; }
figure.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 12px 8px; min-width: 0; }
figcaption { font-size: 13px; color: var(--ink2); margin-top: 6px; }
.plot { width: 100%; min-height: 260px; }
.plot svg { display: block; width: 100%; height: auto; overflow: visible; }
.plot text { fill: var(--ink2); font: 11.5px var(--f-body); }
.plot .axis path, .plot .axis line { stroke: var(--axis); }
.plot .grid line { stroke: var(--grid); }
.plot .grid path { stroke: none; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin: 12px 0 4px; }
.controls label { font-weight: 500; font-size: 14px; }
input[type=range] { width: min(260px, 70vw); accent-color: var(--accent); }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: 500 13px var(--f-body); color: var(--ink2); background: transparent; border: 0; border-right: 1px solid var(--rule); padding: 7px 12px; cursor: pointer; }
.seg button:last-child { border-right: 0; }
.seg button[aria-checked="true"] { background: var(--accent-soft); color: var(--ink); font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.split { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 820px) { .split { grid-template-columns: 1fr; } }
.predictor { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; margin: 18px 0; }
.predictor h3 { margin-top: 0; }
.field { display: grid; grid-template-columns: 110px 1fr; align-items: center; gap: 10px; margin: 8px 0; }
.field input, .field select { font: 14px var(--f-body); padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; background: var(--bg); color: var(--ink); width: 100%; min-width: 0; }
.field input:focus-visible, .field select:focus-visible { outline: 2px solid var(--accent); }
.pout { display: grid; gap: 10px; margin-top: 10px; }
.pout .row { border-top: 1px solid var(--rule); padding-top: 8px; }
.pout .big { font: 700 24px/1.1 var(--f-display); }
.pout .k { font: 500 12px var(--f-mono); color: var(--muted); display: flex; align-items: center; gap: 6px; }
.pout .sw { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
.pout .d { font-size: 12.5px; color: var(--ink2); }
ol.proto { padding-left: 22px; max-width: 78ch; } ol.proto li { margin: 6px 0; }
pre.code { background: var(--code); border-radius: 8px; padding: 12px 14px; overflow-x: auto; font-size: 13px; }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--rule); border-radius: 8px; padding: 8px 10px; font: 12.5px/1.45 var(--f-body); box-shadow: 0 4px 14px rgba(0,0,0,.12); max-width: 260px; }
.tip b { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--ink2); margin: 2px 0 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.plot .legend svg { flex: none; display: inline-block; width: 16px; height: 12px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

JS = r"""
(function(){
'use strict';
const D = __DATA__;
const CARBS = ['NC','0.5 h','1 h','5 h'];
const SHAPE = {'NC': d3.symbolCircle, '0.5 h': d3.symbolTriangle, '1 h': d3.symbolSquare, '5 h': d3.symbolDiamond};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = () => ({age7: css('--age7'), age28: css('--age28'), 'NC': css('--nc'), '0.5 h': css('--c05'), '1 h': css('--c1'), '5 h': css('--c5'),
  ink: css('--ink'), ink2: css('--ink2'), muted: css('--muted'), grid: css('--grid'), axis: css('--axis'), surface: css('--surface'), star: css('--star'),
  seq0: css('--seq0'), seq1: css('--seq1')});
const tip = document.getElementById('tip');
function showTip(ev, html){ tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect(); let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
function hideTip(){ tip.hidden = true; }
const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);

// ---------------------------------------------------------------- model
function row(p){
  const A = (p.RCF - 30) / 20, C = (p.AB - 0.45) / 0.03, Dd = (p.SS - 37.5) / 37.5, K = p.carb === 'NC' ? 0 : 1, g = p.age === 28 ? 0.5 : -0.5;
  const base = {Intercept: 1, A: A, C: C, D: Dd, AC: A*C, AD: A*Dd, CD: C*Dd, A2: A*A, C2: C*C, D2: Dd*Dd, KA: K*A, KC: K*C, KD: K*Dd};
  return D.names.map(nm => {
    if (nm === 'Age') return g;
    let age = false, b = nm;
    if (nm.endsWith('.Age')) { age = true; b = nm.slice(0, -4); }
    let v = b.startsWith('Carb[') ? (p.carb === b.slice(5, -1) ? 1 : 0) : base[b];
    return age ? v * g : v;
  });
}
function lin(x){ let e = 0; for (let j = 0; j < x.length; j++) e += x[j] * D.beta[j];
  let v = 0; for (let i = 0; i < x.length; i++) for (let j = 0; j < x.length; j++) v += x[i] * D.V[i][j] * x[j];
  return [e, Math.sqrt(Math.max(v, 0))]; }
function predict(p){ const [e, se] = lin(row(p)); const s2 = D.s2[p.age === 28 ? 1 : 0]; const t = D.t[String(p.age)];
  const sp = Math.sqrt(se*se + s2);
  return {med: Math.exp(e), lo: Math.exp(e - t*se), hi: Math.exp(e + t*se), plo: Math.exp(e - t*sp), phi: Math.exp(e + t*sp), ln: e}; }
function gainAt(p){ const x7 = row({...p, age: 7}), x28 = row({...p, age: 28}); const x = x28.map((v, j) => v - x7[j]);
  const [e, se] = lin(x); const t = D.t.gain; return {r: Math.exp(e), lo: Math.exp(e - t*se), hi: Math.exp(e + t*se)}; }

// ---------------------------------------------------------------- helpers
function frame(el, h, m){ el.innerHTML = ''; const w = Math.max(300, el.clientWidth);
  const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w).attr('height', h);
  return {svg, w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b, g: svg.append('g').attr('transform', `translate(${m.l},${m.t})`)}; }
function axes(F, x, y, xl, yl, yfmt){
  F.g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(6).tickSize(-F.iw).tickFormat(''));
  F.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(Math.min(8, Math.floor(F.iw / 70))));
  const ya = d3.axisLeft(y).ticks(6); if (yfmt) ya.tickFormat(yfmt);
  F.g.append('g').attr('class', 'axis').call(ya);
  F.g.append('text').attr('x', F.iw / 2).attr('y', F.ih + 36).attr('text-anchor', 'middle').text(xl);
  F.g.append('text').attr('transform', 'rotate(-90)').attr('x', -F.ih / 2).attr('y', -F.m.l + 14).attr('text-anchor', 'middle').text(yl);
}
function legend(el, items){ const L = document.createElement('div'); L.className = 'legend';
  L.innerHTML = items.map(it => `<span><svg width="16" height="12" aria-hidden="true">${it.sym}</svg>${it.label}</span>`).join('');
  el.prepend(L); }
function symSVG(carb, color){ return `<path d="${d3.symbol(SHAPE[carb], 46)()}" transform="translate(8,6)" fill="${color}"/>`; }
function lineSVG(color, w){ return `<line x1="1" x2="15" y1="6" y2="6" stroke="${color}" stroke-width="${w||2}" stroke-linecap="round"/>`; }
function seg(el, opts, val, onChange){ el.innerHTML = ''; opts.forEach(o => { const b = document.createElement('button');
  b.type = 'button'; b.setAttribute('role', 'radio'); b.textContent = o.label; b.setAttribute('aria-checked', o.value === val ? 'true' : 'false');
  b.addEventListener('click', () => { el.querySelectorAll('button').forEach(x => x.setAttribute('aria-checked', 'false')); b.setAttribute('aria-checked', 'true'); onChange(o.value); });
  el.appendChild(b); }); }

// ---------------------------------------------------------------- gain chart
function drawGain(){
  const el = document.getElementById('gainChart'), c = COL();
  const F = frame(el, 330, {l: 54, r: 18, t: 12, b: 46});
  const x = d3.scaleLinear().domain([0, 75]).range([0, F.iw]);
  const gmin = d3.min(D.runs, r => r.f28 / r.f7), gmax = d3.max(D.runs, r => r.f28 / r.f7);
  const y = d3.scaleLog().domain([Math.min(0.95, gmin * 0.93), Math.max(4.6, gmax * 1.08)]).range([F.ih, 0]);
  axes(F, x, y, 'Sodium silicate, SS (%)', 'Gain  f28 / f7', d3.format('.1f'));
  const G = D.gain, n = G.SS.length, idx = d3.range(n);
  F.g.append('path').attr('fill', c.age28).attr('opacity', 0.09).attr('d', d3.area().x(i => x(G.SS[i])).y0(i => y(G.blo[i])).y1(i => y(G.bhi[i]))(idx));
  F.g.append('path').attr('fill', c.age28).attr('opacity', 0.2).attr('d', d3.area().x(i => x(G.SS[i])).y0(i => y(G.lo[i])).y1(i => y(G.hi[i]))(idx));
  F.g.append('path').attr('fill', 'none').attr('stroke', c.age28).attr('stroke-width', 2.2).attr('d', d3.line().x(i => x(G.SS[i])).y(i => y(G.est[i]))(idx));
  F.g.append('line').attr('x1', 0).attr('x2', F.iw).attr('y1', y(1)).attr('y2', y(1)).attr('stroke', c.axis);
  F.g.selectAll('path.pt').data(D.runs).join('path').attr('class', 'pt')
    .attr('d', r => d3.symbol(SHAPE[r.carb], 58)()).attr('transform', r => `translate(${x(r.SS)},${y(r.f28 / r.f7)})`)
    .attr('fill', r => c[r.carb]).attr('stroke', c.surface).attr('stroke-width', 1.5).style('cursor', 'default')
    .on('mousemove', (ev, r) => showTip(ev, `<b>Mix ${r.mix}</b> · ${r.carb}<br>RCF ${f1(r.RCF)} %, SS ${f1(r.SS)} %, A/B ${r.AB.toFixed(3)}<br>7 d ${f2(r.f7)} → 28 d ${f2(r.f28)} MPa<br>gain ×${f2(r.f28 / r.f7)} (model ×${f2(gainAt({RCF: r.RCF, SS: r.SS, AB: r.AB, carb: r.carb}).r)})`))
    .on('mouseleave', hideTip);
  legend(el, [{sym: lineSVG(c.age28, 2.4), label: 'Model (95 % CI, bootstrap band)'}].concat(CARBS.map(k => ({sym: symSVG(k, c[k]), label: k}))));
}

// ---------------------------------------------------------------- curves
let curveRCF = 30, curveCarb = '1 h';
function drawCurves(){
  const el = document.getElementById('curveChart'), c = COL();
  const F = frame(el, 340, {l: 54, r: 18, t: 12, b: 46});
  const x = d3.scaleLinear().domain([0, 75]).range([0, F.iw]);
  const ss = d3.range(0, 75.01, 0.5);
  const data = {7: ss.map(s => predict({RCF: curveRCF, SS: s, AB: 0.45, carb: curveCarb, age: 7})), 28: ss.map(s => predict({RCF: curveRCF, SS: s, AB: 0.45, carb: curveCarb, age: 28}))};
  const near = D.runs.filter(r => r.carb === curveCarb && Math.abs(r.RCF - curveRCF) <= 10);
  const ymax = Math.max(45, d3.max(data[28], d => d.hi) * 1.05, d3.max(near, r => r.f28) || 0);
  const y = d3.scaleLinear().domain([0, ymax]).nice().range([F.ih, 0]);
  axes(F, x, y, 'Sodium silicate, SS (%)', 'Strength (MPa, median)');
  [[7, c.age7], [28, c.age28]].forEach(([a, col]) => {
    const P = data[a];
    F.g.append('path').attr('fill', col).attr('opacity', 0.14).attr('d', d3.area().x((d, i) => x(ss[i])).y0(d => y(d.lo)).y1(d => y(d.hi))(P));
    F.g.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.2).attr('d', d3.line().x((d, i) => x(ss[i])).y(d => y(d.med))(P));
    const k = d3.maxIndex(P, d => d.med);
    F.g.append('circle').attr('cx', x(ss[k])).attr('cy', y(P[k].med)).attr('r', 5).attr('fill', c.surface).attr('stroke', col).attr('stroke-width', 2);
  });
  near.forEach(r => { [[7, c.age7, r.f7], [28, c.age28, r.f28]].forEach(([a, col, v]) => {
    F.g.append('path').attr('d', d3.symbol(a === 7 ? d3.symbolCircle : d3.symbolSquare, 42)()).attr('transform', `translate(${x(r.SS)},${y(v)})`)
      .attr('fill', col).attr('stroke', c.surface).attr('stroke-width', 1.5)
      .on('mousemove', ev => showTip(ev, `<b>Mix ${r.mix}</b> · ${r.carb}, RCF ${f1(r.RCF)} %<br>SS ${f1(r.SS)} % · ${a} d: ${f2(v)} MPa`)).on('mouseleave', hideTip); }); });
  const cross = F.g.append('g').style('display', 'none');
  cross.append('line').attr('y1', 0).attr('y2', F.ih).attr('stroke', c.muted);
  const d7 = cross.append('circle').attr('r', 4).attr('fill', c.age7), d28 = cross.append('circle').attr('r', 4).attr('fill', c.age28);
  F.g.append('rect').attr('width', F.iw).attr('height', F.ih).attr('fill', 'transparent')
    .on('mousemove', ev => { const [mx] = d3.pointer(ev); const s = Math.max(0, Math.min(75, x.invert(mx))); const i = Math.round(s / 0.5);
      const a = data[7][i], b = data[28][i]; cross.style('display', null); cross.select('line').attr('x1', x(ss[i])).attr('x2', x(ss[i]));
      d7.attr('cx', x(ss[i])).attr('cy', y(a.med)); d28.attr('cx', x(ss[i])).attr('cy', y(b.med));
      showTip(ev, `SS ${f1(ss[i])} % · RCF ${curveRCF} % · ${curveCarb}<br><b>7 d</b> ${f1(a.med)} MPa (CI ${f1(a.lo)}–${f1(a.hi)})<br><b>28 d</b> ${f1(b.med)} MPa (CI ${f1(b.lo)}–${f1(b.hi)})<br>gain ×${f2(b.med / a.med)}`); })
    .on('mouseleave', () => { cross.style('display', 'none'); hideTip(); });
  F.svg.node().parentNode.querySelectorAll('.pt').forEach(n => n.parentNode.appendChild(n));
  legend(el, [{sym: lineSVG(c.age7), label: '7 d'}, {sym: lineSVG(c.age28), label: '28 d'}, {sym: `<circle cx="8" cy="6" r="4" fill="none" stroke="${c.ink2}" stroke-width="1.6"/>`, label: 'peak'}]);
}

// ---------------------------------------------------------------- map
let mapAge = 28, mapCarb = '1 h';
function drawMap(){
  const el = document.getElementById('mapChart'), c = COL();
  const F = frame(el, 360, {l: 54, r: 62, t: 12, b: 46});
  const x = d3.scaleLinear().domain([10, 50]).range([0, F.iw]);
  const y = d3.scaleLinear().domain([0, 75]).range([F.ih, 0]);
  const nx = 41, ny = 61, vals = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const R_ = 10 + 40 * i / (nx - 1), S_ = 75 * j / (ny - 1);
    vals.push(predict({RCF: R_, SS: S_, AB: 0.45, carb: mapCarb, age: mapAge}).med); }
  const color = d3.scaleSequential(d3.interpolateRgb(c.seq0, c.seq1)).domain([0, 34]);
  const cw = F.iw / (nx - 1), chh = F.ih / (ny - 1);
  const cells = F.g.append('g');
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) cells.append('rect').attr('x', x(10 + 40 * i / (nx - 1)) - cw / 2).attr('y', y(75 * j / (ny - 1)) - chh / 2)
    .attr('width', cw + 0.6).attr('height', chh + 0.6).attr('fill', color(Math.min(34, vals[j * nx + i])));
  const cont = d3.contours().size([nx, ny]).thresholds(d3.range(4, 40, 4))(vals);
  const tx = d3.scaleLinear().domain([0, nx - 1]).range([0, F.iw]), ty = d3.scaleLinear().domain([0, ny - 1]).range([F.ih, 0]);
  F.g.append('g').selectAll('path').data(cont).join('path').attr('d', d3.geoPath(d3.geoTransform({point: function(px, py){ this.stream.point(tx(px - 0.5), ty(py - 0.5)); }})))
    .attr('fill', 'none').attr('stroke', c.surface).attr('stroke-opacity', 0.55).attr('stroke-width', 0.8);
  F.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(5));
  F.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(6));
  F.g.append('text').attr('x', F.iw / 2).attr('y', F.ih + 36).attr('text-anchor', 'middle').text('RCF (%)');
  F.g.append('text').attr('transform', 'rotate(-90)').attr('x', -F.ih / 2).attr('y', -40).attr('text-anchor', 'middle').text('SS (%)');
  D.runs.filter(r => r.carb === mapCarb).forEach(r => F.g.append('circle').attr('cx', x(r.RCF)).attr('cy', y(r.SS)).attr('r', 4.2)
    .attr('fill', c.surface).attr('stroke', c.ink).attr('stroke-width', 1.1)
    .on('mousemove', ev => showTip(ev, `<b>Mix ${r.mix}</b> (observed)<br>RCF ${f1(r.RCF)} %, SS ${f1(r.SS)} %<br>${mapAge} d: ${f2(mapAge === 7 ? r.f7 : r.f28)} MPa`)).on('mouseleave', hideTip));
  const o = D.optimum.find(o => o.age === mapAge && o.carbonation === mapCarb);
  F.g.append('path').attr('d', d3.symbol(d3.symbolStar, 150)()).attr('transform', `translate(${x(o.RCF)},${y(o.SS)})`).attr('fill', c.star).attr('stroke', c.ink).attr('stroke-width', 0.8);
  F.g.append('rect').attr('width', F.iw).attr('height', F.ih).attr('fill', 'transparent').style('cursor', 'crosshair')
    .on('mousemove', ev => { const [mx, my] = d3.pointer(ev); const R_ = Math.round(x.invert(mx) * 2) / 2, S_ = Math.round(y.invert(my) * 2) / 2;
      const p = predict({RCF: R_, SS: S_, AB: 0.45, carb: mapCarb, age: mapAge});
      showTip(ev, `RCF ${f1(R_)} %, SS ${f1(S_)} % · ${mapCarb} · ${mapAge} d<br><b>${f1(p.med)} MPa</b> (CI ${f1(p.lo)}–${f1(p.hi)})<br>click to load into the predictor`); })
    .on('mouseleave', hideTip)
    .on('click', ev => { const [mx, my] = d3.pointer(ev); document.getElementById('pRCF').value = Math.round(x.invert(mx) * 2) / 2;
      document.getElementById('pSS').value = Math.round(y.invert(my) * 2) / 2; document.getElementById('pCarb').value = mapCarb; updatePred(); });
  // colour legend
  const lg = F.svg.append('g').attr('transform', `translate(${F.m.l + F.iw + 18},${F.m.t})`);
  const ly = d3.scaleLinear().domain([0, 34]).range([F.ih, 0]);
  d3.range(0, 34, 0.5).forEach(v => lg.append('rect').attr('x', 0).attr('y', ly(v + 0.5)).attr('width', 12).attr('height', ly(v) - ly(v + 0.5) + 0.5).attr('fill', color(v)));
  lg.append('g').attr('class', 'axis').attr('transform', 'translate(12,0)').call(d3.axisRight(ly).ticks(5));
  lg.append('text').attr('transform', 'rotate(90)').attr('x', F.ih / 2).attr('y', -46).attr('text-anchor', 'middle').text('MPa');
}

// ---------------------------------------------------------------- observed vs predicted
let opMode = 'fit';
function drawOP(){
  const el = document.getElementById('opChart'), c = COL();
  const F = frame(el, 360, {l: 54, r: 18, t: 12, b: 46});
  const x = d3.scaleLog().domain([2.5, 45]).range([0, F.iw]), y = d3.scaleLog().domain([2.5, 45]).range([F.ih, 0]);
  const tf = d3.format('~g');
  F.g.append('g').attr('class', 'grid').call(d3.axisLeft(y).tickValues([3, 5, 10, 20, 40]).tickSize(-F.iw).tickFormat(''));
  F.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues([3, 5, 10, 20, 40]).tickFormat(tf));
  F.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickValues([3, 5, 10, 20, 40]).tickFormat(tf));
  F.g.append('text').attr('x', F.iw / 2).attr('y', F.ih + 36).attr('text-anchor', 'middle').text('Observed (MPa)');
  F.g.append('text').attr('transform', 'rotate(-90)').attr('x', -F.ih / 2).attr('y', -40).attr('text-anchor', 'middle').text(opMode === 'fit' ? 'Fitted (MPa)' : 'Predicted, mixture left out (MPa)');
  F.g.append('line').attr('x1', x(2.5)).attr('y1', y(2.5)).attr('x2', x(45)).attr('y2', y(45)).attr('stroke', c.axis);
  const k7 = opMode === 'fit' ? 'fit7' : 'cv7', k28 = opMode === 'fit' ? 'fit28' : 'cv28';
  [[7, c.age7, 'f7', k7, d3.symbolCircle], [28, c.age28, 'f28', k28, d3.symbolSquare]].forEach(([a, col, ko, kp, sh]) =>
    F.g.selectAll(null).data(D.runs).join('path').attr('d', d3.symbol(sh, 48)()).attr('transform', r => `translate(${x(r[ko])},${y(r[kp])})`)
      .attr('fill', col).attr('stroke', c.surface).attr('stroke-width', 1.5)
      .on('mousemove', (ev, r) => showTip(ev, `<b>Mix ${r.mix}</b> · ${r.carb} · ${a} d<br>RCF ${f1(r.RCF)} %, SS ${f1(r.SS)} %<br>observed ${f2(r[ko])} · model ${f2(r[kp])} MPa`)).on('mouseleave', hideTip));
  const v = D.val[opMode === 'fit' ? 'fit' : 'mix'];
  F.g.append('text').attr('x', F.iw - 4).attr('y', F.ih - 26).attr('text-anchor', 'end').text(`7 d: R² ${f2(v.predR2_7)}, RMSE ${f1(v.rmse_MPa_7)} MPa`);
  F.g.append('text').attr('x', F.iw - 4).attr('y', F.ih - 10).attr('text-anchor', 'end').text(`28 d: R² ${f2(v.predR2_28)}, RMSE ${f1(v.rmse_MPa_28)} MPa`);
  legend(el, [{sym: `<circle cx="8" cy="6" r="4" fill="${c.age7}"/>`, label: '7 d'}, {sym: `<rect x="4" y="2" width="8" height="8" fill="${c.age28}"/>`, label: '28 d'}]);
}

// ---------------------------------------------------------------- predictor
function updatePred(){
  const c = COL();
  const clamp = (v, a, b) => Math.min(b, Math.max(a, isFinite(v) ? v : a));
  const R_ = clamp(parseFloat(document.getElementById('pRCF').value), 10, 50), S_ = clamp(parseFloat(document.getElementById('pSS').value), 0, 75);
  const carb = document.getElementById('pCarb').value;
  const p7 = predict({RCF: R_, SS: S_, AB: 0.45, carb, age: 7}), p28 = predict({RCF: R_, SS: S_, AB: 0.45, carb, age: 28}), g = gainAt({RCF: R_, SS: S_, AB: 0.45, carb});
  const blk = (lab, col, p) => `<div class="row"><div class="k"><span class="sw" style="background:${col}"></span>${lab}</div><div class="big">${f1(p.med)} MPa</div><div class="d">95 % CI ${f1(p.lo)}–${f1(p.hi)} · one new batch ${f1(p.plo)}–${f1(p.phi)}</div></div>`;
  document.getElementById('pOut').innerHTML = blk('7 days', c.age7, p7) + blk('28 days', c.age28, p28) +
    `<div class="row"><div class="k">Gain f28 / f7</div><div class="big">×${f2(g.r)}</div><div class="d">95 % CI ${f2(g.lo)}–${f2(g.hi)} (depends on SS only)</div></div>`;
}

function drawAll(){ drawGain(); drawCurves(); drawMap(); drawOP(); updatePred(); }
seg(document.getElementById('carbSeg'), CARBS.map(k => ({label: k, value: k})), curveCarb, v => { curveCarb = v; drawCurves(); });
seg(document.getElementById('ageSeg'), [{label: '7 days', value: 7}, {label: '28 days', value: 28}], mapAge, v => { mapAge = v; drawMap(); });
seg(document.getElementById('mapCarbSeg'), CARBS.map(k => ({label: k, value: k})), mapCarb, v => { mapCarb = v; drawMap(); });
seg(document.getElementById('opSeg'), [{label: 'Fitted', value: 'fit'}, {label: 'Mixture left out', value: 'cv'}], opMode, v => { opMode = v; drawOP(); });
const sl = document.getElementById('rcfSlider'); sl.addEventListener('input', () => { curveRCF = +sl.value; document.getElementById('rcfOut').textContent = sl.value; drawCurves(); });
['pRCF', 'pSS', 'pCarb'].forEach(id => document.getElementById(id).addEventListener('input', updatePred));
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
new MutationObserver(drawAll).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
drawAll();
})();
"""


if __name__ == "__main__":
    main()
