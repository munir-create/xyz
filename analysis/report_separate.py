"""
Reports for the two independent age models, from results/separate/separate_results.json:

  report/separate/index.html                 interactive side-by-side report (published as an Artifact)
  report/separate/SEPARATE_MODELS.md         Markdown write-up
  report/separate/Separate_Models_Summary.pdf printable summary (headless Chromium)

Run:  python analysis/report_separate.py
"""
from __future__ import annotations

import glob
import html
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

import design as dz
from build_report import CSS, SUMMARY_CSS, table, md_table, eq_html

RES = os.path.join(dz.ROOT, "results", "separate")
REP = os.path.join(dz.ROOT, "report", "separate")
os.makedirs(os.path.join(REP, "figures"), exist_ok=True)
E = html.escape
AGES = ("7", "28")
EXPECTED = {"7": {"A", "D", "Carb", "AD", "D2", "KA", "KD"}, "28": {"A", "C", "D", "A2"}}

TXT = {"A": "RCF", "C": "A/B", "D": "SS", "Carb": "Carbonation (4 levels)", "AC": "RCF × A/B", "AD": "RCF × SS", "CD": "A/B × SS",
       "A2": "RCF²", "C2": "(A/B)²", "D2": "SS²", "KA": "Carbonated × RCF", "KC": "Carbonated × A/B", "KD": "Carbonated × SS"}
COLTXT = {"Intercept": "Intercept", "A": "RCF (A)", "C": "A/B (C)", "D": "SS (D)", "Carb[0.5 h]": "Carb 0.5 h", "Carb[1 h]": "Carb 1 h",
          "Carb[5 h]": "Carb 5 h", "AC": "RCF·A/B", "AD": "RCF·SS", "CD": "A/B·SS", "A2": "RCF²", "C2": "(A/B)²", "D2": "SS²",
          "KA": "K·RCF", "KC": "K·A/B", "KD": "K·SS"}


def fp(p):
    if p is None or not np.isfinite(p):
        return "–"
    return "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def pe(p):
    return "p " + ("< 0.001" if p is not None and p < 0.001 else "= " + fp(p))


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f1(x): return f"{x:.1f}"
def mn(s): return str(s).replace("-", "−")
def pct(x): return f"{100 * x:.0f} %"


def chip(v):
    return f'<span class="chip {v.split()[0].lower()}">{v}</span>'


def load():
    with open(os.path.join(RES, "separate_results.json")) as f:
        R = json.load(f)
    for a in AGES:
        if set(R[a]["final_terms"]) != EXPECTED[a]:
            raise SystemExit(f"{a}-day model changed ({R[a]['final_terms']}); review report_separate.py text before rebuilding.")
    return R


def ratio_txt(c):
    if not c.get("in_model", True) or c.get("p") is None:
        return "1 (not in model)"
    return f"×{c['ratio']:.2f} [{c['lo']:.2f}–{c['hi']:.2f}], {pe(c['p'])}"


def coded_equation(A):
    coef = pd.DataFrame(A["coef"])
    sym = {"Intercept": "", "A": "A", "C": "C", "D": "D", "Carb[0.5 h]": "C<sub>0.5h</sub>", "Carb[1 h]": "C<sub>1h</sub>",
           "Carb[5 h]": "C<sub>5h</sub>", "AC": "A·C", "AD": "A·D", "CD": "C·D", "A2": "A²", "C2": "C²", "D2": "D²", "KA": "K·A", "KC": "K·C", "KD": "K·D"}
    parts = []
    for _, r in coef.iterrows():
        v = r.coef
        s = f"{abs(v):.4f}" + ("·" + sym[r.column] if sym[r.column] else "")
        parts.append(("−" if v < 0 else "") + s if not parts else (" − " if v < 0 else " + ") + s)
    return "ln f = " + "".join(parts)


def age_numbers(A):
    K = {}
    fit = A["fit"]
    K.update({"R2": f2(fit["R2"]), "adj": f2(fit["adjR2"]), "pred": f2(fit["predR2"]), "s": f3(fit["s_ln"]), "cv": f"{fit['cv_pct']:.0f} %",
              "adeq": f1(fit["adeq_precision"]), "F": f"F({fit['df_model']}, {fit['df_resid']}) = {fit['F_model']:.1f}", "pF": pe(fit["p_model"]),
              "lof": pe(fit["lof"]["p"]), "pe": f3(A["pure_error"]["sd_ln"]), "peM": f1(A["pure_error"]["sd_MPa"]),
              "rm_fit": f1(A["validation"]["fit"]["rmse_MPa"]), "rm_loo": f1(A["validation"]["loo"]["rmse_MPa"]),
              "pr_lodpo": f2(A["validation"]["lodpo"]["predR2"]), "aicc": f2(fit["AICc"]),
              "bc": f"λ = {A['boxcox_final']['lambda']:.2f} (95 % CI {mn(f2(A['boxcox_final']['ci'][0]))} to {A['boxcox_final']['ci'][1]:.2f})",
              "bc_full": f"λ = {mn(f2(A['boxcox_full']['lambda']))} (95 % CI {mn(f2(A['boxcox_full']['ci'][0]))} to {A['boxcox_full']['ci'][1]:.2f})",
              "del_same": int(pd.DataFrame(A["deletion"]).same.sum()),
              "resid_exact": pct(A["bootstrap"]["resid"]["exact_final"]), "sub_exact": pct(A["bootstrap"]["subsample"]["exact_final"])})
    nc = pd.DataFrame(A["nested_cv"])
    ncm = nc[nc.cv == "leave-one-mixture-out"].set_index("procedure")
    K["ncv_aicc"] = f2(ncm.loc["AICc (primary)", "predR2"])
    dg = A["diagnostics"]
    K.update({"sw": pe(dg["shapiro_p"]), "bp": pe(dg["bp_p"]), "maxt": f2(dg["max_t"]), "maxt_mix": dg["max_t_mix"],
              "bonf": pe(dg["min_bonf"]), "cook": f2(dg["cook_max"]), "cook_mix": dg["cook_mix"], "dw": f2(dg["durbin_watson"])})
    return K


def terms_side_by_side(R):
    rows = []
    for t in dz.LEVEL_TERMS:
        row = {"term": TXT[t]}
        for a in AGES:
            A = R[a]
            if t in A["final_terms"]:
                an = pd.DataFrame(A["anova"]).set_index("source").loc[t]
                coef = pd.DataFrame(A["coef"])
                cs = coef[coef.term == t]
                est = "<br>".join((f"{COLTXT[r.column]}: " if len(cs) > 1 else "") + f"{mn(f'{r.coef:.3f}')} [{mn(f'{r.lo:.3f}')}, {mn(f'{r.hi:.3f}')}]" for _, r in cs.iterrows())
                row[f"s{a}"] = "selected"
                row[f"e{a}"] = "<b>" + est + "</b>"
                row[f"p{a}"] = fp(an.p)
            else:
                ad = pd.DataFrame(A["added_terms"]).set_index("term").loc[t]
                q = stats.t.ppf(0.975, ad.df2)
                ests = [(k, v) for k, v in ad.estimates.items() if dz.term_of_column(k) == t]
                est = "<br>".join((f"{COLTXT[k]}: " if len(ests) > 1 else "") + f"{mn(f'{v[0]:.3f}')} [{mn(f'{v[0] - q * v[1]:.3f}')}, {mn(f'{v[0] + q * v[1]:.3f}')}]" for k, v in ests)
                row[f"s{a}"] = "not selected"
                row[f"e{a}"] = '<span class="added">if added: ' + est + "</span>"
                row[f"p{a}"] = fp(ad.p)
            row[f"v{a}"] = A["evidence"][t]
        rows.append(row)
    return pd.DataFrame(rows)


def cross_models(R):
    """Supplementary refits: 7-day terms on the 28-day data, 28-day terms on the 7-day data."""
    import run_separate as rs
    d = dz.load()
    xf = {}
    for src, tgt in (("7", "28"), ("28", "7")):
        terms = R[src]["final_terms"]
        m, X, names = rs.ols(terms, d, np.log(d[f"f{tgt}_mean"].values))
        xf[tgt] = (m, terms)
    return xf, rs


def xf_ratio(xf, rs, tgt, p1, p2):
    m, terms = xf[tgt]
    c = rs.contrast(m, terms, {"AB": 0.45, **p1}, {"AB": 0.45, **p2})
    return ratio_txt(c)


def xf_ss_max(xf, rs, tgt, carb, rcf):
    m, terms = xf[tgt]
    ss = np.linspace(0, 75, 301)
    v = rs.pts_design(terms, pd.DataFrame({"RCF_pct": rcf, "SS_pct": ss, "AB": 0.45, "carbonation": carb})) @ m.params
    j = int(np.argmax(v))
    return f"{ss[j]:.0f} %" + ("" if 0 < j < len(ss) - 1 else " (edge)")


def xf_rcf_max(xf, rs, tgt, carb, ssv):
    m, terms = xf[tgt]
    rc = np.linspace(10, 50, 161)
    v = rs.pts_design(terms, pd.DataFrame({"RCF_pct": rc, "SS_pct": ssv, "AB": 0.45, "carbonation": carb})) @ m.params
    j = int(np.argmax(v))
    return f"{rc[j]:.0f} %" + ("" if 0 < j < len(rc) - 1 else " (edge)")


def trend_tables(R):
    T = {}
    xf, rs = cross_models(R)
    # RCF 50 vs 10
    rows = []
    for carb in ("NC", "1 h"):
        for ss in (0, 25, 50, 75):
            r = {"carb": carb, "SS": ss}
            for a in AGES:
                x = pd.DataFrame(R[a]["trends"]["rcf_50_vs_10"])
                x = x[(x.carbonation == carb) & (x.SS == ss)].iloc[0].to_dict()
                r[a] = ratio_txt(x)
            r["xf"] = xf_ratio(xf, rs, "28", {"RCF_pct": 50, "SS_pct": ss, "carbonation": carb}, {"RCF_pct": 10, "SS_pct": ss, "carbonation": carb})
            rows.append(r)
    T["rcf"] = pd.DataFrame(rows)
    rows = []
    for carb in ("1 h",):
        for rcf in (10, 30, 50):
            for ss in (0, 25, 50, 75):
                r = {"carb": carb, "RCF": rcf, "SS": ss}
                for a in AGES:
                    x = pd.DataFrame(R[a]["trends"]["carb_vs_NC"])
                    x = x[(x.carbonation == carb) & (x.RCF == rcf) & (x.SS == ss)].iloc[0].to_dict()
                    r[a] = ratio_txt(x)
                r["xf"] = xf_ratio(xf, rs, "28", {"RCF_pct": rcf, "SS_pct": ss, "carbonation": carb}, {"RCF_pct": rcf, "SS_pct": ss, "carbonation": "NC"})
                rows.append(r)
    T["carb"] = pd.DataFrame(rows)
    rows = []
    for c in ("1 h / 0.5 h", "5 h / 0.5 h", "5 h / 1 h"):
        r = {"contrast": c}
        for a in AGES:
            x = pd.DataFrame(R[a]["trends"]["duration"]).set_index("contrast").loc[c].to_dict()
            r[a] = ratio_txt(x)
        a_, b_ = c.split(" / ")
        r["xf"] = xf_ratio(xf, rs, "28", {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": a_}, {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": b_})
        rows.append(r)
    T["dur"] = pd.DataFrame(rows)
    rows = []
    for carb in ("NC", "1 h"):
        for rcf in (10, 30, 50):
            r = {"carb": carb, "RCF": rcf}
            for a in AGES:
                sp = pd.DataFrame(R[a]["trends"]["ss_profile"])
                sp = sp[(sp.carbonation == carb) & (sp.RCF == rcf)].iloc[0]
                b = R[a]["bootstrap"]["case"]["opt_ss"][f"{carb}|RCF{rcf}"]
                sc = pd.DataFrame(R[a]["trends"]["ss_contrasts"])
                sc = sc[(sc.carbonation == carb) & (sc.RCF == rcf)].iloc[0]
                r[f"opt{a}"] = (f"{sp.SS_at_max:.0f} % [{b['lo']:.0f}–{b['hi']:.0f}]" if sp.interior else "75 % (edge)*")
                r[f"s0{a}"] = f"×{sc.SS0_vs_max:.2f} [{sc.SS0_lo:.2f}–{sc.SS0_hi:.2f}]"
                r[f"s2{a}"] = f"×{sc.SSSH2_vs_max:.2f} [{sc.SSSH2_lo:.2f}–{sc.SSSH2_hi:.2f}]"
            r["xf"] = xf_ss_max(xf, rs, "28", carb, rcf)
            rows.append(r)
    T["ss"] = pd.DataFrame(rows)
    rows = []
    for carb in ("NC", "1 h"):
        for ss in (0, 37.5, 75):
            r = {"carb": carb, "SS": ss}
            for a in AGES:
                rp = pd.DataFrame(R[a]["trends"]["rcf_profile"])
                rp = rp[(rp.carbonation == carb) & (rp.SS == ss)].iloc[0]
                b = R[a]["bootstrap"]["case"]["opt_rcf"][f"{carb}|SS{ss:g}"]
                r[a] = (f"{rp.RCF_at_max:.0f} % [{b['lo']:.0f}–{b['hi']:.0f}]" if rp.interior else f"{rp.RCF_at_max:.0f} % (edge)")
            r["xf"] = xf_rcf_max(xf, rs, "7", carb, ss)
            rows.append(r)
    T["rcfopt"] = pd.DataFrame(rows)
    T["ab"] = {a: ratio_txt(R[a]["trends"]["ab_042_vs_048"]) for a in AGES}
    T["ab_xf7"] = xf_ratio(xf, rs, "7", {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": "NC", "AB": 0.42},
                           {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": "NC", "AB": 0.48})
    rows = []
    for carb in dz.CARB_LEVELS:
        r = {"carb": carb}
        for a in AGES:
            o = {x["carbonation"]: x for x in R[a]["optimum"]}[carb]
            ab = f", A/B {o['AB']:.2f}" if "C" in R[a]["final_terms"] else ""
            r[f"w{a}"] = f"RCF {o['RCF']:.0f}, SS {o['SS']:.0f}{ab}"
            r[f"f{a}"] = f"{o['median']:.1f} [{o['ci'][0]:.1f}–{o['ci'][1]:.1f}]"
            r[f"pi{a}"] = f"{o['pi'][0]:.1f}–{o['pi'][1]:.1f}"
        rows.append(r)
    T["opt"] = pd.DataFrame(rows)
    return T


def detail_section(a, A, K):
    """HTML for one model's full detail section."""
    top = pd.DataFrame(A["search_top"]).head(10).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["tt"] = top.terms.map(lambda s: ", ".join(TXT.get(t, t) for t in s.split()))
    top_tab = table(top, ["rank", "tt", "p", "R2", "adjR2", "predR2", "AICc", "dAICc", "weight"],
                    ["#", "Terms", "Parameters", "R²", "Adj R²", "Pred R²", "AICc", "ΔAICc", "Akaike weight"],
                    {"R2": f3, "adjR2": f3, "predR2": f3, "AICc": lambda v: mn(f2(v)), "dAICc": f2, "weight": f3},
                    num=["rank", "p", "R2", "adjR2", "predR2", "AICc", "dAICc", "weight"], cls="small")
    fs = pd.DataFrame(A["fit_summary"])
    fs_tab = table(fs, ["model", "p", "seq_p", "lof_p", "R2", "adjR2", "predR2", "s"],
                   ["Model order", "Parameters", "Sequential p", "Lack-of-fit p", "R²", "Adj R²", "Pred R²", "SD (ln)"],
                   {"seq_p": fp, "lof_p": fp, "R2": f3, "adjR2": f3, "predR2": lambda v: mn(f3(v)), "s": f3},
                   num=["p", "seq_p", "lof_p", "R2", "adjR2", "predR2", "s"], cls="small")
    cd = pd.DataFrame(A["coding"])
    cd["lab"] = cd.coding.map({"cat4": "4-level categorical (primary)", "onoff": "Carbonated yes / no", "numeric": "Duration, numeric", "log": "Duration, log₂(1 + h)"})
    cd["bt"] = cd.best_terms.map(lambda s: ", ".join(TXT.get(t, t) for t in s.split()))
    cd_tab = table(cd, ["lab", "bt", "k", "AICc", "dAICc", "predR2_LOO"], ["Carbonation coding", "AICc-best model", "Parameters", "AICc", "ΔAICc", "Pred R² (LOO)"],
                   {"AICc": lambda v: mn(f2(v)), "dAICc": lambda v: mn(f"{v:+.2f}"), "predR2_LOO": f3}, num=["k", "AICc", "dAICc", "predR2_LOO"], cls="small")
    an = pd.DataFrame(A["anova"])
    an["lab"] = an.source.map(TXT)
    rb, sb = A["bootstrap"]["resid"]["inclusion"], A["bootstrap"]["subsample"]["inclusion"]
    an["rb"] = an.source.map(lambda t: pct(rb[t])); an["sb"] = an.source.map(lambda t: pct(sb[t])); an["ev"] = an.source.map(lambda t: A["evidence"][t])
    an_tab = table(an, ["lab", "df", "SS", "F", "p", "rb", "sb", "ev"], ["Term", "df", "Partial SS", "F", "p", "Residual bootstrap", "Subsamples 24/30", "Evidence"],
                   {"SS": lambda v: f"{v:.4f}", "F": f2, "p": fp, "ev": chip}, num=["df", "SS", "F", "p", "rb", "sb"])
    co = pd.DataFrame(A["coef"])
    co["lab"] = co.column.map(COLTXT)
    co["ci"] = [mn(f"{l:.3f} to {h:.3f}") for l, h in zip(co.lo, co.hi)]
    co["br"] = [mn(f"{A['bootstrap']['resid']['beta_ci'][c][0]:.3f} to {A['bootstrap']['resid']['beta_ci'][c][1]:.3f}") for c in co.column]
    co["bc"] = [mn(f"{A['bootstrap']['case']['beta_ci'][c][0]:.3f} to {A['bootstrap']['case']['beta_ci'][c][1]:.3f}") for c in co.column]
    co_tab = table(co, ["lab", "coef", "se", "ci", "p", "VIF", "br", "bc"],
                   ["Term (coded)", "Estimate", "SE", "95 % CI", "p", "VIF", "Residual bootstrap 95 %", "Case bootstrap 95 % (stratified)"],
                   {"coef": lambda v: mn(f"{v:.4f}"), "se": lambda v: f"{v:.4f}", "p": fp, "VIF": lambda v: "–" if v is None else f"{v:.2f}"},
                   num=["coef", "se", "ci", "p", "VIF", "br", "bc"], cls="small")
    ad = pd.DataFrame(A["added_terms"])
    ad["lab"] = ad.added.map(lambda s: " + ".join(TXT.get(t.strip(), t) for t in s.split("+")))
    ad_tab = table(ad, ["lab", "df1", "df2", "F", "p", "dAICc"], ["Term(s) added to the final model", "df", "Residual df", "F", "p", "ΔAICc"],
                   {"F": f2, "p": fp, "dAICc": lambda v: mn(f"{v:+.2f}")}, num=["df1", "df2", "F", "p", "dAICc"], cls="small")
    nc = pd.DataFrame(A["nested_cv"])
    ncm = nc[nc.cv == "leave-one-mixture-out"].copy()
    ncg = nc[nc.cv != "leave-one-mixture-out"].set_index("procedure")
    ncm["g"] = ncm.procedure.map(lambda p: ncg.loc[p, "predR2"])
    ncm["st"] = [("–" if v is None or not np.isfinite(v) else f"{int(v)} / {int(f)}") for v, f in zip(ncm.distinct, ncm.folds)]
    nc_tab = table(ncm, ["procedure", "predR2", "rmse_MPa", "g", "st"], ["Procedure (re-run in every fold)", "Pred R², mixture out", "RMSE (MPa)",
                                                                          "Pred R², design point out", "Distinct models / folds"],
                   {"predR2": lambda v: mn(f2(v)), "rmse_MPa": f1, "g": lambda v: mn(f2(v))}, num=["predR2", "rmse_MPa", "g", "st"], cls="small")
    rob = pd.DataFrame(A["robustness"])
    keys = ["RCF50/10, 1 h, SS 0", "RCF50/10, NC, SS 75", "1 h/NC, RCF 50, SS 0", "1 h/NC, RCF 10, SS 75", "SS 50/0, NC, RCF 30", "SS 75/50, NC, RCF 30", "A/B 0.42/0.48"]
    rob["tt"] = rob.terms.map(lambda s: ", ".join(TXT.get(t, t) for t in s.split()))
    rob_tab = table(rob, ["analysis", "tt"] + keys, ["Analysis", "Terms"] + keys, {"analysis": E, **{k: f2 for k in keys}}, num=keys, cls="small")
    dele = pd.DataFrame(A["deletion"])
    diff = dele[~dele.same]
    del_txt = "; ".join(f"mix {int(r.mix_removed)} → {', '.join(TXT.get(t, t) for t in r.terms.split())}" for _, r in diff.iterrows()) or "none"
    cc = A["cross_checks"]
    cc_txt = "; ".join(f"{k.replace('_', ' ')}: {', '.join(TXT.get(t, t) for t in v)}" for k, v in cc.items())
    top_models_r = ", ".join(f"{', '.join(TXT.get(t, t) for t in m.split())} ({c})" for m, c in A["bootstrap"]["resid"]["top_models"][:3])
    top_models_s = ", ".join(f"{', '.join(TXT.get(t, t) for t in m.split())} ({c})" for m, c in A["bootstrap"]["subsample"]["top_models"][:3])
    dur = A.get("duration_equal_test")
    dur_txt = f" The three carbonated levels do not differ in level (F({dur['df1']}, {dur['df2']}) = {dur['F']:.2f}, {pe(dur['p'])})." if dur else ""
    return f"""
<section id="m{a}">
  <h2>{a}-day model in detail</h2>
  <h3>Scale and model order</h3>
  <p>Box–Cox on the full candidate model: {K['bc_full']}; on the selected model: {K['bc']}. ln is inside both intervals and no transformation (λ = 1) is outside, so the model is fitted to ln(strength). Replicate batches give a pure-error SD of {K['pe']} on the ln scale ({K['peM']} MPa).</p>
  {fs_tab}
  <h3>Carbonation coding</h3>
  {cd_tab}
  <p class="note">The 4-level coding is replaced only if another coding's best model has AICc lower by ≥ 2 and a higher leave-one-out predicted R². {"The rule is not met; the 4-level coding is kept." if not any(cd.switch_rule_met) else "The rule is met, but the primary coding is kept; see the alternative model in the robustness table."}{dur_txt}</p>
  <h3>Term selection</h3>
  {top_tab}
  <p class="note">Top 10 of {A['n_models']} hierarchical models by AICc. Cross-checks: {cc_txt}. With each mixture deleted in turn the search returns the same model {K['del_same']} times out of 30 (different: {del_txt}).</p>
  <h3>Nested cross-validation of the selection procedure</h3>
  {nc_tab}
  <h3>ANOVA</h3>
  {an_tab}
  <p class="note">Model {K['F']}, {K['pF']}. Partial F tests on the ln scale. Lack of fit against replicate batches: {K['lof']}. R² {K['R2']}, adjusted {K['adj']}, predicted (leave one mixture out) {K['pred']}, leave one design point out {K['pr_lodpo']}; residual SD {K['s']} (CV {K['cv']}); adequate precision {K['adeq']}; AICc {mn(K['aicc'])}.</p>
  <h3>Coefficients</h3>
  {co_tab}
  <h3>Diagnostics and stability</h3>
  <p>Residuals: Shapiro–Wilk {K['sw']}, Breusch–Pagan {K['bp']}, Durbin–Watson (mix order) {K['dw']}. Largest externally studentized residual {K['maxt']} (mix {K['maxt_mix']}; Bonferroni {K['bonf']}); largest Cook's distance {K['cook']} (mix {K['cook_mix']}). Residual-bootstrap re-selection returns the exact model in {K['resid_exact']} of {A['bootstrap']['resid']['n']:,} resamples (most frequent: {top_models_r}); subsamples of 24 mixtures in {K['sub_exact']} (most frequent: {top_models_s}).</p>
  <details><summary>Added-term tests for every omitted term</summary>{ad_tab}</details>
  <details><summary>Robustness: trend ratios under alternative analyses</summary>{rob_tab}</details>
</section>"""


def main():
    R = load()
    K = {a: age_numbers(R[a]) for a in AGES}
    TS = terms_side_by_side(R)
    T = trend_tables(R)

    # copy figures
    for fn in glob.glob(os.path.join(RES, "figures", "*.png")):
        shutil.copy(fn, os.path.join(REP, "figures", os.path.basename(fn)))

    # ---------------------------------------------------------------- model cards
    cards = []
    ad28 = pd.DataFrame(R["28"]["added_terms"])
    carb_p28 = float(ad28[ad28.term.isin(["Carb", "KA", "KD", "KC"])].p.min())
    notes = {
        "7": (f"All selection criteria agree (AICc, BIC, backward elimination at α = 0.10 and 0.05). The carbonation terms are kept in "
              f"{pct(R['7']['bootstrap']['resid']['inclusion']['KD'])} of residual-bootstrap re-selections but in "
              f"{pct(R['7']['bootstrap']['subsample']['inclusion']['KD'])} of subsamples of 24 mixtures, because they rest on the 8 uncarbonated mixtures."),
        "28": (f"SS is the only robust term. A/B depends on mix 7 and RCF² on mix 14: each drops out when that mixture is removed. "
               f"No carbonation term is detected (best added-term {pe(carb_p28)}). "
               f"Predictions for a new mixture are much less precise than at 7 days."),
    }
    for a in AGES:
        A = R[a]; k = K[a]
        eq = eq_html(A["equation_actual"], f"ln(f<sub>{a}</sub>)")
        terms_html = " ".join(f'<span class="tchip">{TXT[t]} {chip(A["evidence"][t])}</span>' for t in A["final_terms"])
        cards.append(f"""
<article class="mcard m{a}">
  <p class="eyebrow">{a}-day model · {len(A['coef'])} parameters · 30 mixtures</p>
  <h3>ln(f<sub>{a}</sub>) ~ {' + '.join(TXT[t] for t in A['final_terms'])}</h3>
  <p class="eq">{coded_equation(A)}</p>
  <p class="eq">{eq}</p>
  <div class="strip">
    <div><span class="lab">R² / adj / pred</span><span class="val">{k['R2']}</span><span class="sub">{k['adj']} / {k['pred']}</span></div>
    <div><span class="lab">RMSE fitted / left out</span><span class="val">{k['rm_fit']}</span><span class="sub">{k['rm_loo']} MPa left out</span></div>
    <div><span class="lab">Residual scatter</span><span class="val">{k['cv']}</span><span class="sub">replicate batches {100 * float(k['pe']):.0f} %</span></div>
    <div><span class="lab">Lack of fit</span><span class="val">{k['lof'].replace('p = ', '')}</span><span class="sub">vs replicate error (p)</span></div>
  </div>
  <p class="terms">{terms_html}</p>
  <p class="note">{notes[a]}</p>
</article>""")
    cards_html = "\n".join(cards)

    ts_tab = table(TS, ["term", "e7", "p7", "v7", "e28", "p28", "v28"],
                   ["Candidate term", "7-day model: coefficient [95 % CI]", "p", "Evidence", "28-day model: coefficient [95 % CI]", "p", "Evidence"],
                   {"v7": chip, "v28": chip}, num=["p7", "p28"], cls="small cmp")
    XF = "Supplementary: 28-day data refitted with the 7-day terms"
    rcf_tab = table(T["rcf"], ["carb", "SS", "7", "28", "xf"], ["Carbonation", "SS (%)", "7-day model", "28-day model", XF], num=["SS", "7", "28", "xf"], cls="small xf")
    carb_tab = table(T["carb"], ["RCF", "SS", "7", "28", "xf"], ["RCF (%)", "SS (%)", "7-day model: 1 h / NC", "28-day model: 1 h / NC", XF],
                     num=["RCF", "SS", "7", "28", "xf"], cls="small xf")
    dur_tab = table(T["dur"], ["contrast", "7", "28", "xf"], ["Contrast (RCF 30, SS 37.5)", "7-day model", "28-day model", XF], num=["7", "28", "xf"], cls="small xf")
    ss_tab = table(T["ss"], ["carb", "RCF", "opt7", "opt28", "s07", "s028", "s27", "s228", "xf"],
                   ["Carbonation", "RCF (%)", "7 d: SS at max [95 % boot]", "28 d: SS at max", "7 d: f(SS 0) / f(max)", "28 d: f(SS 0) / f(max)",
                    "7 d: f(SS/SH = 2) / f(max)", "28 d: f(SS/SH = 2) / f(max)", "Suppl.: 28-day data, 7-day terms: SS at max"],
                   num=["RCF", "opt7", "opt28", "s07", "s028", "s27", "s228", "xf"], cls="small xf")
    rcfopt_tab = table(T["rcfopt"], ["carb", "SS", "7", "28", "xf"], ["Carbonation", "SS (%)", "7 d: RCF at maximum [95 % boot]", "28 d: RCF at maximum [95 % boot]",
                                                                    "Supplementary: 7-day data refitted with the 28-day terms"],
                       {"SS": lambda v: f"{v:g}"}, num=["SS", "7", "28", "xf"], cls="small xf")
    opt_tab = table(T["opt"], ["carb", "w7", "f7", "pi7", "w28", "f28", "pi28"],
                    ["Carbonation", "7 d: maximum at", "7 d: median [95 % CI] MPa", "7 d: 95 % PI", "28 d: maximum at", "28 d: median [95 % CI] MPa", "28 d: 95 % PI"],
                    num=["f7", "pi7", "f28", "pi28"], cls="small")
    cf = R["cross_fit"]
    cross_rows = []
    for key, lab in (("7_terms_on_28d", "7-day model terms fitted to the 28-day data"), ("28_terms_on_7d", "28-day model terms fitted to the 7-day data")):
        for r in cf[key]["anova"]:
            cross_rows.append({"fit": lab, "term": TXT[r["source"]], "F": r["F"], "p": r["p"]})
    cross_tab = table(pd.DataFrame(cross_rows), ["fit", "term", "F", "p"], ["Refit", "Term", "F", "p"], {"F": f2, "p": fp}, num=["F", "p"], cls="small")

    page_data = {a: {"names": R[a]["model_export"]["names"], "beta": R[a]["model_export"]["beta"], "cov": R[a]["model_export"]["cov"],
                     "s2": R[a]["model_export"]["s2"], "t": R[a]["model_export"]["t975"]} for a in AGES}
    dd = dz.load()
    page_data["runs"] = [{"mix": int(r.mix), "carb": r.carbonation, "RCF": r.RCF_pct, "SS": r.SS_pct, "AB": r.AB, "f7": r.f7_mean, "f28": r.f28_mean}
                         for _, r in dd.iterrows()]
    js = JS.replace("__DATA__", json.dumps(page_data, separators=(",", ":")))

    A7, A28 = R["7"], R["28"]
    body = f"""
<header class="top">
  <p class="eyebrow">Compressive strength · 30 mixtures · two independent models</p>
  <h1>7-Day and 28-Day Strength Models</h1>
  <p class="lede">One model for the 7-day strengths and one for the 28-day strengths. Each was built from its own 30 results only, with the same protocol: ln scale chosen by Box–Cox, the same 16-parameter candidate set (RCF, A/B and SS up to quadratic terms, carbonation at four levels, carbonated × mixture slopes), an exhaustive AICc search of all {R['meta']['n_candidate_models']} hierarchical models, and the same validation. The two are set side by side for comparison; this page does not judge whether one age's trends hold at the other.</p>
  <p class="meta">Data: <code>data/strength_7d_28d_corrected.csv</code> · Analysis: <code>analysis/run_separate.py</code> · {R['meta']['n_boot']:,} resamples per scheme · built {R['meta']['date']}</p>
  <nav class="toc" aria-label="Sections"><a href="#models">Models</a><a href="#terms">Terms side by side</a><a href="#trends">Trends side by side</a><a href="#explore">Explore</a><a href="#m7">7-day details</a><a href="#m28">28-day details</a><a href="#cross">Cross-fits</a><a href="#protocol">Protocol</a></nav>
</header>
<main>
<section id="models">
  <h2>The two models</h2>
  <p class="note">Coded units: A = (RCF − 30)/20, C = (A/B − 0.45)/0.03, D = (SS − 37.5)/37.5, K = 1 for carbonated RCF, C<sub>x</sub> = carbonation-level indicators. Actual units: RCF and SS in %, C[·] = 1 for that level. Medians in MPa are exp(ln f). Evidence labels use one rule for both models: <b>Robust</b> p &lt; 0.01 and kept in ≥ 90 % of residual-bootstrap and ≥ 80 % of subsample re-selections; <b>Moderate</b> p &lt; 0.05 and ≥ 60 % of residual-bootstrap re-selections; <b>Weak</b> selected but short of Moderate; <b>Hierarchy</b> kept because a higher-order term contains it.</p>
  <div class="mcards">{cards_html}</div>
</section>

<section id="terms">
  <h2>Terms side by side</h2>
  <p>Every candidate term, as each model sees it. For a term a model did not select, the coefficient is the value it takes when added to that model (with any parent terms it needs), and p is the added-term F test.</p>
  {ts_tab}
  <figure class="plate"><img src="figures/H02_coefficients_both_models.png" alt="Coded coefficients of both models with 95 % confidence intervals" loading="lazy"></figure>
  <figure class="plate"><img src="figures/H01_scale_and_term_support.png" alt="Box–Cox profiles and term support for both models" loading="lazy"></figure>
</section>

<section id="trends">
  <h2>Trends side by side</h2>
  <p>The same quantities computed from each model. Ratios compare two predicted strengths with 95 % CIs; a ratio of 1 with no interval means the term behind it is not in that model. A/B is held at 0.45 unless stated.</p>
  <p class="note">The grey "supplementary" columns are not part of either model. They refit one age's data with the other model's terms, so you can see what the omitted effects look like, with their uncertainty, when they are forced in. * 75 % (edge): the 28-day model has no SS² term, so strength rises to the highest SS tested.</p>
  <h3>SS</h3>
  {ss_tab}
  <figure class="plate"><img src="figures/H03_strength_vs_SS_both_models.png" alt="Predicted strength against SS from both models" loading="lazy"></figure>
  <h3>RCF</h3>
  {rcf_tab}
  {rcfopt_tab}
  <figure class="plate"><img src="figures/H04_strength_vs_RCF_both_models.png" alt="Predicted strength against RCF from both models" loading="lazy"></figure>
  <h3>Carbonation</h3>
  {carb_tab}
  {dur_tab}
  <figure class="plate"><img src="figures/H05_effect_ratios_both_models.png" alt="Carbonation and RCF effect ratios from both models" loading="lazy"></figure>
  <h3>A/B</h3>
  <p>Strength at A/B 0.42 relative to 0.48 (RCF 30 %, SS 37.5 %, NC): 7-day model {T['ab']['7']}; 28-day model {T['ab']['28']}. Supplementary: the 7-day data refitted with the 28-day terms give {T['ab_xf7']}.</p>
  <h3>Best combination per carbonation level</h3>
  {opt_tab}
  <p class="note">Maximum of the median prediction over RCF 10–50 %, SS 0–75 % and, where A/B is in the model, A/B 0.42–0.48. No maximum requires extrapolation (leverage below the largest design leverage). The 28-day maximum sits at the edge of the SS and A/B ranges because the 28-day model has no SS or A/B curvature.</p>
  <figure class="plate"><img src="figures/H06_response_surfaces_both_models.png" alt="Response surfaces of both models" loading="lazy"></figure>
</section>

<section id="explore">
  <h2>Explore both models</h2>
  <div class="controls" role="group" aria-label="Curve settings">
    <label for="rcfSlider">RCF <output id="rcfOut">30</output> %</label>
    <input type="range" id="rcfSlider" min="10" max="50" step="1" value="30">
    <div class="seg" id="carbSeg" role="radiogroup" aria-label="Carbonation"></div>
    <label for="abSel">A/B</label><select id="abSel" class="sel"><option>0.42</option><option selected>0.45</option><option>0.48</option></select>
  </div>
  <div class="split">
    <figure class="chart"><div id="curveChart" class="plot" role="img" aria-label="Predicted strength against SS from both models"></div>
    <figcaption>Median prediction of each model with 95 % CI. Markers: runs at the selected carbonation level within ±10 % RCF. Hover for values.</figcaption></figure>
    <div class="predictor"><h3>Predictor</h3>
      <div class="field"><label for="pRCF">RCF (%)</label><input id="pRCF" type="number" min="10" max="50" step="0.5" value="50"></div>
      <div class="field"><label for="pSS">SS (%)</label><input id="pSS" type="number" min="0" max="75" step="0.5" value="45"></div>
      <div class="field"><label for="pAB">A/B</label><input id="pAB" type="number" min="0.42" max="0.48" step="0.005" value="0.45"></div>
      <div class="field"><label for="pCarb">Carbonation</label><select id="pCarb"><option>NC</option><option>0.5 h</option><option selected>1 h</option><option>5 h</option></select></div>
      <div id="pOut" class="pout" aria-live="polite"></div>
    </div>
  </div>
</section>

{detail_section('7', A7, K['7'])}
{detail_section('28', A28, K['28'])}

<section id="validation">
  <h2>Validation figures</h2>
  <figure class="plate"><img src="figures/H07_observed_vs_predicted_both_models.png" alt="Observed against predicted for both models" loading="lazy"></figure>
  <figure class="plate"><img src="figures/H08_residual_diagnostics_both_models.png" alt="Residual diagnostics for both models" loading="lazy"></figure>
  <figure class="plate"><img src="figures/H10_coefficient_intervals_both_models.png" alt="Coefficient intervals for both models" loading="lazy"></figure>
  <figure class="plate"><img src="figures/H09_robustness_both_models.png" alt="Robustness of trend ratios for both models" loading="lazy"></figure>
</section>

<section id="cross">
  <h2>Supplementary: cross-fits</h2>
  <p>Not used to build either model. Each model's terms were refitted to the other age's data, which shows how the other age's data respond to that structure.</p>
  {cross_tab}
</section>

<section id="protocol">
  <h2>Protocol</h2>
  <ol class="proto">
    <li>Scale by Box–Cox profile likelihood; ln if λ = 0 is inside the 95 % CI of the selected model and λ = 1 outside.</li>
    <li>Candidate set: quadratic in RCF, A/B and SS; carbonation as a 4-level factor; carbonated × (RCF, A/B, SS) — 16 parameters, {R['meta']['n_candidate_models']} models that respect hierarchy.</li>
    <li>Carbonation coding replaced only if another coding's best model has AICc lower by ≥ 2 and a higher leave-one-out predicted R².</li>
    <li>Exhaustive AICc search (primary); BIC, PRESS and backward elimination (α = 0.10, 0.05) as cross-checks; nested cross-validation of each selection procedure.</li>
    <li>Validation: ANOVA, lack of fit against replicate batches (6 df), residual diagnostics, leave-one-mixture-out and leave-one-design-point-out prediction, single-mixture deletion, {R['meta']['n_boot']:,} residual-bootstrap and {R['meta']['n_boot']:,} subsample (24 of 30) re-selections, {R['meta']['n_boot']:,} case-bootstrap refits stratified by carbonation level.</li>
    <li>Robustness: other codings, BIC/PRESS/backward models, raw and Box–Cox ±0.5 scales, weights for single-specimen means, Huber regression, influential mixtures removed.</li>
    <li>Trends computed identically from each model.</li>
  </ol>
  <pre class="code">python analysis/run_separate.py        # both models (~{R['runtime_s'] / 60:.0f} min on 4 cores)
python analysis/make_figures_separate.py
python analysis/report_separate.py</pre>
</section>
</main>"""

    css = CSS + EXTRA_CSS
    page = f"""<title>7-Day and 28-Day Strength Models</title>
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
    print("wrote report/separate/index.html")
    build_md(R, K, TS, T, cf)
    build_pdf(R, K, cards_html, ts_tab, ss_tab, rcf_tab, carb_tab, dur_tab, opt_tab, T)


def build_md(R, K, TS, T, cf):
    import re

    def plain(s):
        s = re.sub(r"<sub>(.*?)</sub>", r"\1", s)
        s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
        return html.unescape(re.sub(r"<[^>]+>", "", s))
    parts = [f"""# 7-day and 28-day strength models (independent)

*Generated by `analysis/report_separate.py` from `results/separate/separate_results.json`. Every number comes from the analysis files.*

Two models built separately, each from its own 30 results, with the same protocol: ln scale (Box–Cox), the same 16-parameter candidate set
(RCF, A/B and SS up to quadratic terms; carbonation as a 4-level factor; carbonated × mixture slopes), exhaustive AICc search over all
{R['meta']['n_candidate_models']} hierarchical models, and identical validation. The comparison of trends between ages is left to the reader.
"""]
    for a in ("7", "28"):
        A = R[a]; k = K[a]
        parts.append(f"""## {a}-day model

**ln(f{a}) ~ {' + '.join(TXT[t] for t in A['final_terms'])}**

Coded (A = (RCF − 30)/20, C = (A/B − 0.45)/0.03, D = (SS − 37.5)/37.5, K = 1 if carbonated):

    {plain(coded_equation(A))}

Actual units (RCF, SS in %; C[·] carbonation-level indicators):

    {plain(eq_html(A['equation_actual'], f'ln(f{a})'))}

| Statistic | Value |
|---|---|
| R² / adjusted / predicted (leave one mixture out) | {k['R2']} / {k['adj']} / {k['pred']} |
| Predicted R², leave one design point out | {k['pr_lodpo']} |
| Model F | {k['F']}, {k['pF']} |
| Residual SD (ln) / CV | {k['s']} / {k['cv']} (replicate batches {k['pe']}) |
| RMSE fitted / left out (MPa) | {k['rm_fit']} / {k['rm_loo']} |
| Lack of fit | {k['lof']} |
| Adequate precision | {k['adeq']} |
| Box–Cox (selected model) | {k['bc']} |
| Same model with each mixture deleted | {k['del_same']} / 30 |
| Exact model re-selected: residual bootstrap / subsamples | {k['resid_exact']} / {k['sub_exact']} |
| Nested CV predicted R² of the AICc procedure | {k['ncv_aicc']} |
| Shapiro–Wilk / Breusch–Pagan | {k['sw']} / {k['bp']} |
| Largest studentized residual | {k['maxt']} (mix {k['maxt_mix']}, Bonferroni {k['bonf']}) |

Cross-checks: {'; '.join(f"{kk.replace('_', ' ')}: {', '.join(TXT.get(t, t) for t in v)}" for kk, v in A['cross_checks'].items())}.

{md_table(pd.DataFrame(A['anova']).assign(lab=lambda x: x.source.map(TXT), ev=lambda x: x.source.map(A['evidence'])), ['lab', 'df', 'F', 'p', 'ev'], ['Term', 'df', 'F', 'p', 'Evidence'], {'F': f2, 'p': fp})}

{md_table(pd.DataFrame(A['coef']).assign(lab=lambda x: x.column.map(COLTXT)), ['lab', 'coef', 'se', 'lo', 'hi', 'p'], ['Term (coded)', 'Estimate', 'SE', '95 % CI low', '95 % CI high', 'p'], {'coef': lambda v: f'{v:.4f}', 'se': lambda v: f'{v:.4f}', 'lo': lambda v: f'{v:.4f}', 'hi': lambda v: f'{v:.4f}', 'p': fp})}
""")
    parts.append(f"""## Terms side by side

{md_table(TS.assign(e7=TS.e7.map(plain), e28=TS.e28.map(plain)), ['term', 'e7', 'p7', 'v7', 'e28', 'p28', 'v28'], ['Term', '7-day model: coefficient [95 % CI]', 'p', 'Evidence', '28-day model: coefficient [95 % CI]', 'p', 'Evidence'])}

## Trends side by side

### SS

{md_table(T['ss'], ['carb', 'RCF', 'opt7', 'opt28', 's07', 's028', 's27', 's228'], ['Carbonation', 'RCF', '7 d SS at max', '28 d SS at max', '7 d f(SS0)/f(max)', '28 d f(SS0)/f(max)', '7 d f(SS/SH=2)/f(max)', '28 d f(SS/SH=2)/f(max)'])}

### RCF 50 % / RCF 10 %

{md_table(T['rcf'], ['carb', 'SS', '7', '28'], ['Carbonation', 'SS', '7-day model', '28-day model'])}

{md_table(T['rcfopt'], ['carb', 'SS', '7', '28'], ['Carbonation', 'SS', '7 d RCF at max', '28 d RCF at max'], {'SS': lambda v: f'{v:g}'})}

### Carbonated (1 h) / NC

{md_table(T['carb'], ['RCF', 'SS', '7', '28'], ['RCF', 'SS', '7-day model', '28-day model'])}

### Carbonation duration

{md_table(T['dur'], ['contrast', '7', '28'], ['Contrast', '7-day model', '28-day model'])}

### A/B 0.42 / 0.48

7-day model: {T['ab']['7']}; 28-day model: {T['ab']['28']}.

### Best combination per carbonation level

{md_table(T['opt'], ['carb', 'w7', 'f7', 'pi7', 'w28', 'f28', 'pi28'], ['Carbonation', '7 d at', '7 d median [CI]', '7 d PI', '28 d at', '28 d median [CI]', '28 d PI'])}

## Files

`analysis/run_separate.py`, `analysis/make_figures_separate.py`, `analysis/report_separate.py`; results in `results/separate/`
(`separate_results.json`, tables `S7_*`, `S28_*`, `S_cross_*`, figures `H01–H10`); this report and `index.html` in `report/separate/`.
""")
    with open(os.path.join(REP, "SEPARATE_MODELS.md"), "w") as f:
        f.write("\n".join(parts))
    print("wrote report/separate/SEPARATE_MODELS.md")


def build_pdf(R, K, cards_html, ts_tab, ss_tab, rcf_tab, carb_tab, dur_tab, opt_tab, T):
    fig = "../../results/separate/figures"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>7-Day and 28-Day Strength Models</title>
<style>{SUMMARY_CSS}{PDF_EXTRA}</style></head><body>
<h1>7-day and 28-day strength models (independent)</h1>
<p class="sub">Each model built from its own 30 results with the same protocol · ln scale · exhaustive AICc over {R['meta']['n_candidate_models']} hierarchical models · generated {R['meta']['date']}</p>
{cards_html}
<h2>Terms side by side</h2>{ts_tab}
<div class="fig"><img src="{fig}/H02_coefficients_both_models.png"></div>
<h2>Trends side by side</h2>
<h3>SS</h3>{ss_tab}
<div class="fig"><img src="{fig}/H03_strength_vs_SS_both_models.png"></div>
<h3>RCF 50 % / 10 %</h3>{rcf_tab}
<div class="fig"><img src="{fig}/H04_strength_vs_RCF_both_models.png"></div>
<h3>Carbonated (1 h) / NC</h3>{carb_tab}
<h3>Carbonation duration</h3>{dur_tab}
<p>A/B 0.42 / 0.48: 7-day model {T['ab']['7']}; 28-day model {T['ab']['28']}.</p>
<div class="fig"><img src="{fig}/H05_effect_ratios_both_models.png"></div>
<h3>Best combination per carbonation level</h3>{opt_tab}
<div class="fig"><img src="{fig}/H06_response_surfaces_both_models.png"></div>
<h2>Validation</h2>
<div class="fig"><img src="{fig}/H07_observed_vs_predicted_both_models.png"></div>
<div class="fig"><img src="{fig}/H08_residual_diagnostics_both_models.png"></div>
<div class="fig"><img src="{fig}/H01_scale_and_term_support.png"></div>
</body></html>"""
    p = os.path.join(REP, "summary.html")
    with open(p, "w") as f:
        f.write(doc)
    g = glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome")
    if g:
        out = os.path.join(REP, "Separate_Models_Summary.pdf")
        subprocess.run([g[0], "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "file://" + p],
                       check=False, capture_output=True, timeout=180)
        print("wrote report/separate/Separate_Models_Summary.pdf" if os.path.exists(out) else "PDF not written")


EXTRA_CSS = r"""
.mcards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px; margin: 16px 0; }
.mcard { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; min-width: 0; border-top: 4px solid var(--age7); }
.mcard.m28 { border-top-color: var(--age28); }
.mcard h3 { margin: 4px 0 10px; font-size: 18px; }
.mcard .eq { font-size: 12px; }
.mcard .strip { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
.mcard .strip .val { font-size: 22px; }
.terms { display: flex; flex-wrap: wrap; gap: 6px 10px; margin: 8px 0; }
.tchip { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; border: 1px solid var(--rule); border-radius: 999px; padding: 3px 4px 3px 10px; }
.chip.hierarchy, .chip.weak { color: var(--none); background: var(--none-bg); }
.chip.weak { color: var(--mod); background: transparent; border: 1px solid var(--mod); }
table.cmp td:nth-child(5), table.cmp th:nth-child(5) { border-left: 2px solid var(--rule); }
table.cmp td { font-family: var(--f-mono); font-size: 12px; } table.cmp td:first-child { font-family: var(--f-body); font-size: 13px; }
.added { color: var(--muted); }
table.xf td:last-child, table.xf th:last-child { color: var(--muted); }
.sel { font: 14px var(--f-body); padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--ink); }
"""

PDF_EXTRA = """
.mcards { display: block; } .mcard { border: 1px solid #dde1e6; border-top: 3px solid #2a78d6; padding: 6px 8px; margin: 6px 0; break-inside: avoid; }
.mcard.m28 { border-top-color: #eb6834; } .mcard h3 { font-size: 10pt; margin: 2px 0 4px; } .eyebrow { font-size: 7pt; color: #4b5058; margin: 0; }
.terms .tchip { display: inline-block; margin-right: 8px; font-size: 7.5pt; } .note { font-size: 7.5pt; color: #4b5058; }
h3 { font-size: 9.5pt; margin: 8px 0 3px; }
"""

JS = r"""
(function(){
'use strict';
const D = __DATA__;
const CARBS = ['NC','0.5 h','1 h','5 h'];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = () => ({'7': css('--age7'), '28': css('--age28'), ink: css('--ink'), ink2: css('--ink2'), muted: css('--muted'), grid: css('--grid'), axis: css('--axis'), surface: css('--surface')});
const tip = document.getElementById('tip');
function showTip(ev, h){ tip.innerHTML = h; tip.hidden = false; const r = tip.getBoundingClientRect(); let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14; if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
function hideTip(){ tip.hidden = true; }
const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
function row(M, p){ const A = (p.RCF - 30) / 20, C = (p.AB - 0.45) / 0.03, Dd = (p.SS - 37.5) / 37.5, K = p.carb === 'NC' ? 0 : 1;
  const base = {Intercept: 1, A, C, D: Dd, AC: A*C, AD: A*Dd, CD: C*Dd, A2: A*A, C2: C*C, D2: Dd*Dd, KA: K*A, KC: K*C, KD: K*Dd};
  return M.names.map(nm => nm.startsWith('Carb[') ? (p.carb === nm.slice(5, -1) ? 1 : 0) : base[nm]); }
function pred(age, p){ const M = D[age], x = row(M, p); let e = 0, v = 0;
  for (let i = 0; i < x.length; i++){ e += x[i] * M.beta[i]; for (let j = 0; j < x.length; j++) v += x[i] * M.cov[i][j] * x[j]; }
  const se = Math.sqrt(Math.max(v, 0)), sp = Math.sqrt(se*se + M.s2);
  return {med: Math.exp(e), lo: Math.exp(e - M.t*se), hi: Math.exp(e + M.t*se), plo: Math.exp(e - M.t*sp), phi: Math.exp(e + M.t*sp)}; }
function seg(el, opts, val, cb){ el.innerHTML = ''; opts.forEach(o => { const b = document.createElement('button'); b.type = 'button'; b.setAttribute('role', 'radio');
  b.textContent = o; b.setAttribute('aria-checked', o === val ? 'true' : 'false');
  b.addEventListener('click', () => { el.querySelectorAll('button').forEach(x => x.setAttribute('aria-checked', 'false')); b.setAttribute('aria-checked', 'true'); cb(o); });
  el.appendChild(b); }); }
let RCF = 30, CARB = '1 h', AB = 0.45;
function draw(){
  const el = document.getElementById('curveChart'), c = COL(); el.innerHTML = '';
  const w = Math.max(300, el.clientWidth), h = 360, m = {l: 54, r: 16, t: 30, b: 46}, iw = w - m.l - m.r, ih = h - m.t - m.b;
  const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w).attr('height', h);
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
  const x = d3.scaleLinear().domain([0, 75]).range([0, iw]), ss = d3.range(0, 75.01, 0.5);
  const P = {'7': ss.map(s => pred('7', {RCF, SS: s, AB, carb: CARB})), '28': ss.map(s => pred('28', {RCF, SS: s, AB, carb: CARB}))};
  const near = D.runs.filter(r => r.carb === CARB && Math.abs(r.RCF - RCF) <= 10);
  const y = d3.scaleLinear().domain([0, Math.max(45, d3.max(P['28'], d => d.hi) * 1.05, d3.max(near, r => r.f28) || 0)]).nice().range([ih, 0]);
  g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(6).tickSize(-iw).tickFormat(''));
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(8));
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(6));
  g.append('text').attr('x', iw / 2).attr('y', ih + 36).attr('text-anchor', 'middle').text('Sodium silicate, SS (%)');
  g.append('text').attr('transform', 'rotate(-90)').attr('x', -ih / 2).attr('y', -40).attr('text-anchor', 'middle').text('Strength (MPa, median)');
  ['7', '28'].forEach(a => { const Q = P[a];
    g.append('path').attr('fill', c[a]).attr('opacity', 0.14).attr('d', d3.area().x((d, i) => x(ss[i])).y0(d => y(d.lo)).y1(d => y(d.hi))(Q));
    g.append('path').attr('fill', 'none').attr('stroke', c[a]).attr('stroke-width', 2.2).attr('d', d3.line().x((d, i) => x(ss[i])).y(d => y(d.med))(Q));
    near.forEach(r => g.append('path').attr('d', d3.symbol(a === '7' ? d3.symbolCircle : d3.symbolSquare, 44)()).attr('transform', `translate(${x(r.SS)},${y(r['f' + a])})`)
      .attr('fill', c[a]).attr('stroke', c.surface).attr('stroke-width', 1.5)
      .on('mousemove', ev => showTip(ev, `<b>Mix ${r.mix}</b> · ${r.carb}, RCF ${f1(r.RCF)} %, A/B ${r.AB.toFixed(3)}<br>SS ${f1(r.SS)} % · ${a} d: ${f2(r['f' + a])} MPa`)).on('mouseleave', hideTip)); });
  const lg = svg.append('g').attr('transform', `translate(${m.l},14)`);
  [['7', '7-day model'], ['28', '28-day model']].forEach(([a, t], i) => { const gx = lg.append('g').attr('transform', `translate(${i * 130},0)`);
    gx.append('line').attr('x1', 0).attr('x2', 16).attr('stroke', c[a]).attr('stroke-width', 2.4); gx.append('text').attr('x', 22).attr('y', 4).text(t); });
  const cr = g.append('g').style('display', 'none'); cr.append('line').attr('y1', 0).attr('y2', ih).attr('stroke', c.muted);
  g.append('rect').attr('width', iw).attr('height', ih).attr('fill', 'transparent')
    .on('mousemove', ev => { const [mx] = d3.pointer(ev); const i = Math.round(Math.max(0, Math.min(75, x.invert(mx))) / 0.5); const a = P['7'][i], b = P['28'][i];
      cr.style('display', null); cr.select('line').attr('x1', x(ss[i])).attr('x2', x(ss[i]));
      showTip(ev, `SS ${f1(ss[i])} % · RCF ${RCF} % · ${CARB} · A/B ${AB.toFixed(2)}<br><b>7 d</b> ${f1(a.med)} MPa (CI ${f1(a.lo)}–${f1(a.hi)})<br><b>28 d</b> ${f1(b.med)} MPa (CI ${f1(b.lo)}–${f1(b.hi)})`); })
    .on('mouseleave', () => { cr.style('display', 'none'); hideTip(); });
  near.length; }
function upd(){ const c = COL(); const cl = (v, a, b) => Math.min(b, Math.max(a, isFinite(v) ? v : a));
  const p = {RCF: cl(parseFloat(pRCF.value), 10, 50), SS: cl(parseFloat(pSS.value), 0, 75), AB: cl(parseFloat(pAB.value), 0.42, 0.48), carb: pCarb.value};
  const blk = a => { const q = pred(a, p); return `<div class="row"><div class="k"><span class="sw" style="background:${c[a]}"></span>${a}-day model</div><div class="big">${f1(q.med)} MPa</div><div class="d">95 % CI ${f1(q.lo)}–${f1(q.hi)} · one new batch ${f1(q.plo)}–${f1(q.phi)}</div></div>`; };
  pOut.innerHTML = blk('7') + blk('28'); }
const pRCF = document.getElementById('pRCF'), pSS = document.getElementById('pSS'), pAB = document.getElementById('pAB'), pCarb = document.getElementById('pCarb'), pOut = document.getElementById('pOut');
seg(document.getElementById('carbSeg'), CARBS, CARB, v => { CARB = v; draw(); });
const sl = document.getElementById('rcfSlider'); sl.addEventListener('input', () => { RCF = +sl.value; document.getElementById('rcfOut').textContent = sl.value; draw(); });
document.getElementById('abSel').addEventListener('change', e => { AB = +e.target.value; draw(); });
[pRCF, pSS, pAB, pCarb].forEach(e => e.addEventListener('input', upd));
function all(){ draw(); upd(); }
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(all, 150); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
new MutationObserver(all).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
all();
})();
"""


if __name__ == "__main__":
    main()
