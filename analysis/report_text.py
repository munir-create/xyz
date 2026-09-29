"""
Numbers and narrative for the reports, all derived from results/model_results.json.
Shared by build_report.py (HTML page, printable summary) and the Markdown report.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import design as dz
import model_tools as mt
from lmm import PairedLMM

RES = os.path.join(dz.ROOT, "results")
EXPECTED_FINAL = {"A", "D", "Carb", "AD", "D2", "KA", "KD", "Age", "D.Age", "D2.Age"}


# ---------------------------------------------------------------- formatting
def fp(p, eq=True):
    if p is None or not np.isfinite(p):
        return "–"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}" if eq else f"{p:.3f}"
    return (f"p = {p:.2f}" if eq else f"{p:.2f}") if p >= 0.1 else (f"p = {p:.3f}" if eq else f"{p:.3f}")


def pnum(p):
    return fp(p, eq=False).replace("p < ", "< ")


def rt_p(p):
    return "0.001" if p < 0.001 else (f"{p:.2f}" if p >= 0.1 else f"{p:.3f}")


def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"


def pct(x): return f"{100 * x:.0f} %"


def rt_pct_floor(x):
    return f"{np.floor(100 * x):.0f} %"


def minus(s):
    return s.replace("-", "−")


# ---------------------------------------------------------------- load
def load():
    with open(os.path.join(RES, "model_results.json")) as f:
        R = json.load(f)
    final = R["final_terms"]
    if set(final) != EXPECTED_FINAL:
        raise SystemExit("The selected model differs from the one the narrative was written for "
                         f"({final}); review report_text.py before rebuilding the reports.")
    d = dz.load()
    X, names = dz.design(final, d)
    Y = d[["l7", "l28"]].values
    m = PairedLMM(Y, X, names, R["final_struct"])          # refit (identical numbers; gives Satterthwaite df)
    assert np.allclose(m.beta, R["final"]["beta"], atol=1e-6)
    return R, d, final, m


def extras(R, d, final, m):
    """Quantities computed at report time (fast)."""
    X, names = dz.design(final, d)
    Y = d[["l7", "l28"]].values
    ex = {}
    # consequence of ignoring the pairing: SEs under independence
    mi = PairedLMM(Y, X, names, "IND")
    ex["se_ind_ratio"] = {nm: float(mi.se[j] / m.se[j]) for j, nm in enumerate(names)}
    # representative t quantiles for predictions at each age (median Satterthwaite df over design points)
    from scipy import stats
    pts = pd.DataFrame({"RCF_pct": d.RCF_pct, "SS_pct": d.SS_pct, "AB": d.AB, "carbonation": d.carbonation})
    ex["df_age"], ex["t_age"] = {}, {}
    for age in (7, 28):
        Xa = mt.rows(final, pts.assign(age=age))
        dfs = np.array([m.satterthwaite_df(x) for x in Xa])
        ex["df_age"][age] = float(np.median(dfs))
        ex["t_age"][age] = float(stats.t.ppf(0.975, np.median(dfs)))
    Xg = mt.gain_rows(final, pts.assign(age=28))
    ex["df_gain"] = float(np.median([m.satterthwaite_df(x) for x in Xg]))
    ex["t_gain"] = float(stats.t.ppf(0.975, ex["df_gain"]))
    # predicted gain ratios at SS 0 / 37.5 / 75 with CI
    g = pd.DataFrame(R["gain_curve"]).set_index("SS")
    ex["gain"] = {ss: {"r": g.loc[ss, "ratio_est"], "lo": g.loc[ss, "ratio_lo"], "hi": g.loc[ss, "ratio_hi"]}
                  for ss in (0.0, 37.5, 75.0)}
    # evidence classification per term (pre-specified rule)
    tt = pd.DataFrame(R["final"]["terms_F"]).set_index("term")
    rb, sb = R["bootstrap"]["resid"]["inclusion"], R["bootstrap"]["subsample"]["inclusion"]
    add = pd.DataFrame(R["added_terms"])
    ev = {}
    for t in dz.ALL_TERMS:
        if t in final:
            p = tt.loc[t, "p"]
            if p < 0.01 and rb[t] >= 0.9 and sb[t] >= 0.8:
                ev[t] = "Robust"
            elif p < 0.05 and rb[t] >= 0.6:
                ev[t] = "Moderate"
            else:
                ev[t] = "Retained for hierarchy"
        else:
            row = add[add.terms.str.split().str[0] == t]
            p = float(row.p.iloc[0]) if len(row) else 1.0
            ev[t] = "Suggestive" if (p < 0.10 or rb[t] >= 0.5) else "Not detected"
    ex["evidence"] = ev
    return ex


def key_numbers(R, ex):
    """Dictionary of formatted numbers used across the narrative."""
    D = R["data"]
    fs = R["final"]["fit_stats"]
    vc = R["final"]["variance_components"]
    tt = pd.DataFrame(R["final"]["terms_F"]).set_index("term")
    agesp = pd.DataFrame(R["age_specific"]["table"])
    add = pd.DataFrame(R["added_terms"]).set_index("terms")
    ncv = pd.DataFrame(R["nested_cv"])
    val = R["validation"]
    c = R["contrasts"]
    rcf = pd.DataFrame(R["rcf_effect"]); ce = pd.DataFrame(R["carb_effect"]); cr = pd.DataFrame(R["carb_crossover"])
    sso = pd.DataFrame(R["ss_optimum"])
    bs = R["bootstrap"]
    K = {}
    K["n_models"] = f"{R['search']['UN']['n_models']:,}"
    K["aicc_un"] = f2(R["search"]["UN"]["top"][0]["AICc"]); K["aicc_cs"] = f2(R["search"]["CS"]["top"][0]["AICc"])
    K["daicc_struct"] = f1(R["search"]["CS"]["top"][0]["AICc"] - R["search"]["UN"]["top"][0]["AICc"])
    K["un2"] = f1(R["search"]["UN"]["top"][1]["dAICc"])
    K["w_best"] = f2(R["search"]["UN"]["top"][0]["weight"])
    K["pe7"], K["pe28"] = f3(D["pure_error"]["7 d ln"]["sd"]), f3(D["pure_error"]["28 d ln"]["sd"])
    K["pe7M"], K["pe28M"] = f1(D["pure_error"]["7 d MPa"]["sd"]), f1(D["pure_error"]["28 d MPa"]["sd"])
    K["sd7"], K["sd28"], K["rho"] = f3(vc["sd_7"]), f3(vc["sd_28"]), f2(vc["rho"])
    K["cv7"], K["cv28"] = f"{fs['7']['cv_pct']:.0f} %", f"{fs['28']['cv_pct']:.0f} %"
    K["R2_7"], K["R2_28"] = f2(fs["7"]["R2"]), f2(fs["28"]["R2"])
    K["adj7"], K["adj28"] = f2(fs["7"]["adjR2"]), f2(fs["28"]["adjR2"])
    K["pr7"], K["pr28"] = f2(val["mix"]["predR2_7"]), f2(val["mix"]["predR2_28"])
    K["prg7"], K["prg28"] = f2(val["group"]["predR2_7"]), f2(val["group"]["predR2_28"])
    K["rm7"], K["rm28"] = f1(val["mix"]["rmse_MPa_7"]), f1(val["mix"]["rmse_MPa_28"])
    K["fit_rm7"], K["fit_rm28"] = f1(val["fit"]["rmse_MPa_7"]), f1(val["fit"]["rmse_MPa_28"])
    K["adeq7"], K["adeq28"] = f1(fs["7"]["adeq_precision"]), f1(fs["28"]["adeq_precision"])
    lof = R["final"]["lack_of_fit"]
    K["lof_gain"] = fp(lof["gain (G | X_G)"]["p"]); K["lof_level"] = fp(lof["level given gain (M | X_M, G)"]["p"])
    K["lof7"] = fp(lof["7 d (per-age OLS, same terms)"]["p"]); K["lof28"] = fp(lof["28 d (per-age OLS, same terms)"]["p"])
    K["lof28_sd"] = f3(lof["28 d (per-age OLS, same terms)"]["sd_resid"])
    bc = R["boxcox"]
    K["bc_final"] = f"λ = {bc['final_UN']['lambda']:.2f} (95 % CI {minus(f2(bc['final_UN']['ci'][0]))} to {bc['final_UN']['ci'][1]:.2f})"
    K["bc_full"] = f"λ = {minus(f2(bc['full_UN']['lambda']))} (95 % CI {minus(f2(bc['full_UN']['ci'][0]))} to {minus(f2(bc['full_UN']['ci'][1]))})"
    K["bc_LR1"] = f1(bc["final_UN"]["LR_lambda1"])
    bdel = pd.DataFrame(bc["full_deletion"])
    K["bc_del_inc0"] = int((bdel.hi >= 0).sum())
    cf = R["covariance_full"]
    K["cov_LR"] = f2(cf["LR_UN_vs_CS"]); K["cov_p"] = fp(cf["p"])
    for t in ("D", "D2", "AD", "KA", "KD", "D.Age", "D2.Age", "Carb", "A"):
        K["p_" + t] = fp(tt.loc[t, "p"])
        K["F_" + t] = f"F({int(tt.loc[t, 'df1'])}, {tt.loc[t, 'df2']:.0f}) = {tt.loc[t, 'F']:.2f}"
    K["p_dur"] = fp(R["coding"]["duration_equal_test"]["p"])
    code = pd.DataFrame(R["coding"]["table"]).set_index("coding")
    K["dAICc_onoff"] = minus(f2(code.loc["onoff", "dAICc_vs_cat4"])); K["dAICc_log"] = minus(f2(code.loc["log", "dAICc_vs_cat4"]))
    K["dAICc_num"] = f2(code.loc["numeric", "dAICc_vs_cat4"])
    for key, terms in (("A.Age", "A.Age"), ("C.Age", "C.Age C"), ("Carb.Age", "Carb.Age"), ("C", "C"), ("A2", "A2"),
                       ("C2", "C2 C"), ("AD.Age", "AD.Age A.Age"), ("KA.Age", "KA.Age Carb.Age A.Age"), ("KD.Age", "KD.Age Carb.Age")):
        K["add_" + key] = fp(add.loc[terms, "p"])
    K["add_min_p"] = fp(add.p.min())
    cmask = [any(tok in ("C", "AC", "CD", "C2", "KC", "C.Age") for tok in str(ts).split()) for ts in add.index]
    K["add_C_min"] = rt_p(add.p[cmask].min())
    j = R["age_specific"]["omitted_age_terms_joint"]
    K["omit_joint"] = f"F({j['df1']}, {j['df2']:.0f}) = {j['F']:.2f}, {fp(j['p'])}"
    a = agesp.set_index("column")
    for col in ("KA", "KD", "AD", "Carb[5 h]", "Carb[1 h]", "Carb[0.5 h]", "A"):
        K["chg_" + col] = fp(a.loc[col, "p_change"])
    K["chg_range_int"] = f"{pnum(a.loc[['AD', 'KA', 'KD'], 'p_change'].min())}–{pnum(a.loc[['AD', 'KA', 'KD'], 'p_change'].max())}"
    K["c5_change"] = f2(np.exp(a.loc["Carb[5 h]", "change"]))
    K["c5_change_ci"] = f"{np.exp(a.loc['Carb[5 h]', 'lo_change']):.2f}–{np.exp(a.loc['Carb[5 h]', 'hi_change']):.2f}"
    # gain
    for ss, lab in ((0.0, "0"), (37.5, "37"), (75.0, "75")):
        gg = ex["gain"][ss]
        K["g" + lab] = f2(gg["r"]); K["g" + lab + "_ci"] = f"{gg['lo']:.2f}–{gg['hi']:.2f}"
    K["gmin"] = f2(c["gain_min_SS"]["ratio_at_min"]); K["gmin_ss"] = f"{c['gain_min_SS']['SS']:.0f}"
    K["gmin_ci"] = f"{max(0, c['gain_min_SS']['lo']):.0f}–{min(75, c['gain_min_SS']['hi']):.0f}"
    K["g0v75"] = f2(np.exp(c["gain_SS0_vs_SS75"]["est"])); K["g0v75_ci"] = f"{np.exp(c['gain_SS0_vs_SS75']['lo']):.2f}–{np.exp(c['gain_SS0_vs_SS75']['hi']):.2f}"
    K["g0v75_p"] = fp(c["gain_SS0_vs_SS75"]["p"]); K["g75vmin_p"] = fp(c["gain_SS75_vs_min"]["p"])
    # RCF effects
    def rr(carb, ss):
        r = rcf[(rcf.carbonation == carb) & (rcf.SS == ss) & (rcf.age == 7)].iloc[0]
        return f"×{r.ratio:.2f} (95 % CI {r.lo:.2f}–{r.hi:.2f}; {fp(r.p)})"
    K["rcf_nc0"], K["rcf_nc75"], K["rcf_c0"], K["rcf_c75"] = rr("NC", 0), rr("NC", 75), rr("1 h", 0), rr("1 h", 75)
    def cc(carb, rcf_, ss):
        r = ce[(ce.carbonation == carb) & (ce.RCF == rcf_) & (ce.SS == ss) & (ce.age == 7)].iloc[0]
        return f"×{r.ratio:.2f} (95 % CI {r.lo:.2f}–{r.hi:.2f}; {fp(r.p)})"
    K["carb_50_0"], K["carb_10_75"], K["carb_30_0"] = cc("1 h", 50, 0), cc("1 h", 10, 75), cc("1 h", 30, 0)
    crs = cr[cr.carbonation == "1 h"].set_index("RCF")
    K["cross_1h"] = {int(r): (f"{v:.0f} %" if v is not None and np.isfinite(v) else "none") for r, v in crs.crossover_SS.items()}
    cr5 = cr[cr.carbonation == "5 h"].set_index("RCF")
    K["cross_5h"] = {int(r): (f"{v:.0f} %" if v is not None and np.isfinite(v) else "none") for r, v in cr5.crossover_SS.items()}
    # SS optimum by age
    def so(age, carb, rcf_):
        r = sso[(sso.age == age) & (sso.carbonation == carb) & (sso.RCF == rcf_)].iloc[0]
        return r
    K["sso"] = {(age, carb, r_): so(age, carb, r_) for age in (7, 28) for carb in dz.CARB_LEVELS for r_ in (10, 30, 50)}
    K["opt_boot"] = bs["case"]["opt_ss"]
    # bootstrap
    for mode in ("resid", "subsample"):
        K[mode + "_exact"] = pct(bs[mode]["exact_final"]); K[mode + "_same"] = pct(bs[mode]["same_terms"])
        K[mode + "_gainSS"] = pct(bs[mode]["gain_terms_SS_only"])
        K[mode + "_n"] = f"{bs[mode]['n']:,}"
    K["case_n"] = f"{bs['case']['n']:,}"
    cz = [lab for col, lab in (("KA", "carbonated × RCF"), ("AD", "RCF × SS")) if bs["case"]["beta_ci"][col][0] < 0 < bs["case"]["beta_ci"][col][1]]
    K["case_zero"] = " and ".join(cz) if len(cz) == 2 else ""
    K["min_incl_SS"] = rt_pct_floor(min(bs[m_]["inclusion"][t_] for m_ in ("resid", "subsample") for t_ in ("D", "D2")))
    K["del_same"] = int(pd.DataFrame(R["deletion_selection"]).same_as_final.sum())
    # nested cv
    ncm = ncv[ncv.cv == "leave-one-mixture-out"].set_index("procedure")
    K["ncv_primary7"], K["ncv_primary28"] = f2(ncm.loc["AICc overall (primary)", "predR2_7"]), f2(ncm.loc["AICc overall (primary)", "predR2_28"])
    K["ncv_best28"] = f2(ncm.predR2_28.max())
    K["ncv_perage28"] = f2(ncm.loc["Separate per-age OLS models, AICc (ignores pairing)", "predR2_28"])
    K["ncv_perage7"] = f2(ncm.loc["Separate per-age OLS models, AICc (ignores pairing)", "predR2_7"])
    K["ncv_log7"], K["ncv_log28"] = f2(ncm.loc["AICc overall, log-duration carbonation", "predR2_7"]), f2(ncm.loc["AICc overall, log-duration carbonation", "predR2_28"])
    K["ncv_be7"], K["ncv_be28"] = f2(ncm.loc["Backward a=0.10, UN", "predR2_7"]), f2(ncm.loc["Backward a=0.10, UN", "predR2_28"])
    K["ncv_primary_modal"] = int(ncm.loc["AICc overall (primary)", "final_selected"])
    # diagnostics
    dg = R["diagnostics"]
    K["sw_p"], K["bp_p"], K["bp2_p"] = fp(dg["shapiro_p"]), fp(dg["bp_fitted_age_p"]), fp(dg["bp_factors_p"])
    K["max_t"] = f2(dg["max_abs_t"]); K["max_t_mix"] = dg["max_t_obs"]["mix"]; K["max_t_age"] = dg["max_t_obs"]["age"]
    tab = pd.DataFrame(dg["table"])
    K["max_t_bonf_val"] = float(min(tab.p_bonf7.min(), tab.p_bonf28.min()))
    K["max_t_bonf"] = fp(K["max_t_bonf_val"])
    rowt = tab[tab.mix == K["max_t_mix"]].iloc[0]
    a_ = str(K["max_t_age"])
    K["max_t_obs"], K["max_t_fit"], K["max_t_cv"] = f1(rowt[f"obs{a_}"]), f1(rowt[f"fit{a_}"]), f1(rowt[f"cv{a_}"])
    big28 = tab.reindex(tab.t_del28.abs().sort_values(ascending=False).index).mix.astype(int).tolist()
    K["big28"] = big28[:2]
    K["big28_txt"] = " and ".join(f"mix {m}" for m in big28[:2])
    exc = [r for r in R["robustness"] if " excluded (" in r["analysis"]]
    same = [r["analysis"].endswith("the same model") for r in exc]
    if exc and all(same):
        K["excl_txt"] = ("Excluding either mixture leaves the selected model unchanged." if len(exc) == 2 else
                         "Excluding this mixture leaves the selected model unchanged.")
    else:
        K["excl_txt"] = "Re-selection without them: " + "; ".join(r["analysis"].split("; re-selection gives ")[0].replace("Same terms, ", "")
                                                                  + " → " + r["analysis"].split("; re-selection gives ")[1] for r in exc) + "."
    # change logs
    ch = pd.read_csv(os.path.join(RES, "tables", "T00_changes_vs_previous_7d_data.csv"))
    ch7 = ch[ch.variable == "7-day strength (MPa)"]
    j = ch7.change.abs().idxmax()
    K["t00_big"] = f"the 7-day strength of mix {int(ch7.loc[j, 'mix'])} ({ch7.loc[j, 'previous']:.2f} → {ch7.loc[j, 'corrected']:.2f} MPa)"
    K["t00_n7"] = int(len(ch7))
    v1 = pd.read_csv(os.path.join(RES, "tables", "T00b_changes_vs_previous_corrected_data.csv"))
    K["v1_n"] = int(len(v1))
    m28 = v1[(v1.variable == "28-day strength (MPa)") & (v1.change.abs() > 0.05)]
    m7 = v1[(v1.variable == "7-day strength (MPa)") & (v1.change.abs() > 0.05)]
    r7 = v1[(v1.variable == "7-day strength (MPa)") & (v1.change.abs() <= 0.05)]
    K["v1_28_mixes"] = ", ".join(str(int(x)) for x in m28.mix)
    K["v1_28_n"] = int(len(m28))
    K["v1_28_range"] = f"{minus(f'{m28.change.min():+.2f}')} to {minus(f'{m28.change.max():+.2f}')} MPa"
    K["v1_28_mean"] = minus(f"{m28.change.mean():+.2f}")
    K["v1_7_txt"] = "; ".join(f"mix {int(r.mix)} {r.previous:.2f} → {r.corrected:.2f} MPa" for r in m7.itertuples())
    K["v1_round_n"] = int(len(v1[v1.variable.str.contains('strength') & (v1.change.abs() <= 0.05)]))
    # observed 7 -> 28-day gains
    dd = dz.load()
    gr = dd.f28_mean / dd.f7_mean
    ng = dd[gr < 1.02]
    K["n_gain"] = int((gr >= 1.02).sum())
    K["nogain_txt"] = "; ".join(f"mix {int(r.mix)}, {r.carbonation}, RCF {r.RCF_pct:g} %, SS {r.SS_pct:g} %: {r.f7_mean:.2f} → {r.f28_mean:.2f} MPa"
                                for r in ng.itertuples())
    K["nogain_mixes"] = " and ".join(str(int(m)) for m in ng.mix)
    K["nogain_5h"] = bool(len(ng)) and bool((ng.carbonation == "5 h").all())
    addC = add.loc["Carb.Age"]
    K["add_Carb.Age_val"] = float(addC.p)
    K["add_Carb.Age_dAICc"] = minus(f"{addC.dAICc:+.1f}")
    K["chg_c5_val"] = float(a.loc["Carb[5 h]", "p_change"])
    K["ncv_cs7"], K["ncv_cs28"] = f2(ncm.loc["AICc, CS only", "predR2_7"]), minus(f2(ncm.loc["AICc, CS only", "predR2_28"]))
    # nested-CV procedure with the best 28-day prediction
    b28 = ncm.predR2_28.idxmax()
    K["ncv_best28_proc"] = b28
    K["ncv_best28_7"] = f2(ncm.loc[b28, "predR2_7"])
    # Box-Cox interval of the saturated model
    K["bc_full_excl0"] = bc["full_UN"]["ci"][1] < 0 or bc["full_UN"]["ci"][0] > 0
    # backward elimination at alpha 0.05 relative to the final model
    K["be05_subset"] = set(R["backward"][f"{R['final_struct']}_0.05"]["terms"]) <= set(R["final_terms"])
    K["be05_same"] = set(R["backward"][f"{R['final_struct']}_0.05"]["terms"]) == set(R["final_terms"])
    K["be10_more_age"] = (len([t for t in R["backward"][f"{R['final_struct']}_0.10"]["terms"] if t.endswith(".Age")])
                          > len([t for t in R["final_terms"] if t.endswith(".Age")]))
    # comparison with the same analysis on the previous corrected data set
    with open(os.path.join(RES, "comparison_previous.json")) as f:
        C = json.load(f)
    P_, C_ = C["previous"], C["current"]
    shifts = []
    for col, v in C_["coef"].items():
        if col in P_["coef"]:
            se = (v["hi"] - v["lo"]) / (2 * 1.96)
            shifts.append((abs(v["est"] - P_["coef"][col]["est"]) / se, col, P_["coef"][col]["est"], v["est"]))
    shifts.sort(reverse=True)
    sh = shifts[0]
    lost = [t for t in C_["term_p"] if P_["term_p"].get(t) is not None and P_["term_p"][t] < 0.05 <= C_["term_p"][t]]
    gained = [t for t in C_["term_p"] if P_["term_p"].get(t) is not None and C_["term_p"][t] < 0.05 <= P_["term_p"][t]]
    K["same_model"] = C["same_terms"]
    K["coef_shift"] = sh
    K["chg_intro"] = (
        ("The protocol was re-run unchanged on the corrected data. It selects the same model as before (same terms, "
         f"{C_['final_struct']} covariance)." if C["same_terms"] else
         f"The protocol was re-run unchanged on the corrected data. It now selects {' '.join(C_['final_terms'])} "
         f"({C_['final_struct']}) instead of {' '.join(P_['final_terms'])} ({P_['final_struct']}).")
        + f" The largest coefficient change is {sh[1]} ({minus(f'{sh[2]:.3f}')} → {minus(f'{sh[3]:.3f}')}, {sh[0]:.1f} standard errors)."
        + (f" Terms no longer significant at 5 %: {', '.join(dz.label(t) for t in lost)}." if lost else "")
        + (f" Terms now significant at 5 %: {', '.join(dz.label(t) for t in gained)}." if gained else "")
        + f" Prediction of a left-out mixture: 7 d R² {minus(f2(P_['predR2_7']))} → {minus(f2(C_['predR2_7']))}, "
        f"28 d R² {minus(f2(P_['predR2_28']))} → {minus(f2(C_['predR2_28']))} (RMSE {f1(P_['rmse_pred_MPa_28'])} → {f1(C_['rmse_pred_MPa_28'])} MPa). "
        f"Gain f28/f7 at SS 0 %: {f2(P_['gain_SS0'])} → {f2(C_['gain_SS0'])}; at SS 75 %: {f2(P_['gain_SS75'])} → {f2(C_['gain_SS75'])}.")
    spw = R["specimen_weighting"]
    K["spec_rho28"], K["spec_p28"] = f2(spw["resid_vs_specimen_cv"]["28"]["rho"]), fp(spw["resid_vs_specimen_cv"]["28"]["p"])
    K["spec_rho7"], K["spec_p7"] = f2(spw["resid_vs_specimen_cv"]["7"]["rho"]), fp(spw["resid_vs_specimen_cv"]["7"]["p"])
    K["spec_dAIC"] = minus(f"{spw['dAIC_weighted_minus_unweighted']:+.1f}")
    K["spec_cv7"], K["spec_cv28"] = minus(f2(spw["cv"]["predR2_7"])), minus(f2(spw["cv"]["predR2_28"]))
    sd28 = v1[v1.variable == "28-day SD (MPa)"]
    K["v1_sd28_n"], K["v1_sd28_up"] = int(len(sd28)), int((sd28.change > 0).sum())
    K["cook_max"] = f2(dg["cook_max"]); K["cook_mix"] = dg["cook_max_mix"]
    K["run_rho"], K["run_p"] = f2(dg["run_order_rho"]), fp(dg["run_order_p"])
    # optimum contrasts
    oc = pd.DataFrame(R["optimum_contrast"]).set_index("age")
    K["optc7"] = f"×{oc.loc[7, 'ratio']:.2f} (95 % CI {oc.loc[7, 'lo']:.2f}–{oc.loc[7, 'hi']:.2f}; {fp(oc.loc[7, 'p'])})"
    K["optc28"] = f"×{oc.loc[28, 'ratio']:.2f} (95 % CI {oc.loc[28, 'lo']:.2f}–{oc.loc[28, 'hi']:.2f}; {fp(oc.loc[28, 'p'])})"
    K["se_ind_gain"] = f2(ex["se_ind_ratio"]["D.Age"]); K["se_ind_level"] = f2(ex["se_ind_ratio"]["D"])
    K["sel_lm05_same"] = R["selection_lambda_m05"]["same_as_final"]
    pr = pd.read_csv(os.path.join(RES, "tables", "T23_predictions_all_mixes.csv"))
    for a in (7, 28):
        s = pr[pr.age == a]
        K[f"pi{a}"] = f"{np.median(s.pi_hi / s['median']):.1f}"
    rcf = pd.DataFrame(R["rcf_effect"])
    K["rcf_never_lowers"] = bool((rcf.hi >= 1).all())
    return K
