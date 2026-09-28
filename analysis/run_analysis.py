"""
Complete analysis of the paired 7- and 28-day compressive strengths.

Protocol (fixed before the final model was fitted; see README / report):

 1. Response scale      Box-Cox profile likelihood (joint, both ages, within-mixture
                        covariance unstructured) on the full candidate model and on
                        the selected model; ln is used if lambda = 0 lies inside the
                        95 % CI of the selected model and lambda = 1 outside.
 2. Pairing             every model carries a within-mixture covariance for the 7/28-day
                        pair: CS (random mixture intercept) or UN (random intercept +
                        age-specific residual variance).  Independence structures are
                        reported for reference only.
 3. Carbonation coding  4-level categorical main effect (NC, 0.5, 1, 5 h); carbonation x
                        mixture slopes through the carbonated contrast K (the 0.5 h level
                        has only two distinct compositions).  An alternative coding
                        (on/off, numeric or log duration) replaces it only if its best
                        model has AICc lower by >= 2 AND better leave-one-mixture-out
                        prediction at both ages.
 4. Term selection      exhaustive AICc search over all 53,105 hierarchical models of the
                        full candidate set (quadratic in RCF, A/B, SS + carbonation + K x
                        slopes, all x Age), under CS and under UN; the overall minimum is
                        the final model.  Cross-checks: backward elimination (alpha 0.10 and
                        0.05, hierarchy kept), Akaike weights, nested cross-validation of
                        the selection procedures, single-mixture deletion, bootstraps.
 5. Validation          lack of fit against replicate error, residual diagnostics,
                        leave-one-mixture-out and leave-one-design-point-out prediction,
                        2 x 2,000 bootstrap resamples of the whole selection.
 6. Robustness          alternative scales, covariance, codings, weights, robust fitting,
                        influential mixtures, per-age models.
 7. Interpretation      only after 1-6: contrasts with 95 % CIs, gain ratios, optima.

Run:  python analysis/run_analysis.py            (full: 2 x 2000 bootstrap resamples)
      python analysis/run_analysis.py --quick    (200 resamples, for testing)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import design as dz
import model_tools as mt
import selection as sel
from lmm import PairedLMM

RES = os.path.join(dz.ROOT, "results")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)

FULL = dz.LEVEL_TERMS + ["Age"] + dz.AGE_TERMS
SEED = 20260928
ALPHA_BE = 0.10


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def to_py(o):
    """Recursively convert results to plain JSON types (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [to_py(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return [to_py(r) for r in o.to_dict(orient="records")]
    if isinstance(o, pd.Series):
        return to_py(o.to_dict())
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (frozenset, set, tuple)):
        return list(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    raise TypeError(type(o))


def fit(terms, d, y=None, struct="UN", method="REML", **kw):
    y = d[["l7", "l28"]].values if y is None else y
    X, names = dz.design(terms, d)
    return PairedLMM(y, X, names, struct, method=method, **kw), X, names


def select_overall(d, coding="cat4", structs=("CS", "UN"), check_rank=False, y=None):
    """Exhaustive AICc search under each covariance structure; overall minimum."""
    dz.set_coding(coding)
    if y is None:
        M, G = d.M.values, d.G.values
    else:
        M, G = (y[:, 0] + y[:, 1]) / 2, y[:, 1] - y[:, 0]
    cols = sel.mix_columns(d)
    best = None
    for st in structs:
        top, _ = enumerate_ranked(M, G, cols, st, 1, check_rank)
        a, m_, g, k = top[0]
        if best is None or a < best[0]:
            best = (a, sel.joint_terms(m_, g), st)
    dz.set_coding("cat4")
    return best


_SUBSETS = None


def enumerate_ranked(M, G, cols, struct, n_top, check_rank=False):
    """enumerate_models, optionally dropping rank-deficient subsets (case bootstrap)."""
    global _SUBSETS
    if _SUBSETS is None:
        _SUBSETS = sel.hierarchical_level_subsets()
    subsets = _SUBSETS
    if check_rank:
        n = len(M)
        ok = []
        for s in subsets:
            X = sel.mix_X(s, cols, n)
            Xc = np.column_stack([X, G]) if struct == "UN" else X
            if np.linalg.matrix_rank(Xc) == Xc.shape[1]:
                ok.append(s)
        subsets = ok
    return sel.enumerate_models(M, G, cols, struct, n_top, subsets=subsets)


# ============================================================================ 1. scale
def boxcox(d, lev, gain, struct="UN", lams=np.round(np.arange(-1.5, 1.501, 0.01), 2)):
    Yr = d[["f7_mean", "f28_mean"]].values
    cols = sel.mix_columns(d)
    L = []
    for lam in lams:
        Z = np.log(Yr) if abs(lam) < 1e-9 else (Yr ** lam - 1) / lam
        L.append(sel.loglik_ml(Z, lev, gain, cols, struct)[0] + (lam - 1) * np.log(Yr).sum())
    L = np.array(L)
    cut = L.max() - stats.chi2.ppf(0.95, 1) / 2
    ci = lams[L >= cut]
    i0, i1 = np.argmin(np.abs(lams)), np.argmin(np.abs(lams - 1))
    return {"lambda": float(lams[L.argmax()]), "ci": [float(ci.min()), float(ci.max())],
            "LR_lambda1": float(2 * (L.max() - L[i1])), "LR_lambda0": float(2 * (L.max() - L[i0])),
            "p_lambda1": float(stats.chi2.sf(2 * (L.max() - L[i1]), 1)),
            "p_lambda0": float(stats.chi2.sf(2 * (L.max() - L[i0]), 1)),
            "curve": {"lambda": lams.tolist(), "loglik": (L - L.max()).tolist()}, "cut": float(cut - L.max())}


# ============================================================================ helpers
def cv_predict(d, terms, struct="UN", groups="mix", y=None, coding="cat4", select=None):
    """Leave-one-group-out predictions (ln scale, n x 2).  If `select` is given it is
    called on the training data to choose the terms (nested CV of a procedure)."""
    dz.set_coding(coding)
    Y = d[["l7", "l28"]].values if y is None else y
    keys = d.mix.values if groups == "mix" else d.group.values
    P = np.zeros_like(Y)
    chosen = []
    for gk in np.unique(keys):
        te = keys == gk
        tr = ~te
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        t_use, st_use = terms, struct
        if select is not None:
            t_use, st_use = select(dtr, Y[tr])
            dz.set_coding(coding)
        chosen.append(tuple(t_use) + (st_use,))
        X, names = dz.design(t_use, dtr)
        Xt, _ = dz.design(t_use, dte)
        m = PairedLMM(Y[tr], X, names, st_use, hessian=False)
        P[te] = np.einsum("iap,p->ia", Xt, m.beta)
    dz.set_coding("cat4")
    return P, chosen


def cv_metrics(Y, P, scale="ln"):
    e = Y - P
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    out = {}
    for j, a in enumerate(("7", "28")):
        out[f"predR2_{a}"] = float(1 - (e[:, j] ** 2).sum() / sst[j])
        out[f"rmse_ln_{a}"] = float(np.sqrt((e[:, j] ** 2).mean()))
        if scale == "ln":
            em = np.exp(Y[:, j]) - np.exp(P[:, j])
        else:
            em = Y[:, j] - P[:, j]
        out[f"rmse_MPa_{a}"] = float(np.sqrt((em ** 2).mean()))
        out[f"mae_MPa_{a}"] = float(np.mean(np.abs(em)))
    return out


def term_table(m, names, terms):
    rows = []
    for t in terms:
        if t == "Age":
            L = sel.term_L("Age", names)
        else:
            L = sel.term_L(t, names)
        F, q, df, p = m.f_test(L)
        rows.append({"term": t, "label": dz.label(t), "df1": q, "df2": df, "F": F, "p": p})
    return pd.DataFrame(rows)


def coef_table(m, names):
    rows = []
    for j, nm in enumerate(names):
        l = np.zeros(len(names)); l[j] = 1
        est, se, df, t, p = m.t_test(l)
        q = stats.t.ppf(0.975, df)
        rows.append({"column": nm, "term": dz.term_of_column(nm), "coef": est, "se": se, "df": df,
                     "t": t, "p": p, "lo": est - q * se, "hi": est + q * se})
    return pd.DataFrame(rows)


# ============================================================================ bootstrap worker
_BOOT = {}


def _boot_init(payload):
    _BOOT.update(payload)


def _boot_one(args):
    """One resample.  mode = 'resid'     : residual-pair bootstrap on the fixed design; full selection + refit
                           'case'      : case bootstrap of mixtures, stratified by carbonation level; refit of the final structure only
                                         (duplicated mixtures make AICc favour near-saturated models, so the
                                         case bootstrap is not used for selection)
                           'subsample' : 24 of 30 mixtures without replacement; full selection"""
    seed, mode = args
    rng = np.random.default_rng(seed)
    d = _BOOT["d"]
    final = _BOOT["final"]
    fitted, resid_mod = _BOOT["fitted"], _BOOT["resid_mod"]
    n = len(d)
    terms, st = None, None
    for _attempt in range(50):
        if mode == "resid":
            idx = rng.integers(0, n, n)
            ystar = fitted + resid_mod[idx]
            db = d
        elif mode == "case":
            # stratified by carbonation level: keeps the design's number of mixtures per level
            idx = np.concatenate([rng.choice(ii, size=len(ii), replace=True) for ii in _BOOT["strata"]])
            db = d.iloc[idx].reset_index(drop=True)
            ystar = db[["l7", "l28"]].values
        else:
            idx = np.sort(rng.choice(n, size=_BOOT["n_sub"], replace=False))
            db = d.iloc[idx].reset_index(drop=True)
            ystar = db[["l7", "l28"]].values
        Xf, names = dz.design(final, db)
        XM, _ = dz.mix_design(dz.split_level_gain(final)[0], db)
        if np.linalg.matrix_rank(XM) < XM.shape[1]:
            continue
        try:
            if mode in ("resid", "subsample"):
                a, terms, st = select_overall(db, check_rank=(mode == "subsample"), y=ystar)
            m = PairedLMM(ystar, Xf, names, "UN", hessian=False)
        except Exception:
            continue
        break
    else:
        return None
    pts = _BOOT["gain_pts"]
    gr = mt.gain_rows(final, pts) @ m.beta
    opt = {}
    for key, P in _BOOT["opt_grids"].items():
        v = P @ m.beta
        opt[key] = float(_BOOT["opt_ss"][int(np.argmax(v))])
    return {"terms": terms, "struct": st, "beta": m.beta.tolist(),
            "sigma": [m.Sigma[0, 0], m.Sigma[1, 1], m.Sigma[0, 1]],
            "gain": gr.tolist(), "opt": opt}


# ============================================================================ main
def main(quick=False, n_boot=None, procs=4):
    t_start = time.time()
    n_boot = n_boot or (200 if quick else 2000)
    rng = np.random.default_rng(SEED)
    d = dz.load()
    Y = d[["l7", "l28"]].values
    R = {"meta": {"n_boot": n_boot, "seed": SEED, "alpha_backward": ALPHA_BE,
                  "date": time.strftime("%Y-%m-%d"), "numpy": np.__version__,
                  "statsmodels": sm.__version__ if hasattr(sm, "__version__") else "", "pandas": pd.__version__}}
    with open(os.path.join(RES, "data_summary.json")) as f:
        R["data"] = json.load(f)
    cols = sel.mix_columns(d)

    # ------------------------------------------------------------------ 2. covariance (full model)
    log("covariance structures on the full candidate model")
    cov_rows = []
    for st in ("IND", "INDH", "CS", "UN"):
        m, _, _ = fit(FULL, d, struct=st)
        k = len(m.theta)
        vc = m.variance_components()
        cov_rows.append({"structure": st, "pairing_modelled": st in ("CS", "UN"), "reml_loglik": m.ll,
                         "n_cov_par": k, "AIC_REML": -2 * m.ll + 2 * k,
                         "sd7": vc["sd_7"], "sd28": vc["sd_28"], "rho": vc["rho"]})
    cov_full = pd.DataFrame(cov_rows)
    un, cs = cov_full.set_index("structure").loc["UN"], cov_full.set_index("structure").loc["CS"]
    lr = 2 * (un.reml_loglik - cs.reml_loglik)
    R["covariance_full"] = {"table": cov_full, "LR_UN_vs_CS": lr, "p": stats.chi2.sf(lr, 1)}
    cov_full.to_csv(os.path.join(TAB, "T03_covariance_structures_full_model.csv"), index=False)

    # ------------------------------------------------------------------ 3/4. exhaustive search
    log("exhaustive AICc search (CS and UN)")
    search = {}
    for st in ("CS", "UN"):
        res, ntot = enumerate_ranked(d.M.values, d.G.values, cols, st, None)
        A = np.array([r[0] for r in res])
        w = np.exp(-(A - A.min()) / 2); w /= w.sum()
        imp = Counter()
        for (a, m_, g, k), wi in zip(res, w):
            for t in m_: imp[t] += wi
            for t in g: imp[t + ".Age"] += wi
        top = [{"rank": i + 1, "AICc": r[0], "dAICc": r[0] - A.min(), "weight": float(w[i]), "k": r[3],
                "level_terms": " ".join(sorted(r[1], key=dz.LEVEL_TERMS.index)),
                "gain_terms": " ".join(sorted(r[2], key=dz.LEVEL_TERMS.index)),
                "terms": sel.joint_terms(r[1], r[2])} for i, r in enumerate(res[:25])]
        search[st] = {"n_models": ntot, "top": top, "importance": {t: float(imp.get(t, 0.0)) for t in dz.ALL_TERMS},
                      "n_within_2": int((A - A.min() < 2).sum()), "n_within_4": int((A - A.min() < 4).sum())}
    best_st = min(("CS", "UN"), key=lambda s: search[s]["top"][0]["AICc"])
    final = search[best_st]["top"][0]["terms"]
    STRUCT = best_st
    R["search"] = search
    R["final_terms"] = final
    R["final_struct"] = STRUCT
    log(f"final model ({STRUCT}): {final}")
    pd.DataFrame([{k: v for k, v in r.items() if k != "terms"} | {"structure": st}
                  for st in ("UN", "CS") for r in search[st]["top"][:20]]).to_csv(
        os.path.join(TAB, "T05_top_models_exhaustive_AICc.csv"), index=False)
    pd.DataFrame([{"term": t, "label": dz.label(t), "importance_UN": search["UN"]["importance"][t],
                   "importance_CS": search["CS"]["importance"][t]} for t in dz.ALL_TERMS]).sort_values(
        "importance_UN", ascending=False).to_csv(os.path.join(TAB, "T06_akaike_term_importance.csv"), index=False)

    lev_f, gain_f = dz.split_level_gain(final)

    # ------------------------------------------------------------------ 1. Box-Cox
    log("Box-Cox")
    R["boxcox"] = {"full_UN": boxcox(d, dz.LEVEL_TERMS, dz.LEVEL_TERMS, "UN"),
                   "full_CS": boxcox(d, dz.LEVEL_TERMS, dz.LEVEL_TERMS, "CS"),
                   "final_UN": boxcox(d, lev_f, gain_f, "UN"),
                   "final_CS": boxcox(d, lev_f, gain_f, "CS")}
    # sensitivity of the full-model interval to single mixtures
    bc_del = []
    for i in d.mix:
        dd = d[d.mix != i].reset_index(drop=True)
        b = boxcox(dd, dz.LEVEL_TERMS, dz.LEVEL_TERMS, "UN", lams=np.round(np.arange(-1.5, 1.501, 0.05), 2))
        bc_del.append({"mix_removed": int(i), "lambda": b["lambda"], "lo": b["ci"][0], "hi": b["ci"][1]})
    R["boxcox"]["full_deletion"] = bc_del
    pd.DataFrame([{"model": k, "lambda": v["lambda"], "ci_lo": v["ci"][0], "ci_hi": v["ci"][1],
                   "LR_lambda1": v["LR_lambda1"], "p_lambda1": v["p_lambda1"],
                   "LR_lambda0": v["LR_lambda0"], "p_lambda0": v["p_lambda0"]}
                  for k, v in R["boxcox"].items() if k != "full_deletion"]).to_csv(
        os.path.join(TAB, "T02_boxcox.csv"), index=False)

    # ------------------------------------------------------------------ carbonation coding
    log("carbonation coding comparison")
    coding_rows = []
    for cod in ("cat4", "onoff", "numeric", "log"):
        dz.set_coding(cod)
        c_cols = sel.mix_columns(d)
        best = None
        for st in ("CS", "UN"):
            top, _ = sel.enumerate_models(d.M.values, d.G.values, c_cols, st, 1)
            if best is None or top[0][0] < best[0]:
                best = (top[0][0], sel.joint_terms(top[0][1], top[0][2]), st)
        m_same, _, _ = fit(final, d, struct=STRUCT)
        a_same = m_same.aicc_ml()[0]
        P_same, _ = cv_predict(d, final, STRUCT, coding=cod)
        P_best, _ = cv_predict(d, best[1], best[2], coding=cod)
        dz.set_coding("cat4")
        coding_rows.append({"coding": cod, "AICc_final_structure": a_same,
                            **{f"final_{k}": v for k, v in cv_metrics(Y, P_same).items() if k.startswith("predR2")},
                            "best_AICc": best[0], "best_structure": best[2],
                            "best_terms": " ".join(best[1]),
                            **{f"best_{k}": v for k, v in cv_metrics(Y, P_best).items() if k.startswith("predR2")}})
    coding = pd.DataFrame(coding_rows)
    ref = coding.set_index("coding").loc["cat4"]
    for i, r in coding.iterrows():
        coding.loc[i, "dAICc_vs_cat4"] = r.best_AICc - ref.best_AICc
        coding.loc[i, "switch_rule_met"] = bool(r.coding != "cat4" and r.best_AICc <= ref.best_AICc - 2 and
                                                r.best_predR2_7 > ref.best_predR2_7 and r.best_predR2_28 > ref.best_predR2_28)
    mfin, Xf, names = fit(final, d, struct=STRUCT)
    ic = [names.index(f"Carb[{l}]") for l in dz.CARB_LEVELS[1:]]
    L = np.zeros((2, len(names))); L[0, ic[1]] = 1; L[0, ic[0]] = -1; L[1, ic[2]] = 1; L[1, ic[0]] = -1
    F, q, df, p = mfin.f_test(L)
    R["coding"] = {"table": coding, "duration_equal_test": {"F": F, "df1": q, "df2": df, "p": p},
                   "switch": bool(coding.switch_rule_met.any())}
    coding.to_csv(os.path.join(TAB, "T04_carbonation_coding.csv"), index=False)
    if R["coding"]["switch"]:
        log("WARNING: coding switch rule met; primary coding kept for reproducibility - inspect T04")

    # ------------------------------------------------------------------ backward elimination
    log("backward elimination cross-checks")
    be = {}
    for st in ("UN", "CS"):
        for a in (0.10, 0.05):
            terms, hist = sel.backward_eliminate(d, FULL, Y, struct=st, alpha=a)
            m_be, _, _ = fit(terms, d, struct=st)
            be[f"{st}_{a:.2f}"] = {"terms": terms, "history": hist, "AICc": m_be.aicc_ml()[0],
                                   "same_as_final": set(terms) == set(final)}
    R["backward"] = be
    pd.DataFrame([{"run": k, "step": i + 1, **h} for k, v in be.items() for i, h in enumerate(v["history"])]).to_csv(
        os.path.join(TAB, "T07_backward_elimination.csv"), index=False)

    # ------------------------------------------------------------------ final model
    log("final model")
    m = mfin
    R["final"] = {"names": names, "beta": m.beta, "cov_beta": m.cov_beta, "Sigma": m.Sigma,
                  "variance_components": m.variance_components(), "theta": m.theta,
                  "coef": coef_table(m, names), "terms_F": term_table(m, names, final)}
    aicc, ll_ml, k = m.aicc_ml()
    R["final"]["AICc"], R["final"]["k"] = aicc, k
    fitted = np.einsum("iap,p->ia", Xf, m.beta)
    resid = Y - fitted
    stats_age = {}
    for j, a in enumerate(("7", "28")):
        sst = ((Y[:, j] - Y[:, j].mean()) ** 2).sum()
        s2 = m.Sigma[j, j]
        se_fit = np.sqrt(np.einsum("ip,pq,iq->i", Xf[:, j], m.cov_beta, Xf[:, j]))
        stats_age[a] = {"R2": float(1 - (resid[:, j] ** 2).sum() / sst),
                        "adjR2": float(1 - s2 / (sst / (len(d) - 1))),
                        "sd_resid_ln": float(np.sqrt(s2)),
                        "cv_pct": float(100 * np.sqrt(np.exp(s2) - 1)),
                        "rmse_fit_MPa": float(np.sqrt(np.mean((np.exp(Y[:, j]) - np.exp(fitted[:, j])) ** 2))),
                        "adeq_precision": float((fitted[:, j].max() - fitted[:, j].min()) / np.sqrt(np.mean(se_fit ** 2))),
                        "pure_error_sd_ln": R["data"]["pure_error"][f"{a} d ln"]["sd"]}
    R["final"]["fit_stats"] = stats_age
    coef_table(m, names).to_csv(os.path.join(TAB, "T09_final_model_coefficients_coded.csv"), index=False)
    term_table(m, names, final).to_csv(os.path.join(TAB, "T10_final_model_term_tests.csv"), index=False)

    # actual-unit equations
    eqs = mt.actual_equations(m, final)
    R["final"]["actual"] = eqs
    R["final"]["equations_text"] = {k: mt.format_equation(v, 6, {"7 d": "ln(f7)", "28 d": "ln(f28)",
                                                                  "gain": "ln(f28/f7)"}[k]) for k, v in eqs.items()}
    pd.DataFrame([{"equation": k, **c} for k, v in eqs.items() for c in v]).to_csv(
        os.path.join(TAB, "T11_final_model_actual_units.csv"), index=False)

    # lack of fit (exact tests via the level/gain factorisation)
    DP = pd.get_dummies(d.group).values.astype(float)
    XM, _ = dz.mix_design(lev_f, d)
    XG, _ = dz.mix_design(gain_f, d)

    def lof(y, X, extra=None):
        Xf_ = X if extra is None else np.column_stack([X, extra])
        Xs = DP if extra is None else np.column_stack([DP, extra])
        _, _, _, rss, dfe = sel._ols(y, Xf_)
        _, _, _, rss_s, df_s = sel._ols(y, Xs)
        F = ((rss - rss_s) / (dfe - df_s)) / (rss_s / df_s)
        return {"F": F, "df1": dfe - df_s, "df2": df_s, "p": stats.f.sf(F, dfe - df_s, df_s),
                "sd_resid": np.sqrt(rss / dfe), "sd_pure": np.sqrt(rss_s / df_s)}
    Xu = sel.mix_X(sorted(set(lev_f) | set(gain_f), key=dz.LEVEL_TERMS.index), cols, len(d))
    R["final"]["lack_of_fit"] = {
        "gain (G | X_G)": lof(d.G.values, XG),
        "level given gain (M | X_M, G)": lof(d.M.values, XM, d.G.values) if STRUCT == "UN" else lof(d.M.values, XM),
        "7 d (per-age OLS, same terms)": lof(d.l7.values, Xu),
        "28 d (per-age OLS, same terms)": lof(d.l28.values, Xu)}

    # ------------------------------------------------------------------ age-specific (fully interacted) version
    log("age-specific version and added-term tests")
    full_int = lev_f + ["Age"] + [t + ".Age" for t in lev_f]
    mi, Xi, ni = fit(full_int, d, struct=STRUCT)
    rows = []
    for t in lev_f:
        Lt = sel.term_L(t, ni)
        La = sel.term_L(t + ".Age", ni)
        for r_ in range(Lt.shape[0]):
            col = ni[np.argmax(Lt[r_])]
            l7, l28 = Lt[r_] - 0.5 * La[r_], Lt[r_] + 0.5 * La[r_]
            e7 = mt.lincomb(mi, l7).iloc[0]; e28 = mt.lincomb(mi, l28).iloc[0]; ed = mt.lincomb(mi, La[r_]).iloc[0]
            rows.append({"term": t, "column": col, "label": dz.label(t) if t != "Carb" else col,
                         "eff7": e7.est, "lo7": e7.lo, "hi7": e7.hi, "p7": e7.p,
                         "eff28": e28.est, "lo28": e28.lo, "hi28": e28.hi, "p28": e28.p,
                         "change": ed.est, "lo_change": ed.lo, "hi_change": ed.hi, "p_change": ed.p,
                         "in_final_as_age_term": (t + ".Age") in final})
    agespec = pd.DataFrame(rows)
    joint = {}
    for t in lev_f:
        F, q, df, p = mi.f_test(sel.term_L(t + ".Age", ni))
        joint[t] = {"F": F, "df1": q, "df2": df, "p": p}
    notin = [t + ".Age" for t in lev_f if t + ".Age" not in final]
    Lj = np.vstack([sel.term_L(t, ni) for t in notin]) if notin else None
    R["age_specific"] = {"table": agespec, "term_change_tests": joint,
                         "omitted_age_terms_joint": dict(zip(("F", "df1", "df2", "p"), mi.f_test(Lj))) if notin else None,
                         "AICc": mi.aicc_ml()[0], "Sigma": mi.Sigma}
    agespec.to_csv(os.path.join(TAB, "T12_age_specific_effects_coded.csv"), index=False)

    added = []
    for t in dz.ALL_TERMS:
        if t in final:
            continue
        add = [t]; ch = True
        while ch:
            ch = False
            for u in list(add):
                for p_ in dz.parents(u):
                    if p_ not in final and p_ not in add:
                        add.append(p_); ch = True
        m2, _, n2 = fit(final + add, d, struct=STRUCT)
        F, q, df, p = m2.f_test(np.vstack([sel.term_L(u, n2) for u in add]))
        added.append({"added": " + ".join(dz.label(u) for u in add), "terms": " ".join(add), "F": F, "df1": q,
                      "df2": df, "p": p, "AICc": m2.aicc_ml()[0], "dAICc": m2.aicc_ml()[0] - aicc})
    R["added_terms"] = pd.DataFrame(added)
    R["added_terms"].to_csv(os.path.join(TAB, "T13_added_term_tests.csv"), index=False)

    # ------------------------------------------------------------------ diagnostics
    log("diagnostics")
    w = m.whitened_residuals()
    hat = m.hat_diag()
    del_t, cook, mixF, loo_beta = np.zeros_like(Y), np.zeros(len(d)), np.zeros(len(d)), []
    p_ = len(names)
    Vinv_beta = np.linalg.inv(m.cov_beta)
    df_del = len(d) - 1 - XM.shape[1] - (1 if STRUCT == "UN" else 0)
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        mm = PairedLMM(Y[tr], Xf[tr], names, STRUCT, hessian=False)
        loo_beta.append(mm.beta)
        e = Y[i] - Xf[i] @ mm.beta
        V = Xf[i] @ mm.cov_beta @ Xf[i].T + mm.Sigma
        del_t[i] = e / np.sqrt(np.diag(V))
        T2 = e @ np.linalg.solve(V, e)
        mixF[i] = T2 * (df_del - 1) / (2 * df_del)
        db = m.beta - mm.beta
        cook[i] = db @ Vinv_beta @ db / p_
    p_del = 2 * stats.t.sf(np.abs(del_t), df_del)
    p_mix = stats.f.sf(mixF, 2, df_del - 1)
    wf = w.reshape(-1)
    sw = stats.shapiro(wf)
    # heteroscedasticity: squared whitened residuals vs fitted value and age
    Z = np.column_stack([np.ones(60), fitted.reshape(-1), np.tile([0, 1], 30)])
    bp_fit = sm.OLS(wf ** 2, Z).fit()
    bp_lm = 60 * bp_fit.rsquared
    # vs factors
    Zf = np.column_stack([np.ones(60), np.repeat(d.A.values, 2), np.repeat(d.D.values, 2),
                          np.repeat(d.C.values, 2), np.repeat(d.K.values, 2)])
    bp2 = sm.OLS(wf ** 2, Zf).fit()
    rho_run = stats.spearmanr(d.mix.values, np.abs(w).mean(1))
    dw = [float(np.sum(np.diff(w[:, j]) ** 2) / np.sum(w[:, j] ** 2)) for j in (0, 1)]
    diag = pd.DataFrame({"mix": d.mix, "carbonation": d.carbonation, "RCF": d.RCF_pct, "SS": d.SS_pct, "AB": d.AB,
                         "obs7": d.f7_mean, "fit7": np.exp(fitted[:, 0]), "obs28": d.f28_mean,
                         "fit28": np.exp(fitted[:, 1]), "resid_ln7": resid[:, 0], "resid_ln28": resid[:, 1],
                         "white7": w[:, 0], "white28": w[:, 1], "hat7": hat[:, 0], "hat28": hat[:, 1],
                         "t_del7": del_t[:, 0], "t_del28": del_t[:, 1],
                         "p_bonf7": np.minimum(1, p_del[:, 0] * 60), "p_bonf28": np.minimum(1, p_del[:, 1] * 60),
                         "mix_F": mixF, "p_mix_bonf": np.minimum(1, p_mix * 30), "cook": cook})
    R["diagnostics"] = {"table": diag, "shapiro_W": sw.statistic, "shapiro_p": sw.pvalue,
                        "bp_fitted_age_LM": bp_lm, "bp_fitted_age_p": stats.chi2.sf(bp_lm, 2),
                        "bp_factors_LM": 60 * bp2.rsquared, "bp_factors_p": stats.chi2.sf(60 * bp2.rsquared, 4),
                        "run_order_rho": rho_run.statistic, "run_order_p": rho_run.pvalue, "durbin_watson": dw,
                        "df_deleted_t": df_del, "max_abs_t": float(np.abs(del_t).max()),
                        "max_t_obs": {"mix": int(d.mix.values[np.unravel_index(np.abs(del_t).argmax(), del_t.shape)[0]]),
                                      "age": [7, 28][np.unravel_index(np.abs(del_t).argmax(), del_t.shape)[1]]},
                        "cook_max": float(cook.max()), "cook_max_mix": int(d.mix.values[cook.argmax()]),
                        "hat_max": float(hat.max()),
                        "hat_max_obs": {"mix": int(d.mix.values[np.unravel_index(hat.argmax(), hat.shape)[0]]),
                                        "age": [7, 28][np.unravel_index(hat.argmax(), hat.shape)[1]]}}
    diag.to_csv(os.path.join(TAB, "T14_diagnostics_per_mix.csv"), index=False)
    loo_beta = np.array(loo_beta)
    R["loo_beta"] = {"names": names, "beta": loo_beta, "full": m.beta}

    # ------------------------------------------------------------------ validation
    log("validation (cross-validation)")
    val = {}
    for g in ("mix", "group"):
        P, _ = cv_predict(d, final, STRUCT, groups=g)
        val[g] = cv_metrics(Y, P)
        val[g + "_pred"] = P
    val["fit"] = cv_metrics(Y, fitted)
    diag["cv7"] = np.exp(val["mix_pred"][:, 0]); diag["cv28"] = np.exp(val["mix_pred"][:, 1])
    diag["cvg7"] = np.exp(val["group_pred"][:, 0]); diag["cvg28"] = np.exp(val["group_pred"][:, 1])
    diag.to_csv(os.path.join(TAB, "T14_diagnostics_per_mix.csv"), index=False)

    # nested CV of selection procedures
    log("nested cross-validation of selection procedures")
    def proc_overall(dtr, ytr):
        a, t, st = select_overall(dtr, y=ytr)
        return t, st

    def proc_struct(stname):
        def f(dtr, ytr):
            a, t, st = select_overall(dtr, structs=(stname,), y=ytr)
            return t, st
        return f

    def proc_be(stname, alpha):
        def f(dtr, ytr):
            c = sel.mix_columns(dtr)
            M_, G_ = (ytr[:, 0] + ytr[:, 1]) / 2, ytr[:, 1] - ytr[:, 0]
            if stname == "UN":
                return sel.fast_backward_un(M_, G_, c, FULL, alpha), "UN"
            return sel.fast_backward(M_, G_, c, FULL, alpha), "CS"
        return f
    def proc_coding(cod):
        def f(dtr, ytr):
            a, t, st = select_overall(dtr, coding=cod, y=ytr)
            return t, st
        return f
    procs_cv = {"AICc overall (primary)": (proc_overall, "cat4"), "AICc, UN only": (proc_struct("UN"), "cat4"),
                "AICc, CS only": (proc_struct("CS"), "cat4"), "Backward a=0.10, UN": (proc_be("UN", 0.10), "cat4"),
                "Backward a=0.05, UN": (proc_be("UN", 0.05), "cat4"), "Backward a=0.10, CS": (proc_be("CS", 0.10), "cat4"),
                "AICc overall, on/off carbonation": (proc_coding("onoff"), "onoff"),
                "AICc overall, log-duration carbonation": (proc_coding("log"), "log"),
                "AICc overall, numeric-duration carbonation": (proc_coding("numeric"), "numeric")}
    subsets_all = sel.hierarchical_level_subsets()

    def per_age_nested(g):
        """separate OLS models per age, each AICc-selected inside every fold"""
        keys = d.mix.values if g == "mix" else d.group.values
        P = np.zeros_like(Y)
        for gk in np.unique(keys):
            te = keys == gk; tr = ~te
            ctr = sel.mix_columns(d[tr].reset_index(drop=True)); cte = sel.mix_columns(d[te].reset_index(drop=True))
            ntr = int(tr.sum())
            for j in range(2):
                best = None
                for s_ in subsets_all:
                    Xs = sel.mix_X(s_, ctr, ntr); k_ = Xs.shape[1] + 1
                    if ntr - k_ - 1 <= 0 or np.linalg.matrix_rank(Xs) < Xs.shape[1]:
                        continue
                    ac = -2 * sel.ols_ll(Y[tr, j], Xs) + 2 * k_ + 2 * k_ * (k_ + 1) / (ntr - k_ - 1)
                    if best is None or ac < best[0]:
                        best = (ac, s_)
                b_ = np.linalg.lstsq(sel.mix_X(best[1], ctr, ntr), Y[tr, j], rcond=None)[0]
                P[te, j] = sel.mix_X(best[1], cte, int(te.sum())) @ b_
        return P
    ncv = []
    for g in ("mix", "group"):
        for nm, (f, cod) in procs_cv.items():
            P, chosen = cv_predict(d, None, None, groups=g, select=f, coding=cod)
            c = Counter(chosen)
            ncv.append({"cv": "leave-one-mixture-out" if g == "mix" else "leave-one-design-point-out",
                        "procedure": nm, **cv_metrics(Y, P), "n_folds": len(chosen), "distinct_models": len(c),
                        "modal_count": c.most_common(1)[0][1],
                        "final_selected": sum(1 for ch in chosen if set(ch[:-1]) == set(final) and ch[-1] == STRUCT)})
        P = per_age_nested(g)
        ncv.append({"cv": "leave-one-mixture-out" if g == "mix" else "leave-one-design-point-out",
                    "procedure": "Separate per-age OLS models, AICc (ignores pairing)", **cv_metrics(Y, P),
                    "n_folds": np.nan, "distinct_models": np.nan, "modal_count": np.nan, "final_selected": np.nan})
        for nm, terms_, st_ in [("Full candidate model (no selection)", FULL, "UN"), ("Final model (fixed terms)", final, STRUCT)]:
            P, _ = cv_predict(d, terms_, st_, groups=g)
            ncv.append({"cv": "leave-one-mixture-out" if g == "mix" else "leave-one-design-point-out",
                        "procedure": nm, **cv_metrics(Y, P), "n_folds": np.nan, "distinct_models": np.nan,
                        "modal_count": np.nan, "final_selected": np.nan})
    R["nested_cv"] = pd.DataFrame(ncv)
    R["validation"] = val
    R["nested_cv"].to_csv(os.path.join(TAB, "T08_nested_cross_validation.csv"), index=False)
    pd.DataFrame([{"validation": k, **v} for k, v in val.items() if not k.endswith("_pred")]).to_csv(
        os.path.join(TAB, "T15_validation_final_model.csv"), index=False)

    # selection with each mixture deleted
    del_sel = []
    for i in d.mix:
        dd = d[d.mix != i].reset_index(drop=True)
        a, t, st = select_overall(dd)
        del_sel.append({"mix_removed": int(i), "same_as_final": set(t) == set(final) and st == STRUCT,
                        "terms": " ".join(t), "structure": st})
    R["deletion_selection"] = pd.DataFrame(del_sel)

    # ------------------------------------------------------------------ contrasts / interpretation quantities
    log("contrasts")
    ss_grid = np.round(np.arange(0, 75.01, 2.5), 2)
    gpts = pd.DataFrame({"RCF_pct": 30.0, "SS_pct": ss_grid, "AB": 0.45, "carbonation": "NC", "age": 28})
    Lg = mt.gain_rows(final, gpts)
    gtab = mt.lincomb(m, Lg)
    gtab.insert(0, "SS", ss_grid)
    for c_ in ("est", "lo", "hi"):
        gtab["ratio_" + c_] = np.exp(gtab[c_])
    R["gain_curve"] = gtab
    # does the gain depend on anything besides SS? (only if gain terms other than D, D2)
    # minimum-gain SS (quadratic in D): delta method
    contr = {}
    if "D2.Age" in final and "D.Age" in final:
        jD, jDD = names.index("D.Age"), names.index("D2.Age")
        gD, gDD = m.beta[jD], m.beta[jDD]
        Dstar = -gD / (2 * gDD)
        grad = np.zeros(len(names)); grad[jD] = -1 / (2 * gDD); grad[jDD] = gD / (2 * gDD ** 2)
        seD = np.sqrt(grad @ m.cov_beta @ grad)
        contr["gain_min_SS"] = {"SS": 37.5 + 37.5 * Dstar, "se_SS": 37.5 * seD,
                                "lo": 37.5 + 37.5 * (Dstar - 1.96 * seD), "hi": 37.5 + 37.5 * (Dstar + 1.96 * seD),
                                "ratio_at_min": float(np.exp(m.beta[names.index('Age')] + gD * Dstar + gDD * Dstar ** 2))}
    # gain ratio at selected SS and differences (SS 0 vs 37.5 vs 75)
    pts3 = pd.DataFrame({"RCF_pct": 30.0, "SS_pct": [0, 37.5, 75], "AB": 0.45, "carbonation": "NC", "age": 28})
    L3 = mt.gain_rows(final, pts3)
    contr["gain_SS0_vs_SS75"] = mt.lincomb(m, L3[0] - L3[2]).iloc[0].to_dict()
    contr["gain_SS75_vs_min"] = None
    if "gain_min_SS" in contr:
        pmin = pd.DataFrame({"RCF_pct": 30.0, "SS_pct": [contr["gain_min_SS"]["SS"]], "AB": 0.45,
                             "carbonation": "NC", "age": 28})
        contr["gain_SS75_vs_min"] = mt.lincomb(m, L3[2] - mt.gain_rows(final, pmin)[0]).iloc[0].to_dict()

    # RCF effect (50 vs 10 %) by carbonation state and SS - identical at both ages if no RCF x Age
    rcf_rows = []
    for carb in dz.CARB_LEVELS:
        for ss in (0, 25, 50, 75):
            for age in (7, 28):
                p2 = pd.DataFrame({"RCF_pct": [50.0, 10.0], "SS_pct": ss, "AB": 0.45, "carbonation": carb, "age": age})
                Lr = mt.rows(final, p2)
                e = mt.lincomb(m, Lr[0] - Lr[1]).iloc[0]
                pr = mt.predict_points(m, final, p2)
                rcf_rows.append({"carbonation": carb, "SS": ss, "age": age, "ratio": np.exp(e.est),
                                 "lo": np.exp(e.lo), "hi": np.exp(e.hi), "p": e.p,
                                 "f_RCF50": pr["median"][0], "f_RCF10": pr["median"][1],
                                 "diff_MPa": pr["median"][0] - pr["median"][1]})
    R["rcf_effect"] = pd.DataFrame(rcf_rows)
    R["rcf_effect"].to_csv(os.path.join(TAB, "T20_RCF_effect_50_vs_10.csv"), index=False)

    # carbonation effect vs NC
    carb_rows = []
    for carb in dz.CARB_LEVELS[1:]:
        for rcf in (10, 30, 50):
            for ss in (0, 25, 50, 75):
                for age in (7, 28):
                    p2 = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ss, "AB": 0.45, "carbonation": [carb, "NC"], "age": age})
                    Lr = mt.rows(final, p2)
                    e = mt.lincomb(m, Lr[0] - Lr[1]).iloc[0]
                    pr = mt.predict_points(m, final, p2)
                    carb_rows.append({"carbonation": carb, "RCF": rcf, "SS": ss, "age": age, "ratio": np.exp(e.est),
                                      "lo": np.exp(e.lo), "hi": np.exp(e.hi), "p": e.p,
                                      "f_carb": pr["median"][0], "f_NC": pr["median"][1],
                                      "diff_MPa": pr["median"][0] - pr["median"][1]})
    R["carb_effect"] = pd.DataFrame(carb_rows)
    R["carb_effect"].to_csv(os.path.join(TAB, "T21_carbonation_effect_vs_NC.csv"), index=False)
    # carbonated-vs-NC crossover SS (where the ratio = 1), by level and RCF
    cross = []
    if "KD" in final:
        for carb in dz.CARB_LEVELS[1:]:
            for rcf in (10, 20, 30, 40, 50):
                p2 = pd.DataFrame({"RCF_pct": rcf, "SS_pct": np.linspace(0, 75, 3001), "AB": 0.45,
                                   "carbonation": carb, "age": 7})
                p0 = p2.copy(); p0["carbonation"] = "NC"
                diff = (mt.rows(final, p2) - mt.rows(final, p0)) @ m.beta
                sgn = np.where(np.diff(np.sign(diff)) != 0)[0]
                cross.append({"carbonation": carb, "RCF": rcf,
                              "crossover_SS": float(p2.SS_pct.values[sgn[0]]) if len(sgn) else None,
                              "ratio_SS0": float(np.exp(diff[0])), "ratio_SS75": float(np.exp(diff[-1]))})
    R["carb_crossover"] = pd.DataFrame(cross)

    # SS optimum (max strength) by age, carbonation, RCF + age change in MPa
    opt_ss = []
    ssf = np.linspace(0, 75, 3001)
    for age in (7, 28):
        for carb in dz.CARB_LEVELS:
            for rcf in (10, 30, 50):
                p2 = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ssf, "AB": 0.45, "carbonation": carb, "age": age})
                v = mt.rows(final, p2) @ m.beta
                j = int(np.argmax(v))
                opt_ss.append({"age": age, "carbonation": carb, "RCF": rcf, "SS_opt": float(ssf[j]),
                               "f_opt": float(np.exp(v[j])), "f_SS0": float(np.exp(v[0])),
                               "f_SS75": float(np.exp(v[-1])), "interior": 0 < j < len(ssf) - 1})
    R["ss_optimum"] = pd.DataFrame(opt_ss)
    R["contrasts"] = contr

    # overall optimum per age and carbonation level (RCF 10-50, SS 0-75, A/B not in model -> 0.45)
    log("optimisation")
    Rg = np.arange(10, 50.01, 0.5); Sg = np.arange(0, 75.01, 0.25)
    RR, SS = np.meshgrid(Rg, Sg)
    optimum = []
    design_pts = pd.DataFrame({"RCF_pct": d.RCF_pct, "SS_pct": d.SS_pct, "AB": d.AB, "carbonation": d.carbonation})
    spv_design = {a: mt.scaled_pred_var(m, final, design_pts.assign(age=a)).max() for a in (7, 28)}
    for age in (7, 28):
        for carb in dz.CARB_LEVELS:
            p2 = pd.DataFrame({"RCF_pct": RR.ravel(), "SS_pct": SS.ravel(), "AB": 0.45, "carbonation": carb, "age": age})
            v = mt.rows(final, p2) @ m.beta
            j = int(np.argmax(v))
            pbest = p2.iloc[[j]].reset_index(drop=True)
            pr = mt.predict_points(m, final, pbest).iloc[0]
            spv = float(mt.scaled_pred_var(m, final, pbest)[0])
            # region within 5 % of the optimum
            near = p2[np.exp(v) >= 0.95 * np.exp(v[j])]
            # nearest runs (coded distance)
            dist = np.sqrt(((d.A - (pbest.RCF_pct[0] - 30) / 20) ** 2 + (d.D - (pbest.SS_pct[0] - 37.5) / 37.5) ** 2).values)
            same = (d.carbonation == carb).values
            order = np.argsort(dist + 10 * (~same))[:3]
            optimum.append({"age": age, "carbonation": carb, "RCF": float(pbest.RCF_pct[0]), "SS": float(pbest.SS_pct[0]),
                            "median": pr["median"], "ci": [pr.ci_lo, pr.ci_hi], "pi": [pr.pi_lo, pr.pi_hi],
                            "spv": spv, "spv_design_max": float(spv_design[age]), "extrapolation": spv > spv_design[age],
                            "near95_RCF": [float(near.RCF_pct.min()), float(near.RCF_pct.max())],
                            "near95_SS": [float(near.SS_pct.min()), float(near.SS_pct.max())],
                            "nearest_runs": [{"mix": int(d.mix.values[k]), "RCF": float(d.RCF_pct.values[k]),
                                              "SS": float(d.SS_pct.values[k]), "carbonation": d.carbonation.values[k],
                                              "f": float(d[f"f{age}_mean"].values[k])} for k in order]})
    R["optimum"] = optimum
    pd.DataFrame([{k: (v if not isinstance(v, list) or k in ("ci", "pi") else str(v)) for k, v in o.items()}
                  for o in optimum]).to_csv(os.path.join(TAB, "T22_optimum_by_age_and_carbonation.csv"), index=False)
    # optimum contrast: best carbonated vs best NC at each age
    opt_contr = []
    for age in (7, 28):
        o = {x["carbonation"]: x for x in optimum if x["age"] == age}
        bestc = max(dz.CARB_LEVELS[1:], key=lambda c: o[c]["median"])
        p2 = pd.DataFrame({"RCF_pct": [o[bestc]["RCF"], o["NC"]["RCF"]], "SS_pct": [o[bestc]["SS"], o["NC"]["SS"]],
                           "AB": 0.45, "carbonation": [bestc, "NC"], "age": age})
        Lr = mt.rows(final, p2)
        e = mt.lincomb(m, Lr[0] - Lr[1]).iloc[0]
        opt_contr.append({"age": age, "best_carbonated": bestc, "ratio": np.exp(e.est), "lo": np.exp(e.lo),
                          "hi": np.exp(e.hi), "p": e.p})
    R["optimum_contrast"] = pd.DataFrame(opt_contr)

    # predictions for every mix (observed vs model), with CI/PI
    pm = []
    for age in (7, 28):
        pr = mt.predict_points(m, final, design_pts.assign(age=age))
        pm.append(pd.DataFrame({"mix": d.mix, "age": age, "observed": d[f"f{age}_mean"], **pr.to_dict(orient="list")}))
    pd.concat(pm).to_csv(os.path.join(TAB, "T23_predictions_all_mixes.csv"), index=False)

    # ------------------------------------------------------------------ robustness
    log("robustness analyses")
    ref_pts = pd.DataFrame([
        {"name": "NC, RCF 30, SS 50", "RCF_pct": 30, "SS_pct": 50, "AB": 0.45, "carbonation": "NC"},
        {"name": "1 h, RCF 50, SS 0", "RCF_pct": 50, "SS_pct": 0, "AB": 0.45, "carbonation": "1 h"},
        {"name": "1 h, RCF 50, SS 50", "RCF_pct": 50, "SS_pct": 50, "AB": 0.45, "carbonation": "1 h"},
        {"name": "5 h, RCF 10, SS 75", "RCF_pct": 10, "SS_pct": 75, "AB": 0.45, "carbonation": "5 h"}])

    def summarize(pred_fn, cvP=None, scale="ln", extra=None):
        """pred_fn(points_with_age) -> ln-scale predictions."""
        out = {}
        for ss in (0, 37.5, 75):
            p = pd.DataFrame({"RCF_pct": [30.0], "SS_pct": [ss], "AB": [0.45], "carbonation": ["NC"]})
            out[f"gain_SS{ss:g}"] = float(np.exp(pred_fn(p.assign(age=28))[0] - pred_fn(p.assign(age=7))[0]))
        for _, r in ref_pts.iterrows():
            p = pd.DataFrame([r.drop("name")])
            out[f"f7 [{r['name']}]"] = float(np.exp(pred_fn(p.assign(age=7))[0]))
            out[f"f28 [{r['name']}]"] = float(np.exp(pred_fn(p.assign(age=28))[0]))
        for carb_ss in ((0, 50), (75, 10)):
            ss, rcf = carb_ss
            p = pd.DataFrame({"RCF_pct": [rcf, rcf], "SS_pct": [ss, ss], "AB": 0.45, "carbonation": ["1 h", "NC"]})
            for age in (7, 28):
                v = pred_fn(p.assign(age=age))
                out[f"1h/NC ratio RCF{rcf} SS{ss} {age}d"] = float(np.exp(v[0] - v[1]))
        p = pd.DataFrame({"RCF_pct": [50, 10], "SS_pct": 0, "AB": 0.45, "carbonation": "1 h"})
        for age in (7, 28):
            v = pred_fn(p.assign(age=age))
            out[f"RCF50/10 ratio 1h SS0 {age}d"] = float(np.exp(v[0] - v[1]))
        if cvP is not None:
            out.update({k: v for k, v in cv_metrics(Y if scale == "ln" else np.exp(Y), cvP, scale).items()
                        if k.startswith("predR2") or k.startswith("rmse_MPa")})
        if extra:
            out.update(extra)
        return out

    def lmm_pred(model, terms, coding="cat4", transform="ln", lam=None):
        def f(p):
            dz.set_coding(coding)
            v = mt.rows(terms, p) @ model.beta
            dz.set_coding("cat4")
            if transform == "raw":
                return np.log(np.maximum(v, 1e-6))
            if transform == "bc":
                return np.log(np.maximum(lam * v + 1, 1e-9)) / lam
            return v
        return f

    rob = []
    P_final = val["mix_pred"]
    rob.append({"analysis": "Primary: final model, ln scale, UN, 4-level carbonation",
                **summarize(lmm_pred(m, final), P_final, extra={"AICc": aicc})})
    # CS covariance
    m_cs, _, _ = fit(final, d, struct="CS")
    P, _ = cv_predict(d, final, "CS")
    rob.append({"analysis": "Same terms, CS covariance (random intercept, equal variances)",
                **summarize(lmm_pred(m_cs, final), P, extra={"AICc": m_cs.aicc_ml()[0]})})
    # best CS model
    tcs = search["CS"]["top"][0]["terms"]
    m_cs2, _, _ = fit(tcs, d, struct="CS")
    P, _ = cv_predict(d, tcs, "CS")
    rob.append({"analysis": "AICc-best model under CS: " + " ".join(tcs),
                **summarize(lmm_pred(m_cs2, tcs), P, extra={"AICc": m_cs2.aicc_ml()[0]})})
    # on/off coding
    for cod, lab in (("onoff", "on/off (carbonated yes/no)"), ("log", "log2(1+h) duration"), ("numeric", "numeric duration")):
        dz.set_coding(cod)
        mc, _, _ = fit(final, d, struct=STRUCT)
        ac = mc.aicc_ml()[0]
        dz.set_coding("cat4")
        P, _ = cv_predict(d, final, STRUCT, coding=cod)
        rob.append({"analysis": f"Same terms, carbonation coded {lab}",
                    **summarize(lmm_pred(mc, final, cod), P, extra={"AICc": ac})})
    row_best_log = coding.set_index("coding").loc["log"]
    tlog = row_best_log.best_terms.split()
    dz.set_coding("log")
    mlog, _, _ = fit(tlog, d, struct=row_best_log.best_structure)
    dz.set_coding("cat4")
    P, _ = cv_predict(d, tlog, row_best_log.best_structure, coding="log")
    rob.append({"analysis": "AICc-best model with log-duration coding: " + " ".join(tlog),
                **summarize(lmm_pred(mlog, tlog, "log"), P, extra={"AICc": row_best_log.best_AICc})})
    # raw MPa scale
    Yraw = np.exp(Y)
    mraw, _, _ = fit(final, d, y=Yraw, struct=STRUCT)
    P, _ = cv_predict(d, final, STRUCT, y=Yraw)
    rob.append({"analysis": "Same terms, raw MPa scale", **summarize(lmm_pred(mraw, final, transform="raw"), P, scale="raw")})
    # Box-Cox lambda = -0.5 and 0.5
    for lam in (-0.5, 0.5):
        Yb = (Yraw ** lam - 1) / lam
        mb, _, _ = fit(final, d, y=Yb, struct=STRUCT)
        P, _ = cv_predict(d, final, STRUCT, y=Yb)
        Pln = np.log(np.maximum(lam * P + 1, 1e-9)) / lam
        rob.append({"analysis": f"Same terms, Box-Cox lambda = {lam:+.1f}",
                    **summarize(lmm_pred(mb, final, transform="bc", lam=lam), Pln)})
    # selection repeated on the lambda = -0.5 scale (edge of the full-model interval)
    Yb = (Yraw ** -0.5 - 1) / -0.5
    ab, tb, stb = select_overall(d, y=Yb)
    mb2, _, _ = fit(tb, d, y=Yb, struct=stb)
    P, _ = cv_predict(d, tb, stb, y=Yb)
    rob.append({"analysis": f"Selection repeated on lambda = -0.5 scale ({stb}): " + " ".join(tb),
                **summarize(lmm_pred(mb2, tb, transform="bc", lam=-0.5), np.log(np.maximum(-0.5 * P + 1, 1e-9)) / -0.5)})
    R["selection_lambda_m05"] = {"terms": tb, "struct": stb, "same_as_final": set(tb) == set(final)}
    # replicate-count weights: add specimen variance / n for each mean
    cv7, cv28 = R["data"]["specimen_scatter"]["7"]["pooled_cv"], R["data"]["specimen_scatter"]["28"]["pooled_cv"]
    nn = d[["f7_n", "f28_n"]].values.astype(float)
    ev = np.column_stack([cv7 ** 2 * (1 / nn[:, 0] - 1 / 3), cv28 ** 2 * (1 / nn[:, 1] - 1 / 3)])
    mw, _, _ = fit(final, d, struct=STRUCT, extra_var=ev)
    rob.append({"analysis": "Same terms, single-specimen means down-weighted (n = 1 vs 3)",
                **summarize(lmm_pred(mw, final), None)})
    # Huber-robust (whitened IRLS, Sigma fixed at REML estimate)
    Lc = np.linalg.cholesky(m.Sigma)
    yw = np.linalg.solve(Lc, Y.T).T.reshape(-1)
    Xw = np.einsum("ab,ibp->iap", np.linalg.inv(Lc), Xf).reshape(-1, Xf.shape[2])
    rl = sm.RLM(yw, Xw, M=sm.robust.norms.HuberT()).fit()
    mh = PairedLMM(Y, Xf, names, STRUCT, fit=False); mh.beta = rl.params; mh.cov_beta = rl.cov_params(); mh.Sigma = m.Sigma
    rob.append({"analysis": "Same terms, Huber robust M-estimation (whitened)",
                **summarize(lmm_pred(mh, final), None, extra={"min_weight": float(rl.weights.min()),
                                                              "downweighted_obs": int((rl.weights < 0.99).sum())})})
    R["huber_weights"] = rl.weights.reshape(-1, 2)
    # exclude most influential / outlying mixture(s)
    worst_t_mix = R["diagnostics"]["max_t_obs"]["mix"]
    worst_cook_mix = R["diagnostics"]["cook_max_mix"]
    for mx, why in sorted({(worst_t_mix, "largest |deleted t|"), (worst_cook_mix, "largest Cook's D")}):
        dd = d[d.mix != mx].reset_index(drop=True)
        md, _, _ = fit(final, dd, struct=STRUCT)
        a2, t2, st2 = select_overall(dd)
        Yd = dd[["l7", "l28"]].values
        P, _ = cv_predict(dd, final, STRUCT)
        met = cv_metrics(Yd, P)
        rob.append({"analysis": f"Same terms, mix {mx} excluded ({why}); re-selection gives "
                                + ("the same model" if set(t2) == set(final) and st2 == STRUCT else " ".join(t2) + f" [{st2}]"),
                    **summarize(lmm_pred(md, final), None, extra={k: v for k, v in met.items() if k.startswith("predR2") or k.startswith("rmse_MPa")})})
    # age-specific (fully interacted) final terms
    P, _ = cv_predict(d, full_int, STRUCT)
    rob.append({"analysis": "Final terms, every term interacting with age (age-specific coefficients)",
                **summarize(lmm_pred(mi, full_int), P, extra={"AICc": mi.aicc_ml()[0]})})
    # backward elimination model
    tbe = be[f"{STRUCT}_0.10"]["terms"]
    mbe, _, _ = fit(tbe, d, struct=STRUCT)
    P, _ = cv_predict(d, tbe, STRUCT)
    rob.append({"analysis": f"Backward elimination (alpha 0.10, {STRUCT}): " + " ".join(tbe),
                **summarize(lmm_pred(mbe, tbe), P, extra={"AICc": mbe.aicc_ml()[0]})})
    # separate per-age OLS models, each AICc-selected over hierarchical subsets
    per_age = {}
    subsets = sel.hierarchical_level_subsets()
    for j, a in enumerate(("7", "28")):
        best = None
        for s in subsets:
            Xs = sel.mix_X(s, cols, len(d)); k = Xs.shape[1] + 1
            if len(d) - k - 1 <= 0:
                continue
            ll = sel.ols_ll(Y[:, j], Xs); ac = -2 * ll + 2 * k + 2 * k * (k + 1) / (len(d) - k - 1)
            if best is None or ac < best[0]:
                best = (ac, s)
        per_age[a] = sorted(best[1], key=dz.LEVEL_TERMS.index)
    def per_age_pred(p):
        dfp = dz.new_points(p)
        out = np.zeros(len(p))
        for j, a in enumerate(("7", "28")):
            Xs = sel.mix_X(per_age[a], cols, len(d))
            b = np.linalg.lstsq(Xs, Y[:, j], rcond=None)[0]
            Xp = sel.mix_X(per_age[a], sel.mix_columns(dfp), len(p))
            sel_rows = np.asarray(p["age"]).astype(int) == int(a)
            out[sel_rows] = (Xp @ b)[sel_rows]
        return out
    Pp = np.zeros_like(Y)
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        c_tr = sel.mix_columns(d[tr].reset_index(drop=True)); c_te = sel.mix_columns(d[~tr].reset_index(drop=True))
        for j, a in enumerate(("7", "28")):
            b = np.linalg.lstsq(sel.mix_X(per_age[a], c_tr, 29), Y[tr, j], rcond=None)[0]
            Pp[i, j] = (sel.mix_X(per_age[a], c_te, 1) @ b)[0]
    rob.append({"analysis": f"Separate per-age OLS models (AICc): 7 d = {' '.join(per_age['7'])}; 28 d = {' '.join(per_age['28'])}",
                **summarize(per_age_pred, Pp)})
    R["per_age_models"] = per_age
    R["robustness"] = pd.DataFrame(rob)
    R["robustness"].to_csv(os.path.join(TAB, "T16_robustness.csv"), index=False)

    # ------------------------------------------------------------------ bootstrap
    n_sub = 24
    log(f"resampling: {n_boot} residual-bootstrap + {n_boot} subsample ({n_sub}/30) selections, {n_boot} stratified case-bootstrap refits")
    resid_mod = resid / np.sqrt(np.clip(1 - hat, 0.05, None))
    resid_mod = resid_mod - resid_mod.mean(0)
    ssf_b = np.linspace(0, 75, 301)
    opt_grids = {}
    for age in (7, 28):
        for carb in ("NC", "1 h"):
            for rcf in (10, 50):
                p2 = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ssf_b, "AB": 0.45, "carbonation": carb, "age": age})
                opt_grids[f"{age}d|{carb}|RCF{rcf}"] = mt.rows(final, p2)
    strata = [np.where(d.carbonation.values == lv)[0] for lv in dz.CARB_LEVELS]
    payload = {"d": d, "final": final, "fitted": fitted, "resid_mod": resid_mod, "n_sub": n_sub, "strata": strata,
               "gain_pts": gpts, "opt_grids": opt_grids, "opt_ss": ssf_b}
    seeds = np.random.default_rng(SEED).integers(0, 2 ** 31 - 1, size=3 * n_boot)
    modes = ("resid", "subsample", "case")
    jobs = [(int(seeds[k * n_boot + i]), md) for k, md in enumerate(modes) for i in range(n_boot)]
    with Pool(procs, initializer=_boot_init, initargs=(payload,)) as pool:
        out = pool.map(_boot_one, jobs, chunksize=20)
    boot = {}
    for mode in modes:
        res = [o for o, (s_, md) in zip(out, jobs) if md == mode and o is not None]
        B = np.array([r["beta"] for r in res])
        gains = np.exp(np.array([r["gain"] for r in res]))
        opt = {k: np.array([r["opt"][k] for r in res]) for k in opt_grids}
        boot[mode] = {"n": len(res),
                      "beta_ci": {nm: [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
                                  for j, nm in enumerate(names)},
                      "beta_sign": {nm: float(np.mean(np.sign(B[:, j]) == np.sign(m.beta[j]))) for j, nm in enumerate(names)},
                      "gain_ci": {"lo": np.percentile(gains, 2.5, axis=0).tolist(),
                                  "hi": np.percentile(gains, 97.5, axis=0).tolist()},
                      "opt_ss": {k: {"median": float(np.median(v)), "lo": float(np.percentile(v, 2.5)),
                                     "hi": float(np.percentile(v, 97.5))} for k, v in opt.items()}}
        if mode != "case":
            incl = Counter()
            for r in res:
                for t in r["terms"]:
                    incl[t] += 1
            boot[mode].update({
                "inclusion": {t: incl.get(t, 0) / len(res) for t in dz.ALL_TERMS},
                "exact_final": sum(set(r["terms"]) == set(final) and r["struct"] == STRUCT for r in res) / len(res),
                "same_terms": sum(set(r["terms"]) == set(final) for r in res) / len(res),
                "struct_freq": {k: v / len(res) for k, v in Counter(r["struct"] for r in res).items()},
                "gain_terms_SS_only": sum({t for t in r["terms"] if t.endswith(".Age")} <= {"D.Age", "D2.Age"} for r in res) / len(res),
                "top_models": Counter(" ".join(r["terms"]) + f" [{r['struct']}]" for r in res).most_common(8)})
    R["bootstrap"] = boot
    pd.DataFrame([{"term": t, "label": dz.label(t), "in_final": t in final,
                   "residual_bootstrap": boot["resid"]["inclusion"][t], "subsample_24of30": boot["subsample"]["inclusion"][t],
                   "akaike_importance_UN": search["UN"]["importance"][t]} for t in dz.ALL_TERMS]).to_csv(
        os.path.join(TAB, "T17_bootstrap_selection_frequency.csv"), index=False)
    ct = coef_table(m, names)
    ct["boot_resid_lo"] = [boot["resid"]["beta_ci"][nm][0] for nm in names]
    ct["boot_resid_hi"] = [boot["resid"]["beta_ci"][nm][1] for nm in names]
    ct["boot_case_lo"] = [boot["case"]["beta_ci"][nm][0] for nm in names]
    ct["boot_case_hi"] = [boot["case"]["beta_ci"][nm][1] for nm in names]
    ct["case_sign_agreement"] = [boot["case"]["beta_sign"][nm] for nm in names]
    ct.to_csv(os.path.join(TAB, "T18_final_coefficients_with_bootstrap_CI.csv"), index=False)
    gtab["boot_case_lo"] = boot["case"]["gain_ci"]["lo"]; gtab["boot_case_hi"] = boot["case"]["gain_ci"]["hi"]
    gtab.to_csv(os.path.join(TAB, "T19_gain_ratio_vs_SS.csv"), index=False)
    R["gain_curve"] = gtab

    # ------------------------------------------------------------------ design matrices for the report
    R["design_points"] = design_pts
    R["spv_design_max"] = spv_design
    R["runtime_s"] = time.time() - t_start
    with open(os.path.join(RES, "model_results.json"), "w") as f:
        json.dump(to_py(R), f, allow_nan=False)
    log(f"done in {R['runtime_s']:.0f} s -> results/model_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(quick=a.quick, n_boot=a.n_boot, procs=a.procs)
