"""
Two independent models: one for the 7-day strengths, one for the 28-day strengths.

Each age is analysed with the same protocol and only its own 30 results; no term,
coefficient, scale or covariance information passes from one age to the other.

Protocol (identical for both ages, fixed before either model was fitted):
 1. Scale       Box-Cox profile likelihood; ln if lambda = 0 is inside the 95 % CI of the
                selected model and lambda = 1 outside.
 2. Candidates  quadratic in RCF (A), A/B (C), SS (D); carbonation as a 4-level factor;
                carbonated contrast K x (A, C, D) for the carbonation slopes
                (16 parameters; 716 models that respect hierarchy).
 3. Coding      the 4-level coding is replaced only if another coding's best model has AICc
                lower by >= 2 AND a higher leave-one-out predicted R2.
 4. Selection   exhaustive AICc over all 716 hierarchical models (primary); BIC, PRESS and
                backward elimination (alpha 0.10 / 0.05) as cross-checks; nested
                cross-validation of each selection procedure.
 5. Validation  ANOVA, lack of fit against replicate batches, residual diagnostics, leave-one-
                mixture-out and leave-one-design-point-out prediction, single-mixture deletion,
                2,000 residual-bootstrap and 2,000 subsample (24 of 30) re-selections, 2,000
                case-bootstrap refits stratified by carbonation level.
 6. Robustness  other scales, codings, weights, a robust fit, influential mixtures excluded.
 7. Trends      the same contrasts computed from each model (RCF, carbonation, duration, SS,
                A/B, optimum), for comparison by the reader.

Run:  python analysis/run_separate.py [--n-boot 2000] [--procs 4]
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
import selection as sel

RES = os.path.join(dz.ROOT, "results", "separate")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)
SEED = 20260929
AGES = ("7", "28")
N = 30
LEVELS = dz.CARB_LEVELS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def to_py(o):
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


# ================================================================== model machinery
_SUBSETS = sel.hierarchical_level_subsets()


def order_terms(terms):
    return [t for t in dz.LEVEL_TERMS if t in set(terms)]


def X_of(terms, df, coding=None):
    if coding:
        dz.set_coding(coding)
    X, names = dz.mix_design(order_terms(terms), df)
    if coding:
        dz.set_coding("cat4")
    return X, names


def criteria(y, X):
    """AICc, BIC, PRESS and fit statistics of an OLS fit."""
    n, p = X.shape
    b, XtXi, r, rss, dfe = sel._ols(y, X)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    sst = np.sum((y - y.mean()) ** 2)
    press = np.sum((r / (1 - h)) ** 2) if h.max() < 0.9999 else np.inf
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    k = p + 1
    aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf
    return {"p": p, "rss": rss, "dfe": dfe, "R2": 1 - rss / sst, "adjR2": 1 - (rss / dfe) / (sst / (n - 1)) if dfe > 0 else np.nan,
            "predR2": 1 - press / sst, "press": press, "s": np.sqrt(rss / dfe) if dfe > 0 else np.nan,
            "AICc": aicc, "BIC": -2 * ll + k * np.log(n), "loglik": ll, "hmax": h.max()}


def search(y, df, coding="cat4", check_rank=True):
    """All hierarchical models, sorted by AICc."""
    dz.set_coding(coding)
    cols = sel.mix_columns(df)
    dz.set_coding("cat4")
    n = len(y)
    rows = []
    for s in _SUBSETS:
        X = sel.mix_X(s, cols, n)
        if n - X.shape[1] - 2 <= 0 or (check_rank and np.linalg.matrix_rank(X) < X.shape[1]):
            continue
        c = criteria(y, X)
        c["terms"] = tuple(order_terms(s))
        rows.append(c)
    R = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    return R


def best_by(R, crit):
    return list(R.sort_values(crit).iloc[0].terms)


def backward(y, df, alpha, coding="cat4"):
    dz.set_coding(coding)
    cols = sel.mix_columns(df)
    dz.set_coding("cat4")
    n = len(y)
    terms = list(dz.LEVEL_TERMS)
    hist = []
    while True:
        X = sel.mix_X(terms, cols, n)
        cand = [t for t in terms if not any(t in dz.PARENTS[u] for u in terms)]
        worst = None
        for t in cand:
            order = order_terms(terms)
            start = 1 + sum(cols[u].shape[1] for u in order[:order.index(t)])
            F, p = sel._ftest_drop(y, X, list(range(start, start + cols[t].shape[1])))
            p = 1.0 if not np.isfinite(p) else p
            if worst is None or p > worst[2]:
                worst = (t, F, p)
        if worst is None or worst[2] <= alpha:
            return order_terms(terms), hist
        hist.append({"removed": worst[0], "F": worst[1], "p": worst[2]})
        terms.remove(worst[0])


def ols(terms, df, y, weights=None, coding=None):
    X, names = X_of(terms, df, coding)
    m = (sm.WLS(y, X, weights=weights) if weights is not None else sm.OLS(y, X)).fit()
    m.names_ = names
    return m, X, names


def term_columns_idx(names, term):
    return [j for j, c in enumerate(names) if dz.term_of_column(c) == term]


def anova(m, names, terms):
    """Partial (Type III-like) F test for each term, plus model / residual / lack-of-fit lines."""
    rows = []
    for t in order_terms(terms):
        idx = term_columns_idx(names, t)
        L = np.zeros((len(idx), len(names)))
        for r, j in enumerate(idx):
            L[r, j] = 1
        ft = m.f_test(L)
        ss = float(ft.fvalue) * len(idx) * m.scale
        rows.append({"source": t, "label": dz.label(t), "df": len(idx), "SS": ss, "MS": ss / len(idx),
                     "F": float(ft.fvalue), "p": float(ft.pvalue)})
    return pd.DataFrame(rows)


def lack_of_fit(y, X, df):
    DP = pd.get_dummies(df.group).values.astype(float)
    _, _, _, rss, dfe = sel._ols(y, X)
    _, _, _, rss_s, df_s = sel._ols(y, DP)
    F = ((rss - rss_s) / (dfe - df_s)) / (rss_s / df_s)
    return {"SS_lof": rss - rss_s, "df_lof": dfe - df_s, "SS_pe": rss_s, "df_pe": df_s, "F": F,
            "p": stats.f.sf(F, dfe - df_s, df_s), "sd_resid": np.sqrt(rss / dfe), "sd_pure": np.sqrt(rss_s / df_s)}


def vif(X):
    Xc = X[:, 1:]
    out = []
    for j in range(Xc.shape[1]):
        others = np.delete(Xc, j, axis=1)
        A = np.column_stack([np.ones(len(Xc)), others])
        b = np.linalg.lstsq(A, Xc[:, j], rcond=None)[0]
        r = Xc[:, j] - A @ b
        r2 = 1 - r @ r / np.sum((Xc[:, j] - Xc[:, j].mean()) ** 2)
        out.append(1 / max(1 - r2, 1e-12))
    return [np.nan] + out


def boxcox(df, age, terms, lams=np.round(np.arange(-1.5, 1.501, 0.01), 2)):
    yr = df[f"f{age}_mean"].values
    X, _ = X_of(terms, df)
    L = []
    for lam in lams:
        z = np.log(yr) if abs(lam) < 1e-9 else (yr ** lam - 1) / lam
        L.append(sel.ols_ll(z, X) + (lam - 1) * np.log(yr).sum())
    L = np.array(L)
    ci = lams[L >= L.max() - stats.chi2.ppf(0.95, 1) / 2]
    i0, i1 = np.argmin(np.abs(lams)), np.argmin(np.abs(lams - 1))
    return {"lambda": float(lams[L.argmax()]), "ci": [float(ci.min()), float(ci.max())],
            "LR_lambda1": float(2 * (L.max() - L[i1])), "LR_lambda0": float(2 * (L.max() - L[i0])),
            "curve": {"lambda": lams.tolist(), "loglik": (L - L.max()).tolist()},
            "cut": float(-stats.chi2.ppf(0.95, 1) / 2)}


def fit_summary(y, df):
    """Sequential model-order table (Design-Expert style) on the chosen scale."""
    orders = [("Mean", []), ("Linear", ["A", "C", "D", "Carb"]),
              ("2FI", ["A", "C", "D", "Carb", "AC", "AD", "CD", "KA", "KC", "KD"]),
              ("Quadratic", list(dz.LEVEL_TERMS))]
    rows, prev = [], None
    for nm, terms in orders:
        X, _ = X_of(terms, df)
        c = criteria(y, X)
        lof = lack_of_fit(y, X, df)
        row = {"model": nm, "p": c["p"], "R2": c["R2"], "adjR2": c["adjR2"], "predR2": c["predR2"], "s": c["s"],
               "lof_F": lof["F"], "lof_p": lof["p"], "seq_F": np.nan, "seq_p": np.nan}
        if prev is not None:
            dfn = c["p"] - prev["p"]
            F = ((prev["rss"] - c["rss"]) / dfn) / (c["rss"] / c["dfe"])
            row["seq_F"], row["seq_p"] = F, stats.f.sf(F, dfn, c["dfe"])
        rows.append(row)
        prev = c
    return pd.DataFrame(rows)


def cv_predict(terms, df, y, groups="mix", select=None, coding="cat4"):
    keys = df.mix.values if groups == "mix" else df.group.values
    P = np.zeros(len(y))
    chosen = []
    for k in np.unique(keys):
        te = keys == k
        tr = ~te
        dtr, dte = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
        t_use = terms if select is None else select(y[tr], dtr)
        chosen.append(tuple(order_terms(t_use)))
        Xtr, _ = X_of(t_use, dtr, coding)
        Xte, _ = X_of(t_use, dte, coding)
        b = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
        P[te] = Xte @ b
    return P, chosen


def cv_metrics(y, P, ln=True):
    e = y - P
    em = (np.exp(y) - np.exp(P)) if ln else e
    return {"predR2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse_ln": float(np.sqrt(np.mean(e ** 2))), "rmse_MPa": float(np.sqrt(np.mean(em ** 2))),
            "mae_MPa": float(np.mean(np.abs(em)))}


# ------------------------------------------------------------------ prediction helpers
def pts_design(terms, pts, coding=None):
    df = dz.new_points(pts)
    X, names = X_of(terms, df, coding)
    return X


def predict(m, terms, pts, level=0.95):
    X = pts_design(terms, pts)
    fit = X @ m.params
    se = np.sqrt(np.einsum("ip,pq,iq->i", X, m.cov_params(), X))
    q = stats.t.ppf(0.5 + level / 2, m.df_resid)
    sp = np.sqrt(se ** 2 + m.scale)
    return pd.DataFrame({"ln_fit": fit, "se": se, "median": np.exp(fit), "ci_lo": np.exp(fit - q * se),
                         "ci_hi": np.exp(fit + q * se), "pi_lo": np.exp(fit - q * sp), "pi_hi": np.exp(fit + q * sp)})


def contrast(m, terms, p1, p2):
    """ratio f(p1)/f(p2) with 95 % CI and p (t with residual df)."""
    l = pts_design(terms, pd.DataFrame([p1]))[0] - pts_design(terms, pd.DataFrame([p2]))[0]
    est = float(l @ m.params)
    se = float(np.sqrt(l @ m.cov_params() @ l))
    if se < 1e-12:
        return {"ratio": float(np.exp(est)), "lo": float(np.exp(est)), "hi": float(np.exp(est)), "p": np.nan, "in_model": False}
    q = stats.t.ppf(0.975, m.df_resid)
    return {"ratio": float(np.exp(est)), "lo": float(np.exp(est - q * se)), "hi": float(np.exp(est + q * se)),
            "p": float(2 * stats.t.sf(abs(est / se), m.df_resid)), "in_model": True}


ACTUAL_COLS = ["1", "RCF", "SS", "A/B", "RCF·SS", "RCF·A/B", "SS·A/B", "RCF²", "SS²", "(A/B)²",
               "C[0.5 h]", "C[1 h]", "C[5 h]", "K·RCF", "K·SS", "K·A/B"]


def actual_equation(m, terms):
    g = []
    for c in LEVELS:
        for R in (10, 23, 37, 50):
            for S in (0, 20, 45, 75):
                for B in (0.42, 0.45, 0.48):
                    g.append((R, S, B, c))
    pts = pd.DataFrame(g, columns=["RCF_pct", "SS_pct", "AB", "carbonation"])
    Rr, S, B = pts.RCF_pct.values, pts.SS_pct.values, pts.AB.values
    cc = pts.carbonation.values
    K = (cc != "NC").astype(float)
    Xa = np.column_stack([np.ones(len(pts)), Rr, S, B, Rr * S, Rr * B, S * B, Rr ** 2, S ** 2, B ** 2,
                          (cc == "0.5 h") * 1.0, (cc == "1 h") * 1.0, (cc == "5 h") * 1.0, K * Rr, K * S, K * B])
    Xc = pts_design(terms, pts)
    T = np.linalg.lstsq(Xa, Xc, rcond=None)[0]
    coef = T @ m.params
    cov = T @ m.cov_params() @ T.T
    assert np.allclose(Xa @ coef, Xc @ m.params, atol=1e-8)
    keep = np.abs(coef) > 1e-12 * max(1, np.abs(coef).max())
    return [{"term": ACTUAL_COLS[j], "coef": float(coef[j]), "se": float(np.sqrt(max(cov[j, j], 0)))}
            for j in range(len(coef)) if keep[j]]


# ================================================================== resampling worker
_W = {}


def _init(payload):
    _W.update(payload)


def _one(args):
    seed, age, mode = args
    rng = np.random.default_rng(seed)
    d = _W["d"]
    P = _W[age]
    final = P["final"]
    n = len(d)
    for _ in range(50):
        if mode == "resid":
            db = d
            y = P["fitted"] + P["resid_mod"][rng.integers(0, n, n)]
        elif mode == "subsample":
            idx = np.sort(rng.choice(n, size=24, replace=False))
            db = d.iloc[idx].reset_index(drop=True)
            y = P["y"][idx]
        else:
            idx = np.concatenate([rng.choice(ii, size=len(ii), replace=True) for ii in _W["strata"]])
            db = d.iloc[idx].reset_index(drop=True)
            y = P["y"][idx]
        Xf, _ = X_of(final, db)
        if np.linalg.matrix_rank(Xf) < Xf.shape[1]:
            continue
        terms = None
        if mode in ("resid", "subsample"):
            R = search(y, db)
            terms = list(R.iloc[0].terms)
        beta = np.linalg.lstsq(Xf, y, rcond=None)[0]
        opt = {}
        for key, G in P["opt_grids"].items():
            opt[key] = float(_W["opt_ss"][int(np.argmax(G @ beta))])
        rcf = {}
        for key, G in P["rcf_grids"].items():
            v = G @ beta
            rcf[key] = float(_W["opt_rcf"][int(np.argmax(v))])
        return {"terms": terms, "beta": beta.tolist(), "opt": opt, "rcf_opt": rcf}
    return None


# ================================================================== per-age analysis
def analyze(age, d, n_boot, procs):
    log(f"--- {age}-day model")
    out = {"age": age}
    yr = d[f"f{age}_mean"].values
    y = np.log(yr)
    out["describe"] = {"min": yr.min(), "max": yr.max(), "mean": yr.mean(), "sd": yr.std(ddof=1)}
    g = d.dropna(subset=["replicate_group"])
    ss = sum(((np.log(gg[f"f{age}_mean"]) - np.log(gg[f"f{age}_mean"]).mean()) ** 2).sum() for _, gg in g.groupby("replicate_group"))
    ssm = sum(((gg[f"f{age}_mean"] - gg[f"f{age}_mean"].mean()) ** 2).sum() for _, gg in g.groupby("replicate_group"))
    out["pure_error"] = {"sd_ln": np.sqrt(ss / 6), "sd_MPa": np.sqrt(ssm / 6), "df": 6}

    # scale on the full candidate model
    out["boxcox_full"] = boxcox(d, age, dz.LEVEL_TERMS)
    out["fit_summary"] = fit_summary(y, d)

    # coding comparison
    codes = []
    for cod in ("cat4", "onoff", "numeric", "log"):
        R = search(y, d, cod)
        best = list(R.iloc[0].terms)
        P, _ = cv_predict(best, d, y, coding=cod)
        codes.append({"coding": cod, "best_terms": " ".join(best), "AICc": R.iloc[0].AICc, "k": R.iloc[0].p,
                      "R2": R.iloc[0].R2, "adjR2": R.iloc[0].adjR2, "predR2_LOO": cv_metrics(y, P)["predR2"]})
    codes = pd.DataFrame(codes)
    ref = codes.set_index("coding").loc["cat4"]
    codes["dAICc"] = codes.AICc - ref.AICc
    codes["switch_rule_met"] = [(c != "cat4") and (a <= ref.AICc - 2) and (p > ref.predR2_LOO)
                                for c, a, p in zip(codes.coding, codes.AICc, codes.predR2_LOO)]
    out["coding"] = codes
    coding = "cat4"
    if codes.switch_rule_met.any():
        log(f"  NOTE: coding switch rule met at {age} d; primary coding kept, see coding table")

    # exhaustive search
    R = search(y, d)
    A_ = R.AICc.values
    w = np.exp(-(A_ - A_.min()) / 2); w /= w.sum()
    imp = Counter()
    for t_, wi in zip(R.terms, w):
        for t in t_:
            imp[t] += wi
    R["weight"] = w
    R["dAICc"] = R.AICc - A_.min()
    out["search_top"] = R.head(15).assign(terms=lambda x: x.terms.map(" ".join))[
        ["terms", "p", "R2", "adjR2", "predR2", "s", "AICc", "dAICc", "weight", "BIC"]]
    out["n_models"] = len(R)
    out["importance"] = {t: float(imp.get(t, 0.0)) for t in dz.LEVEL_TERMS}
    final = list(R.iloc[0].terms)
    out["final_terms"] = final
    out["cross_checks"] = {"BIC": best_by(R, "BIC"), "PRESS": best_by(R, "press")}
    for a in (0.10, 0.05):
        t_, h_ = backward(y, d, a)
        out["cross_checks"][f"backward_{a:.2f}"] = t_
        out[f"backward_history_{a:.2f}"] = h_
    log(f"  final: {final}")

    # scale on the selected model
    out["boxcox_final"] = boxcox(d, age, final)

    # final fit
    m, X, names = ols(final, d, y)
    infl = m.get_influence()
    h = infl.hat_matrix_diag
    tq = stats.t.ppf(0.975, m.df_resid)
    coef = pd.DataFrame({"column": names, "term": [dz.term_of_column(c) for c in names], "coef": m.params, "se": m.bse,
                         "t": m.tvalues, "p": m.pvalues, "lo": m.params - tq * m.bse, "hi": m.params + tq * m.bse,
                         "VIF": vif(X)})
    out["coef"] = coef
    out["anova"] = anova(m, names, final)
    c = criteria(y, X)
    lof = lack_of_fit(y, X, d)
    fitted = X @ m.params
    se_fit = np.sqrt(np.einsum("ip,pq,iq->i", X, m.cov_params(), X))
    out["fit"] = {"R2": c["R2"], "adjR2": c["adjR2"], "predR2": c["predR2"], "s_ln": c["s"],
                  "cv_pct": 100 * np.sqrt(np.exp(c["s"] ** 2) - 1), "AICc": c["AICc"], "BIC": c["BIC"],
                  "adeq_precision": (fitted.max() - fitted.min()) / np.sqrt(np.mean(se_fit ** 2)),
                  "F_model": float(m.fvalue), "p_model": float(m.f_pvalue), "df_model": int(m.df_model), "df_resid": int(m.df_resid),
                  "rmse_fit_MPa": float(np.sqrt(np.mean((yr - np.exp(fitted)) ** 2))), "lof": lof}
    out["equation_actual"] = actual_equation(m, final)
    if "Carb" in final:
        idx = [names.index(f"Carb[{l}]") for l in LEVELS[1:]]
        L = np.zeros((2, len(names))); L[0, idx[1]] = 1; L[0, idx[0]] = -1; L[1, idx[2]] = 1; L[1, idx[0]] = -1
        ft = m.f_test(L)
        out["duration_equal_test"] = {"F": float(ft.fvalue), "df1": 2, "df2": int(m.df_resid), "p": float(ft.pvalue)}

    # diagnostics
    tstud = infl.resid_studentized_external
    cook = infl.cooks_distance[0]
    bonf = np.minimum(1, 2 * stats.t.sf(np.abs(tstud), m.df_resid - 1) * N)
    sw = stats.shapiro(m.resid)
    bp = sm.stats.het_breuschpagan(m.resid, X)
    rho = stats.spearmanr(d.mix, np.abs(m.resid))
    dw = float(np.sum(np.diff(m.resid) ** 2) / np.sum(m.resid ** 2))
    Ploo, _ = cv_predict(final, d, y, "mix")
    Pgrp, _ = cv_predict(final, d, y, "group")
    out["diagnostics"] = {"table": pd.DataFrame({"mix": d.mix, "carbonation": d.carbonation, "RCF": d.RCF_pct, "SS": d.SS_pct,
                                                 "AB": d.AB, "observed": yr, "fitted": np.exp(fitted), "loo": np.exp(Ploo),
                                                 "lodpo": np.exp(Pgrp), "resid_ln": m.resid, "t_ext": tstud, "p_bonf": bonf,
                                                 "leverage": h, "cook": cook}),
                          "shapiro_W": sw.statistic, "shapiro_p": sw.pvalue, "bp_LM": bp[0], "bp_p": bp[1],
                          "run_rho": rho.statistic, "run_p": rho.pvalue, "durbin_watson": dw,
                          "max_t": float(np.abs(tstud).max()), "max_t_mix": int(d.mix[np.argmax(np.abs(tstud))]),
                          "min_bonf": float(bonf.min()), "cook_max": float(cook.max()), "cook_mix": int(d.mix[np.argmax(cook)]),
                          "lev_max": float(h.max()), "lev_mix": int(d.mix[np.argmax(h)])}
    out["validation"] = {"fit": cv_metrics(y, fitted), "loo": cv_metrics(y, Ploo), "lodpo": cv_metrics(y, Pgrp)}

    # nested CV of selection procedures
    procs_sel = {"AICc (primary)": lambda yy, dd: list(search(yy, dd).iloc[0].terms),
                 "BIC": lambda yy, dd: best_by(search(yy, dd), "BIC"),
                 "PRESS": lambda yy, dd: best_by(search(yy, dd), "press"),
                 "Backward a=0.10": lambda yy, dd: backward(yy, dd, 0.10)[0],
                 "Backward a=0.05": lambda yy, dd: backward(yy, dd, 0.05)[0]}
    ncv = []
    for gname in ("mix", "group"):
        for nm, f in procs_sel.items():
            P, ch = cv_predict(None, d, y, gname, select=f)
            cn = Counter(ch)
            ncv.append({"cv": "leave-one-mixture-out" if gname == "mix" else "leave-one-design-point-out", "procedure": nm,
                        **cv_metrics(y, P), "distinct": len(cn), "folds": len(ch),
                        "final_selected": sum(1 for t in ch if set(t) == set(final))})
        for nm, t_ in (("Full candidate model", dz.LEVEL_TERMS), ("Final model (fixed terms)", final)):
            P, _ = cv_predict(t_, d, y, gname)
            ncv.append({"cv": "leave-one-mixture-out" if gname == "mix" else "leave-one-design-point-out", "procedure": nm,
                        **cv_metrics(y, P), "distinct": np.nan, "folds": np.nan, "final_selected": np.nan})
    out["nested_cv"] = pd.DataFrame(ncv)

    # single-mixture deletion
    dels = []
    for i in range(N):
        dd = d.drop(index=i).reset_index(drop=True)
        t_ = list(search(np.delete(y, i), dd).iloc[0].terms)
        dels.append({"mix_removed": int(d.mix[i]), "terms": " ".join(t_), "same": set(t_) == set(final)})
    out["deletion"] = pd.DataFrame(dels)

    # added-term tests (every omitted term with its hierarchy parents)
    added = []
    for t in dz.LEVEL_TERMS:
        if t in final:
            continue
        add = [t]
        ch_ = True
        while ch_:
            ch_ = False
            for u in list(add):
                for p_ in dz.PARENTS[u]:
                    if p_ not in final and p_ not in add:
                        add.append(p_); ch_ = True
        m2, X2, n2 = ols(final + add, d, y)
        idx = [j for j, cname in enumerate(n2) if dz.term_of_column(cname) in add]
        L = np.zeros((len(idx), len(n2)))
        for r_, j in enumerate(idx):
            L[r_, j] = 1
        ft = m2.f_test(L)
        est = {n2[j]: (float(m2.params[j]), float(m2.bse[j])) for j in idx}
        added.append({"term": t, "added": " + ".join(order_terms(add)), "df1": len(idx), "df2": int(m2.df_resid),
                      "F": float(ft.fvalue), "p": float(ft.pvalue), "dAICc": criteria(y, X2)["AICc"] - c["AICc"],
                      "estimates": est})
    out["added_terms"] = pd.DataFrame(added)

    # prediction grids / contrasts
    ssf = np.linspace(0, 75, 301)
    rcff = np.linspace(10, 50, 161)
    opt_grids, rcf_grids = {}, {}
    for carb in LEVELS:
        for rcf in (10, 30, 50):
            opt_grids[f"{carb}|RCF{rcf}"] = pts_design(final, pd.DataFrame({"RCF_pct": rcf, "SS_pct": ssf, "AB": 0.45, "carbonation": carb}))
        for ssv in (0, 37.5, 75):
            rcf_grids[f"{carb}|SS{ssv:g}"] = pts_design(final, pd.DataFrame({"RCF_pct": rcff, "SS_pct": ssv, "AB": 0.45, "carbonation": carb}))
    resid_mod = m.resid / np.sqrt(np.clip(1 - h, 0.05, None))
    resid_mod = resid_mod - resid_mod.mean()
    worker_payload = {"final": final, "fitted": fitted, "resid_mod": np.asarray(resid_mod), "y": y,
                      "opt_grids": opt_grids, "rcf_grids": rcf_grids}

    # trend quantities (all from this age's own model)
    ref = {"AB": 0.45}
    tr = {}
    rows = []
    for carb in LEVELS:
        for ssv in (0, 25, 50, 75):
            c_ = contrast(m, final, {"RCF_pct": 50, "SS_pct": ssv, "carbonation": carb, **ref},
                          {"RCF_pct": 10, "SS_pct": ssv, "carbonation": carb, **ref})
            rows.append({"carbonation": carb, "SS": ssv, **c_})
    tr["rcf_50_vs_10"] = pd.DataFrame(rows)
    rows = []
    for carb in LEVELS[1:]:
        for rcf in (10, 30, 50):
            for ssv in (0, 25, 50, 75):
                c_ = contrast(m, final, {"RCF_pct": rcf, "SS_pct": ssv, "carbonation": carb, **ref},
                              {"RCF_pct": rcf, "SS_pct": ssv, "carbonation": "NC", **ref})
                rows.append({"carbonation": carb, "RCF": rcf, "SS": ssv, **c_})
    tr["carb_vs_NC"] = pd.DataFrame(rows)
    rows = []
    for a_, b_ in (("1 h", "0.5 h"), ("5 h", "0.5 h"), ("5 h", "1 h")):
        c_ = contrast(m, final, {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": a_, **ref},
                      {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": b_, **ref})
        rows.append({"contrast": f"{a_} / {b_}", **c_})
    tr["duration"] = pd.DataFrame(rows)
    rows = []
    for carb in LEVELS:
        for rcf in (10, 30, 50):
            G = opt_grids[f"{carb}|RCF{rcf}"]
            v = G @ m.params
            j = int(np.argmax(v))
            j2 = int(np.argmin(np.abs(ssf - 200 / 3)))
            rows.append({"carbonation": carb, "RCF": rcf, "SS_at_max": float(ssf[j]), "f_max": float(np.exp(v[j])),
                         "f_SS0": float(np.exp(v[0])), "f_SS75": float(np.exp(v[-1])), "f_SSSH2": float(np.exp(v[j2])),
                         "ratio_SS0_to_max": float(np.exp(v[0] - v[j])), "ratio_SSSH2_to_max": float(np.exp(v[j2] - v[j])),
                         "interior": bool(0 < j < len(ssf) - 1)})
    tr["ss_profile"] = pd.DataFrame(rows)
    rows = []
    for carb in LEVELS:
        for ssv in (0, 37.5, 75):
            G = rcf_grids[f"{carb}|SS{ssv:g}"]
            v = G @ m.params
            j = int(np.argmax(v))
            rows.append({"carbonation": carb, "SS": ssv, "RCF_at_max": float(rcff[j]), "f_max": float(np.exp(v[j])),
                         "f_RCF10": float(np.exp(v[0])), "f_RCF50": float(np.exp(v[-1])), "interior": bool(0 < j < len(rcff) - 1)})
    tr["rcf_profile"] = pd.DataFrame(rows)
    c_ = contrast(m, final, {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": "NC", "AB": 0.42},
                  {"RCF_pct": 30, "SS_pct": 37.5, "carbonation": "NC", "AB": 0.48})
    tr["ab_042_vs_048"] = c_
    # SS/SH = 2 (SS = 66.7 %) vs the SS maximum, with CI
    rows = []
    for carb in ("NC", "1 h"):
        for rcf in (10, 30, 50):
            r_ = tr["ss_profile"][(tr["ss_profile"].carbonation == carb) & (tr["ss_profile"].RCF == rcf)].iloc[0]
            c2 = contrast(m, final, {"RCF_pct": rcf, "SS_pct": 200 / 3, "carbonation": carb, **ref},
                          {"RCF_pct": rcf, "SS_pct": r_.SS_at_max, "carbonation": carb, **ref})
            c0 = contrast(m, final, {"RCF_pct": rcf, "SS_pct": 0, "carbonation": carb, **ref},
                          {"RCF_pct": rcf, "SS_pct": r_.SS_at_max, "carbonation": carb, **ref})
            rows.append({"carbonation": carb, "RCF": rcf, "SS_at_max": r_.SS_at_max, "SSSH2_vs_max": c2["ratio"], "SSSH2_lo": c2["lo"],
                         "SSSH2_hi": c2["hi"], "SSSH2_p": c2["p"], "SS0_vs_max": c0["ratio"], "SS0_lo": c0["lo"], "SS0_hi": c0["hi"],
                         "SS0_p": c0["p"]})
    tr["ss_contrasts"] = pd.DataFrame(rows)
    out["trends"] = tr

    # optimum per carbonation level (A/B at the level that maximises, if A/B is in the model)
    Rg, Sg = np.arange(10, 50.01, 0.5), np.arange(0, 75.01, 0.25)
    RR, SSg = np.meshgrid(Rg, Sg)
    abs_ = (0.42, 0.45, 0.48) if "C" in final else (0.45,)
    Xdes = X
    lev_design = np.einsum("ip,pq,iq->i", Xdes, np.linalg.inv(Xdes.T @ Xdes), Xdes)
    opts = []
    for carb in LEVELS:
        best = None
        for ab in abs_:
            pts = pd.DataFrame({"RCF_pct": RR.ravel(), "SS_pct": SSg.ravel(), "AB": ab, "carbonation": carb})
            v = pts_design(final, pts) @ m.params
            j = int(np.argmax(v))
            if best is None or v[j] > best[0]:
                best = (v[j], pts.iloc[[j]].reset_index(drop=True), v, pts)
        pb = best[1]
        pr = predict(m, final, pb).iloc[0]
        xb = pts_design(final, pb)[0]
        lev = float(xb @ np.linalg.inv(Xdes.T @ Xdes) @ xb)
        near = best[3][np.exp(best[2]) >= 0.95 * np.exp(best[0])]
        opts.append({"carbonation": carb, "RCF": float(pb.RCF_pct[0]), "SS": float(pb.SS_pct[0]), "AB": float(pb.AB[0]),
                     "median": pr["median"], "ci": [pr.ci_lo, pr.ci_hi], "pi": [pr.pi_lo, pr.pi_hi], "leverage": lev,
                     "leverage_design_max": float(lev_design.max()), "extrapolation": lev > lev_design.max(),
                     "near95_RCF": [float(near.RCF_pct.min()), float(near.RCF_pct.max())],
                     "near95_SS": [float(near.SS_pct.min()), float(near.SS_pct.max())]})
    out["optimum"] = opts

    # robustness
    rob = []

    def summ(pred_ln, extra=None):
        """pred_ln(points) -> ln predictions"""
        s = {}
        for nm_, p1, p2 in (("RCF50/10, 1 h, SS 0", dict(RCF_pct=50, SS_pct=0, carbonation="1 h"), dict(RCF_pct=10, SS_pct=0, carbonation="1 h")),
                            ("RCF50/10, NC, SS 75", dict(RCF_pct=50, SS_pct=75, carbonation="NC"), dict(RCF_pct=10, SS_pct=75, carbonation="NC")),
                            ("1 h/NC, RCF 50, SS 0", dict(RCF_pct=50, SS_pct=0, carbonation="1 h"), dict(RCF_pct=50, SS_pct=0, carbonation="NC")),
                            ("1 h/NC, RCF 10, SS 75", dict(RCF_pct=10, SS_pct=75, carbonation="1 h"), dict(RCF_pct=10, SS_pct=75, carbonation="NC")),
                            ("SS 50/0, NC, RCF 30", dict(RCF_pct=30, SS_pct=50, carbonation="NC"), dict(RCF_pct=30, SS_pct=0, carbonation="NC")),
                            ("SS 75/50, NC, RCF 30", dict(RCF_pct=30, SS_pct=75, carbonation="NC"), dict(RCF_pct=30, SS_pct=50, carbonation="NC")),
                            ("A/B 0.42/0.48", dict(RCF_pct=30, SS_pct=37.5, carbonation="NC", AB=0.42), dict(RCF_pct=30, SS_pct=37.5, carbonation="NC", AB=0.48))):
            p1 = {"AB": 0.45, **p1}; p2 = {"AB": 0.45, **p2}
            v = pred_ln(pd.DataFrame([p1, p2]))
            s[nm_] = float(np.exp(v[0] - v[1]))
        if extra:
            s.update(extra)
        return s

    def lin(model_params, terms, coding=None, transform=None, lam=None):
        def f(p):
            v = pts_design(terms, p, coding) @ model_params
            if transform == "raw":
                return np.log(np.maximum(v, 1e-6))
            if transform == "bc":
                return np.log(np.maximum(lam * v + 1, 1e-9)) / lam
            return v
        return f

    rob.append({"analysis": "PRIMARY: final model (ln, 4-level carbonation, OLS)", "terms": " ".join(final),
                **summ(lin(m.params, final), {"predR2_LOO": out["validation"]["loo"]["predR2"], "R2": c["R2"]})})
    for cod, lab in (("onoff", "carbonation on/off"), ("log", "carbonation log-duration"), ("numeric", "carbonation numeric duration")):
        row = codes.set_index("coding").loc[cod]
        t_ = row.best_terms.split()
        mc, Xc_, _ = ols(t_, d, y, coding=cod)
        rob.append({"analysis": f"AICc-best model, {lab}", "terms": " ".join(t_),
                    **summ(lin(mc.params, t_, cod), {"predR2_LOO": row.predR2_LOO, "R2": mc.rsquared})})
    for crit in ("BIC", "PRESS", "backward_0.10"):
        t_ = out["cross_checks"][crit]
        if set(t_) == set(final):
            continue
        mc, _, _ = ols(t_, d, y)
        P, _ = cv_predict(t_, d, y)
        rob.append({"analysis": f"{crit.replace('_', ' ')}-selected model", "terms": " ".join(t_),
                    **summ(lin(mc.params, t_), {"predR2_LOO": cv_metrics(y, P)["predR2"], "R2": mc.rsquared})})
    # raw scale, other lambdas
    mr, _, _ = ols(final, d, yr)
    P, _ = cv_predict(final, d, yr)
    rob.append({"analysis": "Same terms, raw MPa scale", "terms": " ".join(final),
                **summ(lin(mr.params, final, transform="raw"), {"predR2_LOO": 1 - np.sum((yr - P) ** 2) / np.sum((yr - yr.mean()) ** 2), "R2": mr.rsquared})})
    for lam in (-0.5, 0.5):
        yb = (yr ** lam - 1) / lam
        mb, _, _ = ols(final, d, yb)
        rob.append({"analysis": f"Same terms, Box-Cox lambda = {lam:+.1f}", "terms": " ".join(final),
                    **summ(lin(mb.params, final, transform="bc", lam=lam), {"R2": mb.rsquared})})
    # re-selection on raw scale
    Rraw = search(yr, d)
    t_ = list(Rraw.iloc[0].terms)
    mr2, _, _ = ols(t_, d, yr)
    rob.append({"analysis": "Selection repeated on the raw MPa scale", "terms": " ".join(t_),
                **summ(lin(mr2.params, t_, transform="raw"), {"R2": mr2.rsquared})})
    # weights for n = 1 means
    cvw = np.sqrt(np.nansum((d[f"f{age}_n"] - 1) * (d[f"f{age}_sd"] / d[f"f{age}_mean"]) ** 2) / np.nansum((d[f"f{age}_n"] - 1)[d[f"f{age}_sd"].notna()]))
    s2b = max(out["pure_error"]["sd_ln"] ** 2 - cvw ** 2 / 3, 1e-4)
    wts = 1 / (s2b + cvw ** 2 / d[f"f{age}_n"].values)
    mw, _, _ = ols(final, d, y, weights=wts)
    rob.append({"analysis": "Same terms, single-specimen means down-weighted", "terms": " ".join(final),
                **summ(lin(mw.params, final), {"R2": mw.rsquared})})
    # Huber
    Xf_, _ = X_of(final, d)
    rl = sm.RLM(y, Xf_, M=sm.robust.norms.HuberT()).fit()
    rob.append({"analysis": "Same terms, Huber robust regression", "terms": " ".join(final),
                **summ(lin(rl.params, final), {"min_weight": float(rl.weights.min())})})
    out["huber_weights"] = rl.weights
    # influential mixtures
    for mx in sorted({out["diagnostics"]["max_t_mix"], out["diagnostics"]["cook_mix"]}):
        i = int(np.where(d.mix.values == mx)[0][0])
        dd = d.drop(index=i).reset_index(drop=True)
        yy = np.delete(y, i)
        md, _, _ = ols(final, dd, yy)
        t2 = list(search(yy, dd).iloc[0].terms)
        rob.append({"analysis": f"Same terms, mix {mx} excluded (re-selection: {'same model' if set(t2) == set(final) else ' '.join(t2)})",
                    "terms": " ".join(final), **summ(lin(md.params, final), {"R2": md.rsquared})})
    out["robustness"] = pd.DataFrame(rob)

    # resampling (parallel)
    log(f"  resampling ({n_boot} x 3)")
    _W_local = {"d": d, age: worker_payload, "strata": [np.where(d.carbonation.values == lv)[0] for lv in LEVELS],
                "opt_ss": ssf, "opt_rcf": rcff}
    rng = np.random.default_rng(SEED + int(age))
    seeds = rng.integers(0, 2 ** 31 - 1, size=3 * n_boot)
    modes = ("resid", "subsample", "case")
    jobs = [(int(seeds[k * n_boot + i]), age, md) for k, md in enumerate(modes) for i in range(n_boot)]
    with Pool(procs, initializer=_init, initargs=(_W_local,)) as pool:
        res = pool.map(_one, jobs, chunksize=25)
    boot = {}
    for md in modes:
        rr = [o for o, j in zip(res, jobs) if j[2] == md and o is not None]
        B = np.array([o["beta"] for o in rr])
        entry = {"n": len(rr),
                 "beta_ci": {nm: [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))] for j, nm in enumerate(names)},
                 "sign_agree": {nm: float(np.mean(np.sign(B[:, j]) == np.sign(m.params[j]))) for j, nm in enumerate(names)},
                 "opt_ss": {k: {"median": float(np.median([o["opt"][k] for o in rr])), "lo": float(np.percentile([o["opt"][k] for o in rr], 2.5)),
                                "hi": float(np.percentile([o["opt"][k] for o in rr], 97.5))} for k in opt_grids},
                 "opt_rcf": {k: {"median": float(np.median([o["rcf_opt"][k] for o in rr])), "lo": float(np.percentile([o["rcf_opt"][k] for o in rr], 2.5)),
                                 "hi": float(np.percentile([o["rcf_opt"][k] for o in rr], 97.5))} for k in rcf_grids}}
        if md != "case":
            inc = Counter(t for o in rr for t in o["terms"])
            entry["inclusion"] = {t: inc.get(t, 0) / len(rr) for t in dz.LEVEL_TERMS}
            entry["exact_final"] = float(np.mean([set(o["terms"]) == set(final) for o in rr]))
            entry["top_models"] = Counter(" ".join(o["terms"]) for o in rr).most_common(8)
        boot[md] = entry
    out["bootstrap"] = boot

    # evidence labels (same rule for both ages)
    ev = {}
    an = out["anova"].set_index("source")
    add = out["added_terms"].set_index("term") if len(out["added_terms"]) else None
    for t in dz.LEVEL_TERMS:
        rb, sb = boot["resid"]["inclusion"][t], boot["subsample"]["inclusion"][t]
        if t in final:
            p = an.loc[t, "p"]
            needed = any(t in dz.PARENTS[u] for u in final)
            if p < 0.01 and rb >= 0.9 and sb >= 0.8:
                ev[t] = "Robust"
            elif p < 0.05 and rb >= 0.6:
                ev[t] = "Moderate"
            else:
                ev[t] = "Hierarchy" if needed else "Weak"
        else:
            p = add.loc[t, "p"] if add is not None and t in add.index else 1.0
            ev[t] = "Suggestive" if (p < 0.10 or rb >= 0.5) else "Not detected"
    out["evidence"] = ev

    # design info for the report scripts
    out["model_export"] = {"names": names, "beta": m.params, "cov": m.cov_params(), "s2": m.scale, "df": int(m.df_resid),
                           "t975": float(tq)}
    # CSV tables
    pre = f"S{age}"
    coef.to_csv(os.path.join(TAB, f"{pre}_01_coefficients_coded.csv"), index=False)
    out["anova"].to_csv(os.path.join(TAB, f"{pre}_02_anova.csv"), index=False)
    pd.DataFrame(out["equation_actual"]).to_csv(os.path.join(TAB, f"{pre}_03_equation_actual_units.csv"), index=False)
    out["search_top"].to_csv(os.path.join(TAB, f"{pre}_04_top_models.csv"), index=False)
    out["fit_summary"].to_csv(os.path.join(TAB, f"{pre}_05_fit_summary_by_order.csv"), index=False)
    codes.to_csv(os.path.join(TAB, f"{pre}_06_carbonation_coding.csv"), index=False)
    out["nested_cv"].to_csv(os.path.join(TAB, f"{pre}_07_nested_cv.csv"), index=False)
    out["added_terms"].drop(columns="estimates").to_csv(os.path.join(TAB, f"{pre}_08_added_term_tests.csv"), index=False)
    out["diagnostics"]["table"].to_csv(os.path.join(TAB, f"{pre}_09_diagnostics_per_mix.csv"), index=False)
    out["robustness"].to_csv(os.path.join(TAB, f"{pre}_10_robustness.csv"), index=False)
    pd.DataFrame([{"term": t, "label": dz.label(t), "in_final": t in final, "akaike_importance": out["importance"][t],
                   "residual_bootstrap": boot["resid"]["inclusion"][t], "subsample_24of30": boot["subsample"]["inclusion"][t],
                   "evidence": ev[t]} for t in dz.LEVEL_TERMS]).to_csv(os.path.join(TAB, f"{pre}_11_term_support.csv"), index=False)
    for k, v in tr.items():
        (v if isinstance(v, pd.DataFrame) else pd.DataFrame([v])).to_csv(os.path.join(TAB, f"{pre}_12_trend_{k}.csv"), index=False)
    pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in o.items()} for o in opts]).to_csv(
        os.path.join(TAB, f"{pre}_13_optimum.csv"), index=False)
    return out


def cross_fit(d, res):
    """Supplementary: each model's terms refitted to the other age's data (not used for selection)."""
    out = {}
    for src, tgt in (("7", "28"), ("28", "7")):
        terms = res[src]["final_terms"]
        y = np.log(d[f"f{tgt}_mean"].values)
        m, X, names = ols(terms, d, y)
        tq = stats.t.ppf(0.975, m.df_resid)
        out[f"{src}_terms_on_{tgt}d"] = {"terms": terms, "coef": pd.DataFrame({"column": names, "coef": m.params, "se": m.bse, "p": m.pvalues,
                                                                                 "lo": m.params - tq * m.bse, "hi": m.params + tq * m.bse}),
                                          "anova": anova(m, names, terms), "fit": criteria(y, X)}
    return out


def main(n_boot=2000, procs=4):
    t0 = time.time()
    d = dz.load()
    dz.set_coding("cat4")
    R = {"meta": {"n_boot": n_boot, "seed": SEED, "date": time.strftime("%Y-%m-%d"), "numpy": np.__version__,
                  "statsmodels": sm.__version__ if hasattr(sm, "__version__") else "", "n_candidate_models": len(_SUBSETS)}}
    for age in AGES:
        R[age] = analyze(age, d, n_boot, procs)
    R["cross_fit"] = cross_fit(d, R)
    for k, v in R["cross_fit"].items():
        v["coef"].to_csv(os.path.join(TAB, f"S_cross_{k}.csv"), index=False)
    R["runtime_s"] = time.time() - t0
    with open(os.path.join(RES, "separate_results.json"), "w") as f:
        json.dump(to_py(R), f, allow_nan=False)
    log(f"done in {R['runtime_s']:.0f} s -> results/separate/separate_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(a.n_boot, a.procs)
