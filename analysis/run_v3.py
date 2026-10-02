"""
Analysis of the revised ("noiseless", v3) 7- and 28-day compressive strengths.

Question: one model for both curing ages, or one model per age?  And how do RCF, SS,
A/B and carbonation act on 7-day and on 28-day strength?

Protocol
 0. Data          v3 means (standard deviations not used); every difference from the
                  previous data set (v2) is logged and classified.
 1. Strategy      nested cross-validation (selection repeated in every fold) and AICc on
                  the joint likelihood compare (a) the shared-effects paired model of
                  run_analysis.py (an effect is the same at both ages unless an
                  age interaction is selected) with (b) a bivariate model whose two
                  equations have their own terms, plus reference procedures.
 2. Selection     per age: exhaustive AICc over all 716 hierarchical models of the full
                  candidate set (quadratic in RCF, A/B, SS; 4-level carbonation; K x
                  RCF, A/B, SS slopes); Akaike term weights, leave-one-out reselection,
                  residual and subsample bootstraps.
 3. Final model   bivariate REML/GLS fit with an unstructured 7/28-day covariance
                  (bivariate.py); term tests with Satterthwaite df, actual-unit
                  equations, fit statistics, lack of fit, diagnostics.
 4. Age effects   for every candidate term: its effect at 7 d and at 28 d and a test of
                  equality (term entered at both ages).
 5. Trends        effect ratios with 95 % CIs: SS, RCF, A/B, carbonation, the
                  interactions, strength gain f28/f7, optima.
 6. Sensitivity   same protocol on the previous data (v2), raw MPa scale, carbonation
                  coding, influential mixtures, the 28-d alternative models.

Run:  python analysis/run_v3.py            (2,000 residual-bootstrap and 1,000 subsample resamples)
      python analysis/run_v3.py --quick    (200 / 100, for testing)
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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence

import bivariate as bv
import design as dz
import model_tools as mt
import run_analysis as ra
import selection as sel
from lmm import PairedLMM

DATA_V3 = os.path.join(dz.ROOT, "data", "strength_7d_28d_v3_noiseless.csv")
DATA_V2 = os.path.join(dz.ROOT, "data", "strength_7d_28d_corrected.csv")
RES = os.path.join(dz.ROOT, "results", "v3")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)

SEED = 20261002
AB0 = 0.45                     # A/B at which RCF/SS profiles are reported (design centre)
CARB = dz.CARB_LEVELS
FULL_TERMS = list(dz.LEVEL_TERMS)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(df, name):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return df


def pts_frame(**kw):
    """Cartesian grid of points as a DataFrame (keys: RCF_pct, SS_pct, AB, carbonation, age)."""
    keys = list(kw)
    vals = [np.atleast_1d(kw[k]) for k in keys]
    mesh = np.array(np.meshgrid(*[np.arange(len(v)) for v in vals], indexing="ij")).reshape(len(keys), -1).T
    return pd.DataFrame({k: vals[j][mesh[:, j]] for j, k in enumerate(keys)})


# ============================================================================ 0. data
def revision_log(d2, d3):
    prev = pd.read_csv(os.path.join(dz.ROOT, "results", "tables", "T14_diagnostics_per_mix.csv"))
    rows = []
    for age in bv.AGES:
        for i in range(len(d3)):
            m2, m3 = d2.loc[i, f"f{age}_mean"], d3.loc[i, f"f{age}_mean"]
            s2, s3 = d2.loc[i, f"f{age}_sd"], d3.loc[i, f"f{age}_sd"]
            same_sd = (np.isnan(s2) and np.isnan(s3)) or np.isclose(s2, s3)
            if np.isclose(m2, m3) and same_sd:
                continue
            dm = round(m3 - m2, 2)
            grp = d3.loc[i, "replicate_group"] if isinstance(d3.loc[i, "replicate_group"], str) else ""
            toward_rep = None
            if grp:
                others = d2[(d2.replicate_group == grp) & (d2.mix != i + 1)][f"f{age}_mean"].mean()
                toward_rep = bool(abs(m3 - others) < abs(m2 - others))
            fit_prev = prev.loc[i, f"fit{age}"]
            if abs(dm) <= 0.011:
                kind = "last-digit change (0.01 MPa)"
            elif same_sd:
                kind = "mean shifted, SD unchanged"
            elif s3 > s2:
                kind = "mean shifted, SD increased"
            else:
                kind = "mean shifted, SD decreased"
            rows.append({"mix": i + 1, "age_d": age, "replicate_group": grp,
                         "mean_v2": m2, "mean_v3": m3, "change_MPa": dm,
                         "sd_v2": s2, "sd_v3": s3, "kind": kind,
                         "moved_toward_replicates": toward_rep,
                         "prev_model_fit_MPa": round(float(fit_prev), 2),
                         "moved_toward_prev_model_fit": bool(abs(m3 - fit_prev) < abs(m2 - fit_prev))})
    return pd.DataFrame(rows)


def pure_error(d):
    g = d.dropna(subset=["replicate_group"])
    out = {}
    for age in bv.AGES:
        for scale, col in (("MPa", f"f{age}_mean"), ("ln", f"l{age}")):
            ss = sum(((gg[col] - gg[col].mean()) ** 2).sum() for _, gg in g.groupby("replicate_group"))
            dfp = sum(len(gg) - 1 for _, gg in g.groupby("replicate_group"))
            out[f"{age}_{scale}"] = {"sd": float(np.sqrt(ss / dfp)), "df": int(dfp), "ss": float(ss)}
        out[f"{age}_ranges"] = {k: float(gg[f"f{age}_mean"].max() - gg[f"f{age}_mean"].min())
                                for k, gg in g.groupby("replicate_group")}
    return out


# ============================================================================ helpers
def per_age_ols(d, t, age, y=None):
    X, names = dz.mix_design(bv.sort_terms(t), d)
    y = d[f"l{age}"].values if y is None else y
    return sm.OLS(y, pd.DataFrame(X, columns=names)).fit()


def cv_metrics(Y, P):
    out = {}
    for j, a in enumerate(("7", "28")):
        e = Y[:, j] - P[:, j]
        out[f"Q2_ln_{a}"] = float(1 - (e ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum())
        em = np.exp(Y[:, j]) - np.exp(P[:, j])
        f = np.exp(Y[:, j])
        out[f"Q2_MPa_{a}"] = float(1 - (em ** 2).sum() / ((f - f.mean()) ** 2).sum())
        out[f"RMSE_MPa_{a}"] = float(np.sqrt((em ** 2).mean()))
        out[f"MAE_MPa_{a}"] = float(np.abs(em).mean())
    return out


def cv_bivariate(d, t7=None, t28=None, groups="mix", select=False, Y=None):
    """Leave-one-group-out predictions of the bivariate model; with select=True the
    per-age AICc selection is repeated inside every fold (nested CV)."""
    Y = d[["l7", "l28"]].values if Y is None else Y
    keys = d.mix.values if groups == "mix" else d.group.values
    P = np.zeros_like(Y)
    chosen = []
    for gk in np.unique(keys):
        te = keys == gk
        dtr, dte = d[~te].reset_index(drop=True), d[te].reset_index(drop=True)
        a7, a28 = t7, t28
        if select:
            s = bv.AgeSelector(dtr)
            a7, a28 = s.best(Y[~te, 0]), s.best(Y[~te, 1])
        chosen.append((tuple(bv.sort_terms(a7)), tuple(bv.sort_terms(a28))))
        m, _, _ = bv.fit(dtr, a7, a28, y=Y[~te], hessian=False)
        Xt, _ = bv.design(a7, a28, dte)
        P[te] = np.einsum("iap,p->ia", Xt, m.beta)
    return P, chosen


def joint_selector(structs):
    def f(dtr, ytr):
        a, terms, st = ra.select_overall(dtr, structs=structs, check_rank=True, y=ytr)
        return terms, st
    return f


# ============================================================================ bootstrap workers
_B = {}


def _init(payload):
    _B.update(payload)


def _resid_boot(seed):
    rng = np.random.default_rng(seed)
    d, fitted, res = _B["d"], _B["fitted"], _B["resid"]
    idx = rng.integers(0, len(d), len(d))
    ys = fitted + res[idx]
    s = _B["selector"]
    b7, b28 = s.best(ys[:, 0]), s.best(ys[:, 1])
    m, _, _ = bv.fit(d, _B["t7"], _B["t28"], y=ys, hessian=False)
    return {"t7": bv.sort_terms(b7), "t28": bv.sort_terms(b28),
            "q": (_B["Q"] @ m.beta).tolist()}


def _subsample(seed):
    rng = np.random.default_rng(seed)
    d = _B["d"]
    idx = np.sort(rng.choice(len(d), size=_B["n_sub"], replace=False))
    ds = d.iloc[idx].reset_index(drop=True)
    s = bv.AgeSelector(ds)
    return {"t7": bv.sort_terms(s.best(ds.l7.values)), "t28": bv.sort_terms(s.best(ds.l28.values))}


# ============================================================================ main
def main(quick=False, procs=4):
    t_start = time.time()
    n_boot, n_sub = (200, 100) if quick else (2000, 1000)
    R = {"meta": {"seed": SEED, "n_resid_boot": n_boot, "n_subsample": n_sub, "AB_profiles": AB0,
                  "data": os.path.relpath(DATA_V3, dz.ROOT), "previous_data": os.path.relpath(DATA_V2, dz.ROOT)}}
    d = dz.load(DATA_V3)
    d2 = dz.load(DATA_V2)
    Y = d[["l7", "l28"]].values
    n = len(d)

    # ------------------------------------------------------------------ 0. data
    log("data revision log")
    for c in ("carbonation", "RCF_pct", "SS_pct", "AB", "replicate_group"):
        assert d[c].astype(str).equals(d2[c].astype(str)), c
    rv = save(revision_log(d2, d3=d), "V00_revision_log_v2_to_v3")
    pe3, pe2 = pure_error(d), pure_error(d2)
    big = rv[rv.change_MPa.abs() > 0.011]
    R["revision"] = {
        "n_entries_changed": int(len(rv)), "n_mean_changes": int(len(big)),
        "n_last_digit": int((rv.change_MPa.abs() <= 0.011).sum()),
        "by_age": {str(a): int((big.age_d == a).sum()) for a in bv.AGES},
        "kinds": big.kind.value_counts().to_dict(),
        "sd_unchanged_shift": big[big.kind == "mean shifted, SD unchanged"][["mix", "age_d", "change_MPa"]].to_dict("records"),
        "n_sd_increased": int((rv.sd_v3 > rv.sd_v2 + 1e-9).sum()),
        "n_sd_decreased": int((rv.sd_v3 < rv.sd_v2 - 1e-9).sum()),
        "replicate_members_changed": int(big.replicate_group.astype(bool).sum()),
        "toward_replicates": int(big.moved_toward_replicates.fillna(False).astype(bool).sum()),
        "toward_prev_fit": int(big.moved_toward_prev_model_fit.sum()),
        "pure_error_v2": pe2, "pure_error_v3": pe3,
        "mix_variables_identical": True,
    }
    R["data"] = {"n_mixes": n, "n_design_points": int(d.group.nunique()),
                 "replicate_groups": {k: list(map(int, v)) for k, v in d.dropna(subset=["replicate_group"]).groupby("replicate_group").mix.apply(list).items()},
                 "f7": d.f7_mean.describe().to_dict(), "f28": d.f28_mean.describe().to_dict(),
                 "gain": (d.f28_mean / d.f7_mean).describe().to_dict(),
                 "corr_ln": float(np.corrcoef(d.l7, d.l28)[0, 1])}

    # ------------------------------------------------------------------ 2. per-age selection
    log("per-age exhaustive AICc selection")
    S = bv.AgeSelector(d)
    R["selection"] = {"n_models_per_age": len(S.subsets)}
    final = {}
    for age, j in ((7, 0), (28, 1)):
        rk = S.ranked(Y[:, j])
        final[age] = bv.sort_terms(rk[0][1])
        top = []
        for a, s, p in rk[:15]:
            o = per_age_ols(d, s, age)
            top.append({"rank": len(top) + 1, "AICc": a, "dAICc": a - rk[0][0], "n_coef": p,
                        "R2_ln": o.rsquared, "adjR2_ln": o.rsquared_adj,
                        "terms": " ".join(bv.sort_terms(s)), "labels": ", ".join(dz.label(t) for t in bv.sort_terms(s))})
        save(pd.DataFrame(top), f"V02_top_models_{age}d")
        w = bv.akaike_term_weights(rk)
        # leave-one-mixture-out reselection
        loo = Counter()
        loo_mix = {}
        for i in range(n):
            dd = d.drop(index=i).reset_index(drop=True)
            b = tuple(bv.sort_terms(bv.AgeSelector(dd).best(dd[f"l{age}"].values)))
            loo[b] += 1
            loo_mix.setdefault(b, []).append(i + 1)
        R["selection"][str(age)] = {
            "best": final[age], "AICc": rk[0][0], "within2": int(sum(r[0] - rk[0][0] < 2 for r in rk)),
            "within4": int(sum(r[0] - rk[0][0] < 4 for r in rk)), "akaike_weights": w,
            "loo_reselection": [{"terms": list(k), "count": c, "mixes_removed": loo_mix[k] if k != loo.most_common(1)[0][0] else []}
                                for k, c in loo.most_common()]}
    t7, t28 = final[7], final[28]
    R["final_terms"] = {"7": t7, "28": t28}
    log(f"final terms 7 d {t7}; 28 d {t28}")

    # Box-Cox per age (on its selected model) and for the bivariate model
    lams = np.round(np.arange(-1.5, 1.501, 0.01), 2)
    bc = {}
    cols = sel.mix_columns(d)
    for age, t in ((7, t7), (28, t28)):
        Yr = d[f"f{age}_mean"].values
        X = sel.mix_X(t, cols, n)
        L = np.array([sel.ols_ll(np.log(Yr) if abs(l) < 1e-9 else (Yr ** l - 1) / l, X) + (l - 1) * np.log(Yr).sum() for l in lams])
        ci = lams[L >= L.max() - stats.chi2.ppf(0.95, 1) / 2]
        i1 = np.argmin(np.abs(lams - 1))
        bc[str(age)] = {"lambda": float(lams[L.argmax()]), "ci": [float(ci.min()), float(ci.max())],
                        "LR_lambda1": float(2 * (L.max() - L[i1])), "p_lambda1": float(stats.chi2.sf(2 * (L.max() - L[i1]), 1)),
                        "curve": (L - L.max()).tolist()}
    Yr = d[["f7_mean", "f28_mean"]].values
    Lb = []
    for l in lams:
        Z = np.log(Yr) if abs(l) < 1e-9 else (Yr ** l - 1) / l
        m_ = PairedLMM(Z, *bv.design(t7, t28, d), "UN", method="ML", hessian=False)
        Lb.append(m_.ll + (l - 1) * np.log(Yr).sum())
    Lb = np.array(Lb)
    ci = lams[Lb >= Lb.max() - stats.chi2.ppf(0.95, 1) / 2]
    bc["bivariate"] = {"lambda": float(lams[Lb.argmax()]), "ci": [float(ci.min()), float(ci.max())],
                       "curve": (Lb - Lb.max()).tolist()}
    bc["lambdas"] = lams.tolist()
    R["boxcox"] = bc

    # ------------------------------------------------------------------ 1. strategy comparison
    log("strategy comparison: AICc on the joint likelihood")
    M, G = d.M.values, d.G.values
    shared = {}
    for st in ("UN", "CS"):
        top, nmod = sel.enumerate_models(M, G, sel.mix_columns(d), struct=st, n_top=1)
        shared[st] = (top[0][0], sel.joint_terms(top[0][1], top[0][2]))
    m_fin, X_fin, names = bv.fit(d, t7, t28)
    aic_rows = [{"model": "Bivariate, age-specific terms (final)", "struct": "UN",
                 "AICc": m_fin.aicc_ml()[0], "k": m_fin.aicc_ml()[2],
                 "terms": f"7 d: {' '.join(t7)} | 28 d: {' '.join(t28)}"}]
    m_ind, _, _ = bv.fit(d, t7, t28, struct="INDH")
    aic_rows.append({"model": "Same terms, pairing ignored (independent ages)", "struct": "INDH",
                     "AICc": m_ind.aicc_ml()[0], "k": m_ind.aicc_ml()[2], "terms": "as final"})
    for st in ("UN", "CS"):
        aic_rows.append({"model": f"Shared-effects paired model, AICc-best ({st})", "struct": st,
                         "AICc": shared[st][0], "k": None, "terms": " ".join(shared[st][1])})
    aic = pd.DataFrame(aic_rows)
    aic["dAICc"] = aic.AICc - aic.AICc.min()
    save(aic, "V05_strategy_AICc")
    R["strategy_aicc"] = aic

    log("strategy comparison: nested cross-validation")
    procs_cv = []
    for groups, gname in (("mix", "leave one mixture out"), ("group", "leave one design point out")):
        P, ch = cv_bivariate(d, groups=groups, select=True)
        procs_cv.append({"cv": gname, "procedure": "Bivariate, per-age AICc selection (final procedure)",
                         "distinct_models": len(set(ch)), **cv_metrics(Y, P)})
        for structs, nm in ((("UN",), "Shared-effects paired model, AICc (UN)"),
                            (("CS", "UN"), "Shared-effects paired model, AICc (CS + UN; earlier protocol)")):
            P, ch = ra.cv_predict(d, None, groups=groups, select=joint_selector(structs))
            procs_cv.append({"cv": gname, "procedure": nm, "distinct_models": len(set(ch)), **cv_metrics(Y, P)})
        P, _ = cv_bivariate(d, FULL_TERMS, FULL_TERMS, groups=groups)
        procs_cv.append({"cv": gname, "procedure": "Full candidate model at both ages (no selection)",
                         "distinct_models": 1, **cv_metrics(Y, P)})
        P, _ = cv_bivariate(d, ["D", "D2"], ["D", "D2"], groups=groups)
        procs_cv.append({"cv": gname, "procedure": "SS + SS² only at both ages (reference)",
                         "distinct_models": 1, **cv_metrics(Y, P)})
        P, _ = cv_bivariate(d, t7, t28, groups=groups)
        procs_cv.append({"cv": gname, "procedure": "Final model, terms fixed (not a nested estimate)",
                         "distinct_models": 1, **cv_metrics(Y, P)})
        prev_final = ['A', 'D', 'Carb', 'AD', 'D2', 'KA', 'KD', 'Age', 'D.Age', 'D2.Age']
        P, _ = ra.cv_predict(d, prev_final, struct="UN", groups=groups)
        procs_cv.append({"cv": gname, "procedure": "Earlier shared-effects final model, terms fixed",
                         "distinct_models": 1, **cv_metrics(Y, P)})
    R["strategy_cv"] = save(pd.DataFrame(procs_cv), "V04_strategy_nested_cv")

    # ------------------------------------------------------------------ 3. final model
    log("final bivariate model")
    m = m_fin
    vc = m.variance_components()
    R["final"] = {"variance": vc, "AICc": m.aicc_ml()[0], "loglik_REML": m.ll}
    ct = []
    for j, nm in enumerate(names):
        l = np.zeros(len(names)); l[j] = 1
        est, se, df, t, p = m.t_test(l)
        q = stats.t.ppf(0.975, df)
        base, age = bv.split_name(nm)
        ct.append({"age_d": age, "column": base, "term": dz.term_of_column(base), "label": dz.label(dz.term_of_column(base)) if base != "Intercept" else "Intercept",
                   "coef": est, "se": se, "df": df, "t": t, "p": p, "lo": est - q * se, "hi": est + q * se})
    ct = pd.DataFrame(ct)
    # per-age OLS estimates for comparison
    for age, t in ((7, t7), (28, t28)):
        o = per_age_ols(d, t, age)
        ct.loc[ct.age_d == age, "ols_coef"] = o.params.values
        ct.loc[ct.age_d == age, "ols_se"] = o.bse.values
    R["coef"] = save(ct, "V06_final_coefficients_coded")
    tt = []
    for age, t in ((7, t7), (28, t28)):
        for term in t:
            F, q, df, p = m.f_test(bv.term_L(names, term, age))
            tt.append({"age_d": age, "term": term, "label": dz.label(term), "df1": q, "df2": df, "F": F, "p": p})
    R["term_tests"] = save(pd.DataFrame(tt), "V07_final_term_tests")
    eqs = {k: bv.actual_equation(m, t7, t28, k) for k in ("7 d", "28 d", "gain")}
    R["equations"] = eqs
    save(pd.concat([pd.DataFrame(v).assign(equation=k) for k, v in eqs.items()]), "V08_equations_actual_units")

    # fit statistics per age
    fitted = np.einsum("iap,p->ia", X_fin, m.beta)
    P_loo, _ = cv_bivariate(d, t7, t28, groups="mix")
    P_ldp, _ = cv_bivariate(d, t7, t28, groups="group")
    se_fit = np.sqrt(np.einsum("iap,pq,iaq->ia", X_fin, m.cov_beta, X_fin))
    fs = {}
    for age, j, t in ((7, 0, t7), (28, 1, t28)):
        y, f = Y[:, j], fitted[:, j]
        p = int((ct.age_d == age).sum())
        sst = ((y - y.mean()) ** 2).sum()
        sse = ((y - f) ** 2).sum()
        o = per_age_ols(d, t, age)
        pe = pe3[f"{age}_ln"]
        sslof, dflof = o.ssr - pe["ss"], o.df_resid - pe["df"]
        Flof = (sslof / dflof) / (pe["ss"] / pe["df"])
        fy, ff = np.exp(y), np.exp(f)
        fs[str(age)] = {
            "n_coef": p, "R2_ln": 1 - sse / sst, "adjR2_ln": 1 - (sse / (n - p)) / (sst / (n - 1)),
            "R2_MPa": 1 - ((fy - ff) ** 2).sum() / ((fy - fy.mean()) ** 2).sum(),
            "predR2_ln_LOO": 1 - ((y - P_loo[:, j]) ** 2).sum() / sst,
            "predR2_ln_LODP": 1 - ((y - P_ldp[:, j]) ** 2).sum() / sst,
            "predR2_MPa_LOO": 1 - ((fy - np.exp(P_loo[:, j])) ** 2).sum() / ((fy - fy.mean()) ** 2).sum(),
            "RMSE_fit_MPa": float(np.sqrt(((fy - ff) ** 2).mean())),
            "RMSE_pred_MPa": float(np.sqrt(((fy - np.exp(P_loo[:, j])) ** 2).mean())),
            "resid_sd_ln": float(np.sqrt(m.Sigma[j, j])),
            "pure_error_sd_ln": pe["sd"],
            "adeq_precision": float((f.max() - f.min()) / np.sqrt((se_fit[:, j] ** 2).mean())),
            "lof_F": Flof, "lof_df": [int(dflof), pe["df"]], "lof_p": float(stats.f.sf(Flof, dflof, pe["df"])),
            "ols_R2_ln": o.rsquared, "ols_adjR2_ln": o.rsquared_adj,
        }
    R["fit_stats"] = fs

    # diagnostics (per-age OLS influence measures + whitened residuals of the bivariate fit)
    W = m.whitened_residuals()
    diag = d[["mix", "carbonation", "RCF_pct", "SS_pct", "AB", "f7_mean", "f28_mean"]].copy()
    R["diagnostics"] = {}
    for age, j, t in ((7, 0, t7), (28, 1, t28)):
        o = per_age_ols(d, t, age)
        inf = OLSInfluence(o)
        tdel = inf.resid_studentized_external
        diag[f"fit{age}"] = np.exp(fitted[:, j])
        diag[f"resid_ln{age}"] = Y[:, j] - fitted[:, j]
        diag[f"t_del{age}"] = tdel
        diag[f"hat{age}"] = inf.hat_matrix_diag
        diag[f"cook{age}"] = inf.cooks_distance[0]
        dfr = o.df_resid - 1
        diag[f"p_bonf{age}"] = np.minimum(1, n * 2 * stats.t.sf(np.abs(tdel), dfr))
        bp = het_breuschpagan(o.resid, sm.add_constant(o.fittedvalues))
        R["diagnostics"][str(age)] = {
            "shapiro_p": float(stats.shapiro(tdel).pvalue), "bp_p": float(bp[1]),
            "max_abs_tdel": float(np.abs(tdel).max()), "max_tdel_mix": int(diag.mix.iloc[int(np.abs(tdel).argmax())]),
            "min_p_bonf": float(diag[f"p_bonf{age}"].min()),
            "max_cook": float(inf.cooks_distance[0].max()), "max_cook_mix": int(diag.mix.iloc[int(inf.cooks_distance[0].argmax())]),
            "order_spearman": [float(x) for x in stats.spearmanr(d.mix, Y[:, j] - fitted[:, j])],
        }
    R["diagnostics"]["whitened_shapiro_p"] = float(stats.shapiro(W.ravel()).pvalue)
    save(diag, "V09_diagnostics_per_mix")
    pred_all = []
    for age in bv.AGES:
        p_ = d[["RCF_pct", "SS_pct", "AB", "carbonation"]].copy(); p_["age"] = age
        pr = bv.predict(m, t7, t28, p_)
        pr.insert(0, "observed", d[f"f{age}_mean"].values); pr.insert(0, "age_d", age); pr.insert(0, "mix", d.mix.values)
        pr["loo_pred"] = np.exp(P_loo[:, 0 if age == 7 else 1])
        pred_all.append(pr)
    save(pd.concat(pred_all), "V10_predictions_all_mixes")

    # ------------------------------------------------------------------ 4. age comparison + added terms
    log("effects at 7 d vs 28 d")
    cmp_rows = []
    for t in FULL_TERMS:
        u7 = bv.sort_terms(set(t7) | {t} | set(dz.PARENTS[t]))
        u28 = bv.sort_terms(set(t28) | {t} | set(dz.PARENTS[t]))
        mu, _, nu = bv.fit(d, u7, u28)
        L7, L28 = bv.term_L(nu, t, 7), bv.term_L(nu, t, 28)
        F, q, df, p = mu.f_test(L28 - L7)
        F7 = mu.f_test(L7); F28 = mu.f_test(L28)
        row = {"term": t, "label": dz.label(t), "in_7d_model": t in t7, "in_28d_model": t in t28,
               "p_7d": F7[3], "p_28d": F28[3], "F_diff": F, "df1_diff": q, "df2_diff": df, "p_diff": p}
        if L7.shape[0] == 1:
            e7 = mt.lincomb(mu, L7).iloc[0]; e28 = mt.lincomb(mu, L28).iloc[0]
            row.update({"coef_7d": e7.est, "lo_7d": e7.lo, "hi_7d": e7.hi, "coef_28d": e28.est, "lo_28d": e28.lo, "hi_28d": e28.hi})
        else:   # carbonation: one coefficient per level (0.5 h, 1 h, 5 h vs NC)
            row.update({"levels": "; ".join(f"{lv}: {b7:+.3f} / {b28:+.3f}" for lv, b7, b28 in
                                            zip(CARB[1:], mu.beta[np.argmax(L7, axis=1)], mu.beta[np.argmax(L28, axis=1)]))})
        cmp_rows.append(row)
    R["age_comparison"] = save(pd.DataFrame(cmp_rows), "V11_effect_7d_vs_28d")
    added = []
    for age, t_own in ((7, t7), (28, t28)):
        for t in FULL_TERMS:
            if t in t_own:
                continue
            ext = bv.sort_terms(set(t_own) | {t} | set(dz.PARENTS[t]))
            mu, _, nu = bv.fit(d, ext if age == 7 else t7, ext if age == 28 else t28)
            L = np.vstack([bv.term_L(nu, x, age) for x in ext if x not in t_own])
            F, q, df, p = mu.f_test(L)
            added.append({"age_d": age, "added": "+".join(x for x in ext if x not in t_own),
                          "label": ", ".join(dz.label(x) for x in ext if x not in t_own), "F": F, "df1": q, "df2": df, "p": p})
    R["added_terms"] = save(pd.DataFrame(added), "V12_added_term_tests")

    # ------------------------------------------------------------------ 5. trends and interactions
    log("trends, interactions, gain, optima")

    def contrast(pa, pb):
        """ln f(pa) - ln f(pb) rows (same age columns in pa/pb)."""
        return bv.rows(t7, t28, pa) - bv.rows(t7, t28, pb)

    # SS: effect of SS 0 -> 37.5 -> 75 and SS at maximum strength
    ss_grid = np.round(np.arange(0, 75.01, 0.25), 2)
    ss_rows = []
    for age in bv.AGES:
        for c in CARB:
            for rcf in (10, 30, 50):
                g = pts_frame(RCF_pct=rcf, SS_pct=ss_grid, AB=AB0, carbonation=c, age=age)
                v = bv.rows(t7, t28, g) @ m.beta
                k = int(np.argmax(v))
                base = pts_frame(RCF_pct=rcf, SS_pct=0.0, AB=AB0, carbonation=c, age=age)
                for ss in (37.5, 75.0, float(ss_grid[k])):
                    r = bv.ratio(m, contrast(pts_frame(RCF_pct=rcf, SS_pct=ss, AB=AB0, carbonation=c, age=age), base)).iloc[0]
                    ss_rows.append({"age_d": age, "carbonation": c, "RCF": rcf, "SS_to": ss,
                                    "is_optimum": ss == float(ss_grid[k]) and k not in (0, len(ss_grid) - 1),
                                    "ss_at_max": float(ss_grid[k]), "ratio_vs_SS0": r.ratio, "lo": r.lo, "hi": r.hi, "p": r.p})
    R["trend_ss"] = save(pd.DataFrame(ss_rows), "V13_trend_SS")

    # RCF: RCF 50 vs 10 and RCF at maximum
    rcf_grid = np.round(np.arange(10, 50.01, 0.25), 2)
    rc_rows = []
    for age in bv.AGES:
        for c in CARB:
            for ss in (0.0, 37.5, 75.0):
                g = pts_frame(RCF_pct=rcf_grid, SS_pct=ss, AB=AB0, carbonation=c, age=age)
                v = bv.rows(t7, t28, g) @ m.beta
                k = int(np.argmax(v))
                r = bv.ratio(m, contrast(pts_frame(RCF_pct=50, SS_pct=ss, AB=AB0, carbonation=c, age=age),
                                         pts_frame(RCF_pct=10, SS_pct=ss, AB=AB0, carbonation=c, age=age))).iloc[0]
                rc_rows.append({"age_d": age, "carbonation": c, "SS": ss, "rcf_at_max": float(rcf_grid[k]),
                                "ratio_RCF50_vs_10": r.ratio, "lo": r.lo, "hi": r.hi, "p": r.p})
    R["trend_rcf"] = save(pd.DataFrame(rc_rows), "V14_trend_RCF")

    # carbonation level vs NC
    cb_rows = []
    for age in bv.AGES:
        for c in CARB[1:]:
            for rcf in (10, 30, 50):
                for ss in (0.0, 37.5, 75.0):
                    r = bv.ratio(m, contrast(pts_frame(RCF_pct=rcf, SS_pct=ss, AB=AB0, carbonation=c, age=age),
                                             pts_frame(RCF_pct=rcf, SS_pct=ss, AB=AB0, carbonation="NC", age=age))).iloc[0]
                    cb_rows.append({"age_d": age, "carbonation": c, "RCF": rcf, "SS": ss,
                                    "ratio_vs_NC": r.ratio, "lo": r.lo, "hi": r.hi, "p": r.p})
    R["trend_carb"] = save(pd.DataFrame(cb_rows), "V15_trend_carbonation")
    # carbonation durations among themselves (both ages)
    dur = []
    for age in bv.AGES:
        L = np.vstack([contrast(pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation=a, age=age),
                                pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation="1 h", age=age)) for a in ("0.5 h", "5 h")])
        F, q, df, p = m.f_test(L)
        L2 = np.vstack([contrast(pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation=a, age=age),
                                 pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation="NC", age=age)) for a in CARB[1:]])
        F2, q2, df2, p2 = m.f_test(L2)
        dur.append({"age_d": age, "durations_differ_F": F, "durations_differ_p": p, "any_vs_NC_F": F2, "any_vs_NC_p": p2})
    R["carb_durations"] = dur

    # A/B
    ab_rows = []
    ab_grid = np.round(np.arange(0.42, 0.4801, 0.001), 3)
    for age in bv.AGES:
        g = pts_frame(RCF_pct=30, SS_pct=37.5, AB=ab_grid, carbonation="NC", age=age)
        v = bv.rows(t7, t28, g) @ m.beta
        for ab in (0.42, 0.48):
            r = bv.ratio(m, contrast(pts_frame(RCF_pct=30, SS_pct=37.5, AB=ab, carbonation="NC", age=age),
                                     pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.45, carbonation="NC", age=age))).iloc[0]
            ab_rows.append({"age_d": age, "AB_to": ab, "ratio_vs_AB045": r.ratio, "lo": r.lo, "hi": r.hi, "p": r.p,
                            "ab_at_min": float(ab_grid[int(np.argmin(v))]), "ab_at_max": float(ab_grid[int(np.argmax(v))])})
    R["trend_ab"] = save(pd.DataFrame(ab_rows), "V16_trend_AB")

    # strength gain f28 / f7
    gn = []
    for c in CARB:
        for rcf in (10, 30, 50):
            for ss in (0.0, 37.5, 75.0):
                for ab in (0.42, 0.45, 0.48):
                    r = bv.ratio(m, bv.gain_rows(t7, t28, pts_frame(RCF_pct=rcf, SS_pct=ss, AB=ab, carbonation=c))).iloc[0]
                    gn.append({"carbonation": c, "RCF": rcf, "SS": ss, "AB": ab, "gain_f28_over_f7": r.ratio, "lo": r.lo, "hi": r.hi})
    R["gain"] = save(pd.DataFrame(gn), "V17_gain_ratio")

    # optimum per age and carbonation over the design region
    og = pts_frame(RCF_pct=np.arange(10, 50.01, 1.0), SS_pct=np.arange(0, 75.01, 1.0), AB=np.round(np.arange(0.42, 0.4801, 0.005), 3))
    opt = []
    for age in bv.AGES:
        uses_ab = any(t in (t7 if age == 7 else t28) for t in ("C", "AC", "CD", "C2", "KC"))
        for c in CARB:
            g = og.assign(carbonation=c, age=age)
            v = bv.rows(t7, t28, g) @ m.beta
            k = int(np.argmax(v))
            p_ = g.iloc[[k]].reset_index(drop=True)
            pr = bv.predict(m, t7, t28, p_).iloc[0]
            within = g[v >= v.max() + np.log(0.95)]
            pc = dz.new_points(p_)
            same = d[d.carbonation == c]
            dist = float(np.min(np.sqrt(((np.column_stack([same.A, same.C, same.D]) - np.array([pc.A[0], pc.C[0], pc.D[0]])) ** 2).sum(1))))
            opt.append({"age_d": age, "carbonation": c, "RCF": float(p_.RCF_pct[0]), "SS": float(p_.SS_pct[0]),
                        "AB": float(p_.AB[0]) if uses_ab else None, "AB_in_model": uses_ab,
                        "at_design_boundary": bool(p_.RCF_pct[0] in (10, 50) or p_.SS_pct[0] in (0, 75) or (uses_ab and p_.AB[0] in (0.42, 0.48))),
                        "median": pr["median"], "ci_lo": pr.ci_lo, "ci_hi": pr.ci_hi, "pi_lo": pr.pi_lo, "pi_hi": pr.pi_hi,
                        "within5_RCF": [float(within.RCF_pct.min()), float(within.RCF_pct.max())],
                        "within5_SS": [float(within.SS_pct.min()), float(within.SS_pct.max())],
                        "within5_AB": [float(within.AB.min()), float(within.AB.max())],
                        "coded_distance_to_nearest_mix_same_carb": dist})
    R["optimum"] = save(pd.DataFrame(opt), "V18_optimum")

    # which compositions carry the 28-day effects?  (a) leave one design point out:
    # terms re-selected without it and its prediction error; (b) the final terms refitted
    # without each replicate group: carbonation and A/B effects at 28 d
    log("design-point influence")
    ldp = []
    for gk in d.group.unique():
        te = (d.group == gk).values
        dtr, dte = d[~te].reset_index(drop=True), d[te].reset_index(drop=True)
        Sg = bv.AgeSelector(dtr)
        a7, a28 = bv.sort_terms(Sg.best(dtr.l7.values)), bv.sort_terms(Sg.best(dtr.l28.values))
        mg, _, _ = bv.fit(dtr, a7, a28, hessian=False)
        Xt, _ = bv.design(a7, a28, dte)
        pg = np.exp(np.einsum("iap,p->ia", Xt, mg.beta))
        for k_, (_, r) in enumerate(dte.iterrows()):
            ldp.append({"design_point": gk, "mix": int(r.mix), "carbonation": r.carbonation, "RCF": r.RCF_pct, "SS": r.SS_pct, "AB": r.AB,
                        "obs7": r.f7_mean, "pred7": pg[k_, 0], "err_ln7": np.log(r.f7_mean / pg[k_, 0]), "same_terms_7": a7 == t7,
                        "obs28": r.f28_mean, "pred28": pg[k_, 1], "err_ln28": np.log(r.f28_mean / pg[k_, 1]),
                        "terms_28_without_it": " ".join(a28), "carb_in_28_without_it": "Carb" in a28})
    R["design_point_influence"] = save(pd.DataFrame(ldp), "V22_leave_design_point_out")
    grp = []
    for gk in ["none"] + sorted(d.dropna(subset=["replicate_group"]).replicate_group.unique()):
        dg = d if gk == "none" else d[d.replicate_group != gk].reset_index(drop=True)
        mg, _, ng = bv.fit(dg, t7, t28)
        F, q, df_, p = mg.f_test(bv.term_L(ng, "Carb", 28))
        row = {"excluded": gk, "n_mixes": len(dg), "carb28_F": F, "carb28_p": p}
        for c in CARB[1:]:
            L = bv.rows(t7, t28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation=c, age=28)) - \
                bv.rows(t7, t28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation="NC", age=28))
            rr = bv.ratio(mg, L).iloc[0]
            row[f"28d {c}/NC"] = rr.ratio; row[f"p {c}"] = rr.p
        L = bv.rows(t7, t28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.42, carbonation="NC", age=28)) - \
            bv.rows(t7, t28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.45, carbonation="NC", age=28))
        rr = bv.ratio(mg, L).iloc[0]
        row["28d AB0.42/0.45"] = rr.ratio; row["p AB"] = rr.p
        row["C2_28_p"] = mg.f_test(bv.term_L(ng, "C2", 28))[3]
        grp.append(row)
    R["replicate_group_exclusion"] = save(pd.DataFrame(grp), "V23_28d_effects_without_each_replicate_group")

    # ------------------------------------------------------------------ 6. resampling
    log(f"resampling: {n_boot} residual-pair bootstrap, {n_sub} subsamples (24/30)")
    resid = Y - fitted
    for j, age in enumerate(bv.AGES):
        p = int((ct.age_d == age).sum())
        resid[:, j] *= np.sqrt(n / (n - p))
    # quantities tracked in the bootstrap (rows of Q, l'beta on the ln scale)
    qdef = []
    for ss in (0.0, 37.5, 75.0):
        qdef.append((f"gain NC RCF30 SS{ss:g}", bv.gain_rows(t7, t28, pts_frame(RCF_pct=30, SS_pct=ss, AB=AB0, carbonation="NC"))[0]))
    for age in bv.AGES:
        for c in CARB[1:]:
            qdef.append((f"{age}d {c}/NC RCF30 SS37.5", contrast(pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation=c, age=age),
                                                               pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation="NC", age=age))[0]))
        qdef.append((f"{age}d RCF50/10 NC SS0", contrast(pts_frame(RCF_pct=50, SS_pct=0, AB=AB0, carbonation="NC", age=age),
                                                       pts_frame(RCF_pct=10, SS_pct=0, AB=AB0, carbonation="NC", age=age))[0]))
        qdef.append((f"{age}d RCF50/10 1h SS0", contrast(pts_frame(RCF_pct=50, SS_pct=0, AB=AB0, carbonation="1 h", age=age),
                                                       pts_frame(RCF_pct=10, SS_pct=0, AB=AB0, carbonation="1 h", age=age))[0]))
        qdef.append((f"{age}d AB0.42/0.45", contrast(pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.42, carbonation="NC", age=age),
                                                    pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.45, carbonation="NC", age=age))[0]))
    qdef = [(nm, q) for nm, q in qdef if not np.allclose(q, 0)]    # e.g. A/B at 7 d: not in that equation
    Q = np.array([q for _, q in qdef])
    payload = {"d": d, "fitted": fitted, "resid": resid, "selector": S, "t7": t7, "t28": t28, "Q": Q, "n_sub": 24}
    rng = np.random.default_rng(SEED)
    seeds_b = rng.integers(0, 2 ** 31, n_boot)
    seeds_s = rng.integers(0, 2 ** 31, n_sub)
    with Pool(procs, initializer=_init, initargs=(payload,)) as pool:
        boot = pool.map(_resid_boot, seeds_b, chunksize=20)
        subs = pool.map(_subsample, seeds_s, chunksize=20)
    freq = []
    for nm, res_ in (("residual bootstrap", boot), ("subsample 24/30", subs)):
        for age, t_own in ((7, t7), (28, t28)):
            key = f"t{age}"
            for t in FULL_TERMS:
                freq.append({"scheme": nm, "age_d": age, "term": t, "label": dz.label(t), "in_final": t in t_own,
                             "freq": float(np.mean([t in r[key] for r in res_]))})
            freq.append({"scheme": nm, "age_d": age, "term": "EXACT", "label": "exact final model", "in_final": True,
                         "freq": float(np.mean([r[key] == t_own for r in res_]))})
    R["boot_freq"] = save(pd.DataFrame(freq), "V19_selection_frequency")
    qb = np.array([r["q"] for r in boot])
    qrows = []
    for (nm, l), col in zip(qdef, qb.T):
        est, se, df_, tt_, p_ = m.t_test(l)
        q = stats.t.ppf(0.975, df_)
        qrows.append({"quantity": nm, "ratio": np.exp(est), "lo_t": np.exp(est - q * se), "hi_t": np.exp(est + q * se),
                      "lo_boot": float(np.exp(np.percentile(col, 2.5))), "hi_boot": float(np.exp(np.percentile(col, 97.5)))})
    R["boot_ci"] = save(pd.DataFrame(qrows), "V20_key_ratios_t_and_bootstrap_CI")

    # ------------------------------------------------------------------ 7. sensitivity
    log("sensitivity analyses")
    sens = []

    def summarize(label, dd, a7, a28, scale="ln", note=""):
        Yd = dd[["l7", "l28"]].values
        if scale == "MPa":
            Ys = dd[["f7_mean", "f28_mean"]].values
            mm, Xs, _ = bv.fit(dd, a7, a28, y=Ys)
            ff = np.einsum("iap,p->ia", Xs, mm.beta)
            P, _ = cv_bivariate(dd, a7, a28, Y=Ys)
            r2 = [1 - ((Ys[:, j] - ff[:, j]) ** 2).sum() / ((Ys[:, j] - Ys[:, j].mean()) ** 2).sum() for j in (0, 1)]
            q2 = [1 - ((Ys[:, j] - P[:, j]) ** 2).sum() / ((Ys[:, j] - Ys[:, j].mean()) ** 2).sum() for j in (0, 1)]
            sens.append({"analysis": label, "terms_7d": " ".join(a7), "terms_28d": " ".join(a28), "R2_MPa_7": r2[0], "R2_MPa_28": r2[1],
                         "predR2_MPa_7": q2[0], "predR2_MPa_28": q2[1], "note": note})
            return
        mm, Xs, nn = bv.fit(dd, a7, a28)
        ff = np.einsum("iap,p->ia", Xs, mm.beta)
        P, _ = cv_bivariate(dd, a7, a28)
        r2 = [1 - ((Yd[:, j] - ff[:, j]) ** 2).sum() / ((Yd[:, j] - Yd[:, j].mean()) ** 2).sum() for j in (0, 1)]
        out = {"analysis": label, "terms_7d": " ".join(a7), "terms_28d": " ".join(a28),
               "R2_ln_7": r2[0], "R2_ln_28": r2[1], "AICc": mm.aicc_ml()[0], **cv_metrics(Yd, P), "note": note}
        for c in CARB[1:]:
            L = bv.rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation=c, age=28)) - \
                bv.rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=AB0, carbonation="NC", age=28))
            out[f"28d {c}/NC"] = float(np.exp(L @ mm.beta)[0])
        L = bv.rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.42, carbonation="NC", age=28)) - \
            bv.rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=37.5, AB=0.45, carbonation="NC", age=28))
        out["28d AB0.42/0.45"] = float(np.exp(L @ mm.beta)[0])
        L = bv.gain_rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=0.0, AB=AB0, carbonation="NC"))
        out["gain SS0"] = float(np.exp(L @ mm.beta)[0])
        L = bv.gain_rows(a7, a28, pts_frame(RCF_pct=30, SS_pct=75.0, AB=AB0, carbonation="NC"))
        out["gain SS75"] = float(np.exp(L @ mm.beta)[0])
        sens.append(out)

    summarize("Final model (v3 data)", d, t7, t28)
    S2 = bv.AgeSelector(d2)
    v2_7, v2_28 = bv.sort_terms(S2.best(d2.l7.values)), bv.sort_terms(S2.best(d2.l28.values))
    R["v2_selection"] = {"7": v2_7, "28": v2_28,
                         "akaike_weights_28": bv.akaike_term_weights(S2.ranked(d2.l28.values)),
                         "akaike_weights_7": bv.akaike_term_weights(S2.ranked(d2.l7.values))}
    summarize("Previous data (v2): same protocol, re-selected", d2, v2_7, v2_28)
    summarize("Previous data (v2): final v3 terms", d2, t7, t28)
    # the v3-selected 28-d terms tested on v2 data
    m2, _, n2 = bv.fit(d2, t7, t28)
    R["v2_with_v3_terms_tests"] = [{"age_d": 28, "term": t, "p": m2.f_test(bv.term_L(n2, t, 28))[3]} for t in t28] + \
                                  [{"age_d": 7, "term": t, "p": m2.f_test(bv.term_L(n2, t, 7))[3]} for t in t7]
    # alternative 28-d models within 4 AICc
    for a, s, p in S.ranked(Y[:, 1], 6)[1:4]:
        summarize(f"28-d alternative (dAICc {a - S.ranked(Y[:, 1], 1)[0][0]:.1f})", d, t7, bv.sort_terms(s))
    # influential mixtures
    for mix in sorted({R["diagnostics"]["28"]["max_tdel_mix"], R["diagnostics"]["28"]["max_cook_mix"],
                       R["diagnostics"]["7"]["max_cook_mix"]}):
        dd = d[d.mix != mix].reset_index(drop=True)
        Sd = bv.AgeSelector(dd)
        r7, r28 = bv.sort_terms(Sd.best(dd.l7.values)), bv.sort_terms(Sd.best(dd.l28.values))
        summarize(f"Mix {mix} excluded, re-selected", dd, r7, r28,
                  note=f"re-selection: 7 d {'same' if r7 == t7 else ' '.join(r7)}; 28 d {'same' if r28 == t28 else ' '.join(r28)}")
    summarize("Raw MPa scale, final terms", d, t7, t28, scale="MPa")
    # carbonation coded on/off at 28 d
    dz.set_coding("onoff")
    So = bv.AgeSelector(d)
    ro = So.ranked(Y[:, 1], 1)[0]
    dz.set_coding("cat4")
    R["carb_onoff_28"] = {"best_terms": bv.sort_terms(ro[1]), "AICc": ro[0], "AICc_cat4": S.ranked(Y[:, 1], 1)[0][0]}
    R["sensitivity"] = save(pd.DataFrame(sens), "V21_sensitivity")

    R["meta"]["runtime_s"] = time.time() - t_start
    with open(os.path.join(RES, "v3_results.json"), "w") as f:
        json.dump(ra.to_py(R), f, allow_nan=False, indent=1)
    log(f"done in {time.time() - t_start:.0f} s -> results/v3/v3_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    main(quick=a.quick, procs=a.procs)
