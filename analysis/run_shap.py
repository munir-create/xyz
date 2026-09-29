"""
Exact SHAP (Shapley additive explanation) values for the finalised strength models.

Models explained
  7d      independent 7-day model   ln f7  ~ RCF + SS + SS^2 + Carb + RCF.SS + K.RCF + K.SS
  28d     independent 28-day model  ln f28 ~ RCF + RCF^2 + A/B + SS
  paired  paired 7/28-day mixed model (curing age is a model input)

Features = the actual model inputs: RCF (%), SS (%), A/B, Carbonation (one categorical feature
with levels NC, 0.5 h, 1 h, 5 h) and, for the paired model, curing age (7 / 28 d).
Carbonation is never treated as a number: when it is "absent" from a coalition, its level is
taken from a background mixture, so its level indicators and its K-interactions move together.

SHAP values are exact interventional Shapley values: every coalition of features is evaluated
(2^4 = 16, or 2^5 = 32 with age) and the absent features are averaged over the background set,
which is the 30 real mixtures (60 mixture x age rows for the paired model).  Shapley interaction
values follow Lundberg et al. (2020).  Explanations are computed on the ln(strength) scale (the
scale the models were fitted on, where the contributions of additive terms are exact and the
interaction values correspond to the model's interaction terms) and on the MPa scale (median
prediction exp(ln f)).  Values are cross-checked against shap.ExactExplainer.

Uncertainty of the mean |SHAP| importances: 1,000 case-bootstrap refits stratified by
carbonation level (mixtures resampled with both ages kept together for the paired model).

Run:  python analysis/run_shap.py
"""
from __future__ import annotations

import json
import os
import time
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

import design as dz
import model_tools as mt
from lmm import PairedLMM

OUT = os.path.join(dz.ROOT, "results", "shap")
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
LEVELS = dz.CARB_LEVELS
FEATS = ["RCF", "SS", "A/B", "Carbonation"]
FEATS_P = FEATS + ["Curing age"]
N_BOOT = 1000
SEED = 20260929


# ------------------------------------------------------------------ models as functions of the raw inputs
def to_frame(X, with_age=False):
    df = pd.DataFrame({"RCF_pct": X[:, 0], "SS_pct": X[:, 1], "AB": X[:, 2],
                       "carbonation": [LEVELS[int(round(c))] for c in X[:, 3]]})
    if with_age:
        df["age"] = X[:, 4].astype(int)
    return df


class OLSModel:
    def __init__(self, A):
        e = A["model_export"]
        self.terms = [t for t in dz.LEVEL_TERMS if t in A["final_terms"]]
        self.names = e["names"]
        self.beta = np.array(e["beta"])
        self.with_age = False

    def design(self, X):
        df = dz.new_points(to_frame(X))
        D, names = dz.mix_design(self.terms, df)
        assert names == self.names
        return D

    def ln(self, X, beta=None):
        return self.design(X) @ (self.beta if beta is None else beta)


class PairedModel:
    def __init__(self, R):
        self.terms = R["final_terms"]
        self.names = R["final"]["names"]
        self.beta = np.array(R["final"]["beta"])
        self.with_age = True

    def design(self, X):
        return mt.rows(self.terms, to_frame(X, True))

    def ln(self, X, beta=None):
        return self.design(X) @ (self.beta if beta is None else beta)


# ------------------------------------------------------------------ exact Shapley machinery
def coalition_designs(model, X, B):
    """Average design row over the background for every instance and coalition:
    Z[i, s, :] = mean_b design(hybrid(x_i, b, s)).  Also returns the full hybrid design
    (needed for the non-linear MPa scale)."""
    n, M = X.shape
    nb = len(B)
    S = np.array([[(s >> j) & 1 for j in range(M)] for s in range(2 ** M)], dtype=bool)  # (2^M, M)
    H = np.where(S[None, :, None, :], X[:, None, None, :], B[None, None, :, :])       # (n, 2^M, nb, M)
    Dh = model.design(H.reshape(-1, M)).reshape(n, 2 ** M, nb, -1)
    return S, Dh


def shapley_from_v(v, S):
    """v: (n, 2^M) coalition values.  Returns phi (n, M), base (n,) and interaction values (n, M, M)."""
    n, K = v.shape
    M = S.shape[1]
    idx = {tuple(row): k for k, row in enumerate(S)}
    size = S.sum(1)
    phi = np.zeros((n, M))
    for i in range(M):
        for k in range(K):
            if S[k, i]:
                continue
            s = size[k]
            w = factorial(s) * factorial(M - s - 1) / factorial(M)
            kk = tuple(np.where(np.arange(M) == i, True, S[k]))
            phi[:, i] += w * (v[:, idx[kk]] - v[:, k])
    inter = np.zeros((n, M, M))
    for i, j in combinations(range(M), 2):
        acc = np.zeros(n)
        for k in range(K):
            if S[k, i] or S[k, j]:
                continue
            s = size[k]
            w = factorial(s) * factorial(M - s - 2) / (2 * factorial(M - 1))
            ki = tuple(np.where(np.arange(M) == i, True, S[k]))
            kj = tuple(np.where(np.arange(M) == j, True, S[k]))
            kij = tuple(np.where((np.arange(M) == i) | (np.arange(M) == j), True, S[k]))
            acc += w * (v[:, idx[kij]] - v[:, idx[ki]] - v[:, idx[kj]] + v[:, k])
        inter[:, i, j] = inter[:, j, i] = acc
    for i in range(M):
        inter[:, i, i] = phi[:, i] - inter[:, i, :].sum(1) + inter[:, i, i]
    base = v[:, 0]
    return phi, base, inter


def explain(model, X, B, beta=None, scales=("ln", "MPa"), Dh=None, S=None, check=True):
    if Dh is None:
        S, Dh = coalition_designs(model, X, B)
    b = model.beta if beta is None else beta
    lnh = Dh @ b                                    # (n, 2^M, nb)
    out = {}
    for sc in scales:
        v = lnh.mean(2) if sc == "ln" else np.exp(lnh).mean(2)
        phi, base, inter = shapley_from_v(v, S)
        fx = model.ln(X, b) if sc == "ln" else np.exp(model.ln(X, b))
        err = max(np.max(np.abs(phi.sum(1) + base - fx) / np.maximum(np.abs(fx), 1)),
                  np.max(np.abs(inter.sum((1, 2)) + base - fx) / np.maximum(np.abs(fx), 1)))
        if check and not err < 1e-9:
            raise AssertionError(f"efficiency violated ({sc}): {err}")
        out[sc] = {"phi": phi, "base": float(base[0]), "inter": inter, "fx": fx}
    return out, S, Dh


# ------------------------------------------------------------------ inputs
def inputs(d, with_age=False):
    X = np.column_stack([d.RCF_pct, d.SS_pct, d.AB, d.carbonation.map({l: i for i, l in enumerate(LEVELS)}).astype(float)])
    if not with_age:
        return X
    X7 = np.column_stack([X, np.full(len(X), 7.0)])
    X28 = np.column_stack([X, np.full(len(X), 28.0)])
    return np.vstack([X7, X28])


def crosscheck(model, X, B, mine):
    """Compare with shap.ExactExplainer (values and interaction values, ln scale)."""
    try:
        import shap
    except ImportError:
        return {"available": False}
    f = lambda Z: model.ln(np.asarray(Z, float))
    masker = shap.maskers.Independent(B, max_samples=len(B))
    ex = shap.explainers.Exact(f, masker)
    sv = ex(X, silent=True)
    iv = ex(X, interactions=2, silent=True)
    d1 = float(np.max(np.abs(sv.values - mine["phi"])))
    d0 = float(np.max(np.abs(np.asarray(sv.base_values) - mine["base"])))
    d2 = float(np.max(np.abs(iv.values - mine["inter"])))
    return {"available": True, "shap_version": shap.__version__, "max_abs_diff_values": d1, "max_abs_diff_base": d0,
            "max_abs_diff_interactions": d2}


def analytic_check_28(model, X, B, phi):
    """The 28-day model is additive on the ln scale, so each SHAP value is closed form:
    phi_RCF = bA (A - mean A) + bAA (A^2 - mean A^2), phi_AB = bC (C - mean C), phi_SS = bD (D - mean D)."""
    b = dict(zip(model.names, model.beta))
    A, C, D = (X[:, 0] - 30) / 20, (X[:, 2] - 0.45) / 0.03, (X[:, 1] - 37.5) / 37.5
    Ab, Cb, Db = (B[:, 0] - 30) / 20, (B[:, 2] - 0.45) / 0.03, (B[:, 1] - 37.5) / 37.5
    ref = np.column_stack([b["A"] * (A - Ab.mean()) + b["A2"] * (A ** 2 - (Ab ** 2).mean()), b["D"] * (D - Db.mean()),
                           b["C"] * (C - Cb.mean()), np.zeros(len(X))])
    return float(np.max(np.abs(ref - phi)))


# ------------------------------------------------------------------ bootstrap
def boot_betas(key, d, R_sep, R_pair, rng):
    strata = [np.where(d.carbonation.values == lv)[0] for lv in LEVELS]
    out = []
    if key in ("7d", "28d"):
        age = key[:-1]
        A = R_sep[age]
        terms = [t for t in dz.LEVEL_TERMS if t in A["final_terms"]]
        y = np.log(d[f"f{age}_mean"].values)
        while len(out) < N_BOOT:
            idx = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in strata])
            Xd, _ = dz.mix_design(terms, d.iloc[idx].reset_index(drop=True))
            if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
                continue
            out.append(np.linalg.lstsq(Xd, y[idx], rcond=None)[0])
    else:
        terms = R_pair["final_terms"]
        while len(out) < N_BOOT:
            idx = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in strata])
            db = d.iloc[idx].reset_index(drop=True)
            Xf, names = dz.design(terms, db)
            if np.linalg.matrix_rank(Xf.reshape(-1, Xf.shape[2])) < Xf.shape[2]:
                continue
            m = PairedLMM(db[["l7", "l28"]].values, Xf, names, R_pair["final_struct"], hessian=False)
            out.append(m.beta)
    return np.array(out)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    d = dz.load()
    with open(os.path.join(dz.ROOT, "results", "separate", "separate_results.json")) as f:
        R_sep = json.load(f)
    with open(os.path.join(dz.ROOT, "results", "model_results.json")) as f:
        R_pair = json.load(f)
    models = {"7d": OLSModel(R_sep["7"]), "28d": OLSModel(R_sep["28"]), "paired": PairedModel(R_pair)}
    res = {"meta": {"date": time.strftime("%Y-%m-%d"), "n_boot": N_BOOT, "seed": SEED, "features": FEATS, "features_paired": FEATS_P,
                    "levels": LEVELS, "background": "the 30 mixtures (60 mixture x age rows for the paired model)"},
           "mixes": d[["mix", "carbonation", "RCF_pct", "SS_pct", "AB", "f7_mean", "f28_mean"]].to_dict(orient="records")}
    rng = np.random.default_rng(SEED)
    for key, model in models.items():
        print(f"--- {key}")
        X = inputs(d, model.with_age)
        B = X.copy()
        ex, S, Dh = explain(model, X, B)
        cc = crosscheck(model, X, B, ex["ln"])
        print("  cross-check vs shap.ExactExplainer:", cc)
        entry = {"X": X, "crosscheck": cc, "features": FEATS_P if model.with_age else FEATS}
        if key == "28d":
            entry["analytic_check_max_diff"] = analytic_check_28(model, X, B, ex["ln"]["phi"])
            print("  analytic check (28 d additive):", entry["analytic_check_max_diff"])
        for sc in ("ln", "MPa"):
            e = ex[sc]
            entry[sc] = {"phi": e["phi"], "base": e["base"], "inter": e["inter"], "fx": e["fx"],
                         "mean_abs": np.abs(e["phi"]).mean(0), "mean_abs_inter": np.abs(e["inter"]).mean(0)}
        # bootstrap uncertainty of the importances (instances and background fixed, model refitted)
        betas = boot_betas(key, d, R_sep, R_pair, rng)
        imp = {"ln": [], "MPa": [], "ln_inter": []}
        n_bad = 0
        for b in betas:
            try:
                eb, _, _ = explain(model, X, B, beta=b, Dh=Dh, S=S)
            except AssertionError as e:
                n_bad += 1
                print("  skipped bootstrap draw:", e)
                continue
            imp["ln"].append(np.abs(eb["ln"]["phi"]).mean(0))
            imp["MPa"].append(np.abs(eb["MPa"]["phi"]).mean(0))
            imp["ln_inter"].append(np.abs(eb["ln"]["inter"]).mean(0))
        for sc in ("ln", "MPa"):
            a = np.array(imp[sc])
            entry[sc]["mean_abs_ci"] = np.percentile(a, [2.5, 97.5], axis=0)
            entry[sc]["rank_first_share"] = np.bincount(np.argmax(a, 1), minlength=a.shape[1]) / len(a)
        a = np.array(imp["ln_inter"])
        entry["ln"]["mean_abs_inter_ci"] = np.percentile(a, [2.5, 97.5], axis=0)
        # grid dependence curves (ln scale): SHAP of one feature at grid values, other features at every instance
        entry["n_boot"] = len(betas) - n_bad
        entry["n_boot_skipped"] = n_bad
        res[key] = entry
        # tables
        feats = entry["features"]
        rows = []
        for i in range(len(X)):
            r = {"mix": int(d.mix.values[i % 30])}
            if model.with_age:
                r["age_d"] = int(X[i, 4])
            r.update({"RCF": X[i, 0], "SS": X[i, 1], "AB": X[i, 2], "carbonation": LEVELS[int(X[i, 3])]})
            for sc in ("ln", "MPa"):
                r[f"prediction_{sc}"] = entry[sc]["fx"][i]
                r[f"base_{sc}"] = entry[sc]["base"]
                for j, fn in enumerate(feats):
                    r[f"SHAP_{sc}_{fn}"] = entry[sc]["phi"][i, j]
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(TAB, f"shap_values_{key}.csv"), index=False)
        imp_rows = []
        for j, fn in enumerate(feats):
            imp_rows.append({"feature": fn,
                             "mean_abs_SHAP_ln": entry["ln"]["mean_abs"][j], "ci_lo_ln": entry["ln"]["mean_abs_ci"][0][j],
                             "ci_hi_ln": entry["ln"]["mean_abs_ci"][1][j], "share_ranked_first_ln": entry["ln"]["rank_first_share"][j],
                             "mean_abs_SHAP_MPa": entry["MPa"]["mean_abs"][j], "ci_lo_MPa": entry["MPa"]["mean_abs_ci"][0][j],
                             "ci_hi_MPa": entry["MPa"]["mean_abs_ci"][1][j]})
        pd.DataFrame(imp_rows).to_csv(os.path.join(TAB, f"shap_importance_{key}.csv"), index=False)
        pd.DataFrame(entry["ln"]["mean_abs_inter"], index=feats, columns=feats).to_csv(os.path.join(TAB, f"shap_interaction_matrix_ln_{key}.csv"))

    def py(o):
        if isinstance(o, dict):
            return {k: py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [py(v) for v in o]
        if isinstance(o, np.ndarray):
            return py(o.tolist())
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    res["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "shap_results.json"), "w") as f:
        json.dump(py(res), f)
    print(f"done in {res['runtime_s']:.0f} s -> results/shap/shap_results.json")


if __name__ == "__main__":
    main()
