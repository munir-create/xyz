"""
Prediction, contrast and actual-unit utilities for a fitted PairedLMM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import design as dz

AGES = {7: 0, 28: 1}


def rows(terms, pts):
    """Design rows for new points.  pts: DataFrame with RCF_pct, SS_pct, AB,
    carbonation and age (7 or 28).  Returns (m, p)."""
    df = dz.new_points(pts)
    X, _ = dz.design(terms, df)
    a = np.asarray(pts["age"]).astype(int)
    return X[np.arange(len(df)), np.where(a == 28, 1, 0), :]


def gain_rows(terms, pts):
    """Rows l such that l'beta = ln(f28/f7) at each point (age column ignored)."""
    p7 = pts.copy(); p7["age"] = 7
    p28 = pts.copy(); p28["age"] = 28
    return rows(terms, p28) - rows(terms, p7)


def lincomb(model, L, level=0.95):
    """Estimates, SE, Satterthwaite df, CI and p for rows of L (m, p)."""
    L = np.atleast_2d(L)
    out = []
    for l in L:
        est, se, df, t, p = model.t_test(l)
        q = stats.t.ppf(0.5 + level / 2, df)
        out.append((est, se, df, est - q * se, est + q * se, p))
    return pd.DataFrame(out, columns=["est", "se", "df", "lo", "hi", "p"])


def lincomb_fast(model, L, df, level=0.95):
    """Same as lincomb but with a supplied df (vectorised, for grids)."""
    L = np.atleast_2d(L)
    est = L @ model.beta
    se = np.sqrt(np.einsum("ip,pq,iq->i", L, model.cov_beta, L))
    q = stats.t.ppf(0.5 + level / 2, df)
    return est, se, est - q * se, est + q * se


def predict_points(model, terms, pts, level=0.95, df=None):
    """Median predictions (MPa) with confidence and prediction intervals for one
    new batch.  Prediction variance adds the age-specific residual variance."""
    X = rows(terms, pts)
    fit = X @ model.beta
    se = np.sqrt(np.einsum("ip,pq,iq->i", X, model.cov_beta, X))
    a = np.where(np.asarray(pts["age"]).astype(int) == 28, 1, 0)
    s2 = np.array([model.Sigma[0, 0], model.Sigma[1, 1]])[a]
    if df is None:
        df = np.array([model.satterthwaite_df(x) for x in X])
    q = stats.t.ppf(0.5 + level / 2, df)
    sp = np.sqrt(se ** 2 + s2)
    return pd.DataFrame({"ln_fit": fit, "se": se, "df": df,
                         "median": np.exp(fit), "ci_lo": np.exp(fit - q * se), "ci_hi": np.exp(fit + q * se),
                         "pi_lo": np.exp(fit - q * sp), "pi_hi": np.exp(fit + q * sp),
                         "mean": np.exp(fit + s2 / 2)})


def scaled_pred_var(model, terms, pts):
    """x' Cov(beta) x / sigma_age^2 : scaled prediction variance (leverage analogue)."""
    X = rows(terms, pts)
    a = np.where(np.asarray(pts["age"]).astype(int) == 28, 1, 0)
    s2 = np.array([model.Sigma[0, 0], model.Sigma[1, 1]])[a]
    return np.einsum("ip,pq,iq->i", X, model.cov_beta, X) / s2


# ------------------------------------------------------------------ actual units
ACTUAL_COLS = ["1", "RCF", "SS", "A/B", "RCF·SS", "RCF·A/B", "SS·A/B", "RCF²", "SS²", "(A/B)²",
               "C[0.5 h]", "C[1 h]", "C[5 h]", "K·RCF", "K·SS", "K·A/B"]


def _actual_design(pts):
    R, S, B = pts.RCF_pct.values, pts.SS_pct.values, pts.AB.values
    c = pts.carbonation.astype(str).values
    K = (c != "NC").astype(float)
    cols = [np.ones(len(pts)), R, S, B, R * S, R * B, S * B, R ** 2, S ** 2, B ** 2,
            (c == "0.5 h") * 1.0, (c == "1 h") * 1.0, (c == "5 h") * 1.0, K * R, K * S, K * B]
    return np.column_stack(cols)


def _grid_points():
    g = []
    for c in dz.CARB_LEVELS:
        for R in (10, 23, 37, 50):
            for S in (0, 20, 45, 75):
                for B in (0.42, 0.45, 0.48):
                    g.append((R, S, B, c))
    return pd.DataFrame(g, columns=["RCF_pct", "SS_pct", "AB", "carbonation"])


def actual_equations(model, terms):
    """Actual-unit coefficients (with SE) of ln f at 7 d, at 28 d and of the gain
    ln(f28/f7).  Exact: the coded model is a polynomial in the actual variables, so
    the map coded -> actual is solved exactly on a design grid."""
    pts = _grid_points()
    Xa = _actual_design(pts)
    out = {}
    for key in ("7 d", "28 d", "gain"):
        if key == "gain":
            Xc = gain_rows(terms, pts)
        else:
            p = pts.copy(); p["age"] = 7 if key == "7 d" else 28
            Xc = rows(terms, p)
        # T maps coded beta to actual coefficients: Xa T = Xc
        T, res, rk, _ = np.linalg.lstsq(Xa, Xc, rcond=None)
        coef = T @ model.beta
        cov = T @ model.cov_beta @ T.T
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        keep = np.abs(coef) > 1e-12 * max(1, np.abs(coef).max())
        # verify exactness
        assert np.allclose(Xa @ coef, Xc @ model.beta, atol=1e-8)
        out[key] = [{"term": ACTUAL_COLS[j], "coef": float(coef[j]), "se": float(se[j])}
                    for j in range(len(coef)) if keep[j]]
    return out


def format_equation(coefs, digits=6, lhs="ln(f)"):
    parts = []
    for c in coefs:
        v = c["coef"]
        term = "" if c["term"] == "1" else "·" + c["term"]
        s = f"{abs(v):.{digits}g}{term}"
        if not parts:
            parts.append(("−" if v < 0 else "") + s)
        else:
            parts.append((" − " if v < 0 else " + ") + s)
    return f"{lhs} = " + "".join(parts)
