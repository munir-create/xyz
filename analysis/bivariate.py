"""
Bivariate (two-equation) model of the paired 7- and 28-day strengths with
age-specific terms.

    ln f_i7  = x_i7'  b7  + e_i7
    ln f_i28 = x_i28' b28 + e_i28,      (e_i7, e_i28) ~ N(0, Sigma),  Sigma unstructured

Each curing age has its own term set (selected per age).  Both equations are
estimated together by REML/GLS with PairedLMM and a block design, so

* the 7/28-day pair of a mixture is treated as correlated (no pseudo-replication),
* cross-age quantities (strength gain f28/f7, change of an effect from 7 to 28 d)
  get correct standard errors,
* one fitted model predicts both ages.

With identical terms at both ages the GLS estimates equal per-age OLS; with
different terms GLS borrows information through the residual correlation
(seemingly unrelated regressions, Zellner 1962).  The shared-effects model of
run_analysis.py (level + age x level terms) is the special case in which an
effect has the same coefficient at both ages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import design as dz
import model_tools as mt
import selection as sel
from lmm import PairedLMM

AGES = (7, 28)


def sort_terms(terms):
    return sorted(set(terms), key=dz.LEVEL_TERMS.index)


def design(t7, t28, df):
    """Block design (n, 2, p7 + p28): 7-d columns are zero on the 28-d row and vice versa."""
    X7, n7 = dz.mix_design(sort_terms(t7), df)
    X28, n28 = dz.mix_design(sort_terms(t28), df)
    n, p7 = len(df), X7.shape[1]
    X = np.zeros((n, 2, p7 + X28.shape[1]))
    X[:, 0, :p7] = X7
    X[:, 1, p7:] = X28
    return X, [f"{c}@7" for c in n7] + [f"{c}@28" for c in n28]


def fit(d, t7, t28, y=None, struct="UN", hessian=True, method="REML"):
    y = d[["l7", "l28"]].values if y is None else y
    X, names = design(t7, t28, d)
    return PairedLMM(y, X, names, struct, method=method, hessian=hessian), X, names


def split_name(col):
    base, age = col.rsplit("@", 1)
    return base, int(age)


def term_L(names, term, age):
    """Rows selecting the coefficient(s) of `term` in the equation for `age`."""
    idx = [j for j, c in enumerate(names)
           if split_name(c)[1] == age and dz.term_of_column(split_name(c)[0]) == term]
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0
    return L


def rows(t7, t28, pts):
    """Design rows for new points (RCF_pct, SS_pct, AB, carbonation, age)."""
    df = dz.new_points(pts)
    X, _ = design(t7, t28, df)
    a = np.asarray(pts["age"]).astype(int)
    return X[np.arange(len(df)), np.where(a == 28, 1, 0), :]


def gain_rows(t7, t28, pts):
    """Rows l with l'beta = ln(f28 / f7) at each point."""
    p7, p28 = pts.copy(), pts.copy()
    p7["age"], p28["age"] = 7, 28
    return rows(t7, t28, p28) - rows(t7, t28, p7)


def predict(m, t7, t28, pts, level=0.95):
    """Median strength (MPa) with confidence and prediction intervals for one new batch."""
    X = rows(t7, t28, pts)
    fit_ = X @ m.beta
    se = np.sqrt(np.einsum("ip,pq,iq->i", X, m.cov_beta, X))
    a = np.where(np.asarray(pts["age"]).astype(int) == 28, 1, 0)
    s2 = np.array([m.Sigma[0, 0], m.Sigma[1, 1]])[a]
    df = np.array([m.satterthwaite_df(x) for x in X])
    q = stats.t.ppf(0.5 + level / 2, df)
    sp = np.sqrt(se ** 2 + s2)
    return pd.DataFrame({"ln_fit": fit_, "se": se, "df": df, "median": np.exp(fit_),
                         "ci_lo": np.exp(fit_ - q * se), "ci_hi": np.exp(fit_ + q * se),
                         "pi_lo": np.exp(fit_ - q * sp), "pi_hi": np.exp(fit_ + q * sp)})


def ratio(m, L, level=0.95):
    """exp(L beta) with CI and p (H0: ratio = 1) for each row of L.  A row that is
    identically zero (the factor is not in that age's equation) gives ratio 1, p NaN."""
    L = np.atleast_2d(L)
    out = []
    for l in L:
        if np.allclose(l, 0):
            out.append({"est": 0.0, "se": 0.0, "df": np.nan, "lo": 0.0, "hi": 0.0, "p": np.nan})
        else:
            out.append(mt.lincomb(m, l, level).iloc[0].to_dict())
    out = pd.DataFrame(out)
    return pd.DataFrame({"ratio": np.exp(out.est), "lo": np.exp(out.lo), "hi": np.exp(out.hi),
                         "est_ln": out.est, "se_ln": out.se, "df": out.df, "p": out.p})


def actual_equation(m, t7, t28, key):
    """Actual-unit coefficients (with SE) of ln f at 7 d, at 28 d, or of the gain
    ln(f28/f7).  Exact: the coded model is a polynomial in the actual variables."""
    pts = mt._grid_points()
    Xa = mt._actual_design(pts)
    if key == "gain":
        Xc = gain_rows(t7, t28, pts)
    else:
        p = pts.copy()
        p["age"] = 7 if key == "7 d" else 28
        Xc = rows(t7, t28, p)
    T = np.linalg.lstsq(Xa, Xc, rcond=None)[0]
    coef = T @ m.beta
    cov = T @ m.cov_beta @ T.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    assert np.allclose(Xa @ coef, Xc @ m.beta, atol=1e-8)
    keep = np.abs(coef) > 1e-12 * max(1, np.abs(coef).max())
    return [{"term": mt.ACTUAL_COLS[j], "coef": float(coef[j]), "se": float(se[j])}
            for j in range(len(coef)) if keep[j]]


# ------------------------------------------------------------------ per-age selection
class AgeSelector:
    """Exhaustive AICc over all hierarchical single-age models (716 for the full
    candidate set), with the hat matrices precomputed so that a new response vector
    on the same mixtures is ranked with one batched product (used in bootstraps)."""

    def __init__(self, d, subsets=None):
        self.n = len(d)
        cols = sel.mix_columns(d)
        self.subsets, H, k = [], [], []
        for s in (subsets or sel.hierarchical_level_subsets()):
            X = sel.mix_X(s, cols, self.n)
            p = X.shape[1]
            if self.n - (p + 1) - 1 <= 0 or np.linalg.matrix_rank(X) < p:
                continue
            Q, _ = np.linalg.qr(X)
            self.subsets.append(s)
            H.append(Q @ Q.T)
            k.append(p + 1)
        self.R = np.eye(self.n)[None] - np.array(H)           # residual makers (m, n, n)
        self.k = np.array(k, float)
        n = self.n
        self.pen = 2 * self.k + 2 * self.k * (self.k + 1) / (n - self.k - 1) + n * (1 + np.log(2 * np.pi))

    def aicc(self, y):
        r = self.R @ np.asarray(y, float)
        rss = np.einsum("mi,mi->m", r, r)
        return self.n * np.log(rss / self.n) + self.pen

    def best(self, y):
        return self.subsets[int(np.argmin(self.aicc(y)))]

    def ranked(self, y, top=None):
        a = self.aicc(y)
        o = np.argsort(a)
        o = o if top is None else o[:top]
        return [(float(a[i]), self.subsets[i], int(self.k[i]) - 1) for i in o]


def akaike_term_weights(ranked):
    a = np.array([r[0] for r in ranked])
    w = np.exp(-(a - a.min()) / 2)
    w /= w.sum()
    return {t: float(sum(wi for wi, r in zip(w, ranked) if t in r[1])) for t in dz.LEVEL_TERMS}
