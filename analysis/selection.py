"""
Model-term selection for the paired 7/28-day model.

* backward_eliminate : backward elimination under strong hierarchy, dropping at each
  step the removable term with the largest Wald-F p-value (Satterthwaite df) from the
  joint REML fit, until every removable term has p <= alpha.
* fast_backward      : the same procedure computed on the exact level/gain
  factorisation (compound-symmetry covariance), used inside the bootstraps.
* enumerate_cs       : exhaustive AICc search over all hierarchical models under
  compound symmetry, where the joint likelihood factorises exactly into an OLS
  likelihood for the mixture level M = (ln f7 + ln f28)/2 and one for the gain
  G = ln f28 - ln f7.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

import design as dz
from lmm import PairedLMM


# ------------------------------------------------------------------ joint LMM
def term_L(term, names):
    idx = [j for j, c in enumerate(names) if dz.term_of_column(c) == term]
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0
    return L


def backward_eliminate(d, start_terms, y, struct="UN", alpha=0.10, verbose=False, protect=()):
    terms = list(start_terms)
    history = []
    while True:
        X, names = dz.design(terms, d)
        m = PairedLMM(y, X, names, struct)
        cand = [t for t in dz.removable(terms) if t not in protect]
        if not cand:
            break
        tests = {t: m.f_test(term_L(t, names)) for t in cand}
        worst = max(cand, key=lambda t: tests[t][3])
        F, q, df, p = tests[worst]
        if p <= alpha:
            break
        history.append({"removed": worst, "F": F, "df1": q, "df2": df, "p": p})
        if verbose:
            print(f"  drop {worst:10s} F={F:6.3f} ({q},{df:5.1f}) p={p:.4f}")
        terms.remove(worst)
    return terms, history


# ------------------------------------------------------------------ OLS on M / G
def _ols(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    n, p = X.shape
    rss = r @ r
    return b, XtX_inv, r, rss, n - p


def ols_ll(y, X):
    n = len(y)
    b, _, r, rss, _ = _ols(y, X)
    return -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)


def mix_columns(d):
    """Precompute every level-term column block (n x k) once."""
    return {t: np.column_stack(list(dz.term_columns(t, d).values())) for t in dz.LEVEL_TERMS}


def mix_X(terms, cols, n):
    blocks = [np.ones((n, 1))] + [cols[t] for t in dz.LEVEL_TERMS if t in terms]
    return np.hstack(blocks)


def _ftest_drop(y, X_full, drop_idx):
    b, XtX_inv, r, rss, dfe = _ols(y, X_full)
    keep = [j for j in range(X_full.shape[1]) if j not in drop_idx]
    _, _, _, rss0, _ = _ols(y, X_full[:, keep])
    q = len(drop_idx)
    if dfe <= 0 or rss <= 0:
        return np.nan, 1.0
    F = ((rss0 - rss) / q) / (rss / dfe)
    return F, stats.f.sf(F, q, dfe)


def fast_backward(M, G, cols, start_terms, alpha=0.10, protect=()):
    """Backward elimination on the level/gain factorisation (exact under CS)."""
    n = len(M)
    terms = list(start_terms)
    while True:
        lev, gain = dz.split_level_gain(terms)
        best = None
        for t in dz.removable(terms):
            if t in protect:
                continue
            if t.endswith(".Age"):
                base = t[:-4]
                Xf = mix_X(gain, cols, n)
                start = 1 + sum(cols[u].shape[1] for u in dz.LEVEL_TERMS if u in gain and dz.LEVEL_TERMS.index(u) < dz.LEVEL_TERMS.index(base))
                idx = list(range(start, start + cols[base].shape[1]))
                F, p = _ftest_drop(G, Xf, idx)
            else:
                Xf = mix_X(lev, cols, n)
                start = 1 + sum(cols[u].shape[1] for u in dz.LEVEL_TERMS if u in lev and dz.LEVEL_TERMS.index(u) < dz.LEVEL_TERMS.index(t))
                idx = list(range(start, start + cols[t].shape[1]))
                F, p = _ftest_drop(M, Xf, idx)
            if not np.isfinite(p):
                p = 1.0
            if best is None or p > best[1]:
                best = (t, p)
        if best is None or best[1] <= alpha:
            return terms
        terms.remove(best[0])


# ------------------------------------------------------------------ exhaustive CS search
def hierarchical_level_subsets():
    """All hierarchical subsets of the level terms."""
    mains = ["A", "C", "D", "Carb"]
    higher = [t for t in dz.LEVEL_TERMS if t not in mains]
    out = []
    for r in range(len(mains) + 1):
        for ms in itertools.combinations(mains, r):
            ms = set(ms)
            allowed = [t for t in higher if set(dz.PARENTS[t]) <= ms]
            for k in range(len(allowed) + 1):
                for hs in itertools.combinations(allowed, k):
                    out.append(frozenset(ms | set(hs)))
    return out


def enumerate_cs(M, G, cols, n_top=25):
    """Exhaustive AICc (ML, CS covariance) over all hierarchical joint models.
    A gain term T.Age requires T in the level part and the gain parts of T's parents."""
    n = len(M)
    subsets = hierarchical_level_subsets()
    llM, kM, llG, kG = {}, {}, {}, {}
    for s in subsets:
        X = mix_X(s, cols, n)
        llM[s] = ols_ll(M, X); kM[s] = X.shape[1]
        llG[s] = ols_ll(G, X); kG[s] = X.shape[1]
    N = 2 * n
    res = []
    sub_list = list(subsets)
    for g in sub_list:
        for m_ in sub_list:
            if not g <= m_:
                continue
            k = kM[m_] + kG[g] + 2
            if N - k - 1 <= 0:
                continue
            ll = llM[m_] + llG[g]
            aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / (N - k - 1)
            res.append((aicc, m_, g, k))
    res.sort(key=lambda r: r[0])
    return res[:n_top], len(res)


def enumerate_models(M, G, cols, struct="UN", n_top=25, subsets=None):
    """Exhaustive AICc (ML) over all hierarchical joint models.

    struct='CS': the likelihood factorises into OLS(M | X_M) and OLS(G | X_G).
    struct='UN': because hierarchy makes the gain terms a subset of the level terms,
    the unstructured-covariance likelihood factorises exactly into OLS(G | X_G) and
    OLS(M | X_M, G) (conditional regression of the level on the gain), with
    p_M + p_G + 3 parameters.  Returns the n_top best models and the model count."""
    n = len(M)
    subsets = subsets or hierarchical_level_subsets()
    llM, kM, llG, kG = {}, {}, {}, {}
    for s in subsets:
        X = mix_X(s, cols, n)
        llG[s] = ols_ll(G, X); kG[s] = X.shape[1]
        if struct == "UN":
            llM[s] = ols_ll(M, np.column_stack([X, G]))
        else:
            llM[s] = ols_ll(M, X)
        kM[s] = X.shape[1]
    extra = 3 if struct == "UN" else 2
    N = 2 * n
    res = []
    for g in subsets:
        for m_ in subsets:
            if not g <= m_:
                continue
            k = kM[m_] + kG[g] + extra
            if N - k - 1 <= 0 or (struct == "UN" and n - kM[m_] - 1 <= 0):
                continue
            ll = llM[m_] + llG[g]
            aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / (N - k - 1)
            res.append((aicc, m_, g, k))
    res.sort(key=lambda r: r[0])
    return (res if n_top is None else res[:n_top]), len(res)


def fast_backward_un(M, G, cols, start_terms, alpha=0.10, protect=()):
    """Backward elimination with exact finite-sample F tests under the unstructured
    covariance: gain terms are tested in OLS(G | X_G); a removable level term (its
    age interaction absent) is tested in the conditional regression OLS(M | X_M, G)."""
    n = len(M)
    terms = list(start_terms)
    while True:
        lev, gain = dz.split_level_gain(terms)
        best = None
        for t in dz.removable(terms):
            if t in protect:
                continue
            if t.endswith(".Age"):
                base, y, sset, extra = t[:-4], G, gain, None
            else:
                base, y, sset, extra = t, M, lev, G
            Xf = mix_X(sset, cols, n)
            start = 1 + sum(cols[u].shape[1] for u in dz.LEVEL_TERMS
                            if u in sset and dz.LEVEL_TERMS.index(u) < dz.LEVEL_TERMS.index(base))
            idx = list(range(start, start + cols[base].shape[1]))
            if extra is not None:
                Xf = np.column_stack([Xf, extra])
            F, p = _ftest_drop(y, Xf, idx)
            if not np.isfinite(p):
                p = 1.0
            if best is None or p > best[1]:
                best = (t, p)
        if best is None or best[1] <= alpha:
            return terms
        terms.remove(best[0])


def loglik_ml(Z, lev, gain, cols, struct="UN"):
    """Exact ML log-likelihood of a joint model for a paired response Z (n x 2)."""
    n = len(Z)
    M = (Z[:, 0] + Z[:, 1]) / 2
    G = Z[:, 1] - Z[:, 0]
    XM, XG = mix_X(lev, cols, n), mix_X(gain, cols, n)
    if struct == "UN":
        return ols_ll(G, XG) + ols_ll(M, np.column_stack([XM, G])), XM.shape[1] + XG.shape[1] + 3
    return ols_ll(G, XG) + ols_ll(M, XM), XM.shape[1] + XG.shape[1] + 2


def joint_terms(level_set, gain_set):
    return sorted(level_set, key=dz.LEVEL_TERMS.index) + ["Age"] + \
        [t + ".Age" for t in sorted(gain_set, key=dz.LEVEL_TERMS.index)]
