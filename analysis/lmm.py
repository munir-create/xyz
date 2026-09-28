"""
Bivariate (7-day, 28-day) linear mixed model for paired measurements on the
same mixtures.

Every mixture i contributes two observations y_i = (y_i7, y_i28)'.  The model is

    y_i = X_i beta + e_i,      e_i ~ N(0, Sigma),     independent over mixtures,

with X_i the 2 x p fixed-effects design rows of mixture i.  Sigma is the
within-mixture covariance and can take the structures

    'IND'  : s^2 I                          (ignores the pairing; reference only)
    'CS'   : s^2 [[1, r], [r, 1]]           (random mixture intercept, tau^2 = r s^2)
    'UN'   : [[s7^2, r s7 s28], [r s7 s28, s28^2]]
             (random intercept + age-specific residual variance; with two
              occasions this is the unstructured 2 x 2 covariance)

Estimation is by REML (variance parameters) and GLS (fixed effects); ML fits
are provided for likelihood comparisons of fixed-effects structures.
Inference uses Wald t / F statistics with Satterthwaite degrees of freedom,
computed from the numerically differentiated REML information, i.e. the same
approach as lmerTest in R.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

LOG2PI = np.log(2 * np.pi)


def _sigma(theta, struct):
    if struct == "IND":
        s2 = np.exp(2 * theta[0])
        return np.array([[s2, 0.0], [0.0, s2]])
    if struct == "CS":
        s2 = np.exp(2 * theta[0])
        r = np.tanh(theta[1])
        return s2 * np.array([[1.0, r], [r, 1.0]])
    if struct == "UN":
        s7, s28 = np.exp(theta[0]), np.exp(theta[1])
        r = np.tanh(theta[2])
        return np.array([[s7 ** 2, r * s7 * s28], [r * s7 * s28, s28 ** 2]])
    if struct == "INDH":
        s7, s28 = np.exp(theta[0]), np.exp(theta[1])
        return np.array([[s7 ** 2, 0.0], [0.0, s28 ** 2]])
    raise ValueError(struct)


N_THETA = {"IND": 1, "CS": 2, "UN": 3, "INDH": 2}


class PairedLMM:
    """y: (n, 2) array; X: (n, 2, p) array (row 0 = 7 d, row 1 = 28 d)."""

    def __init__(self, y, X, names, struct="UN", method="REML", extra_var=None, fit=True,
                 hessian=True):
        """extra_var: optional (n, 2) known variances added to the diagonal of Sigma
        for each mixture (used for the replicate-count sensitivity analysis)."""
        self.extra_var = None if extra_var is None else np.asarray(extra_var, float)
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.names = list(names)
        self.struct = struct
        self.method = method
        self.n, _, self.p = self.X.shape
        self.N = 2 * self.n
        self._want_hessian = hessian
        if fit:
            self._fit()

    # ---------------------------------------------------------------- core
    def _sigmas(self, Sigma):
        """per-mixture covariance matrices (n, 2, 2)"""
        S = np.broadcast_to(Sigma, (self.n, 2, 2)).copy()
        if self.extra_var is not None:
            S[:, 0, 0] += self.extra_var[:, 0]
            S[:, 1, 1] += self.extra_var[:, 1]
        return S

    def _gls(self, Sigma):
        Si = np.linalg.inv(self._sigmas(Sigma))
        XtVX = np.einsum("iap,iab,ibq->pq", self.X, Si, self.X)
        XtVy = np.einsum("iap,iab,ib->p", self.X, Si, self.y)
        XtVX_inv = np.linalg.inv(XtVX)
        beta = XtVX_inv @ XtVy
        r = self.y - np.einsum("iap,p->ia", self.X, beta)
        quad = np.einsum("ia,iab,ib->", r, Si, r)
        return beta, XtVX, XtVX_inv, r, quad

    def loglik(self, theta, method=None):
        method = method or self.method
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 30:
            return -np.inf
        Sigma = _sigma(theta, self.struct)
        sign, logdet = np.linalg.slogdet(self._sigmas(Sigma))
        if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
            return -np.inf
        logdet = logdet.sum() / self.n
        try:
            beta, XtVX, _, _, quad = self._gls(Sigma)
        except np.linalg.LinAlgError:
            return -np.inf
        if method == "ML":
            return -0.5 * (self.n * logdet + quad + self.N * LOG2PI)
        _, ld_xtvx = np.linalg.slogdet(XtVX)
        return -0.5 * (self.n * logdet + ld_xtvx + quad + (self.N - self.p) * LOG2PI)

    def _start(self):
        # per-age OLS residual covariance as start values
        b = np.linalg.lstsq(self.X.reshape(-1, self.p), self.y.reshape(-1), rcond=None)[0]
        r = self.y - np.einsum("iap,p->ia", self.X, b)
        S = r.T @ r / max(self.n - self.p / 2, 1)
        s7, s28 = np.sqrt(S[0, 0]), np.sqrt(S[1, 1])
        rho = np.clip(S[0, 1] / (s7 * s28), -0.9, 0.9)
        s = np.sqrt((S[0, 0] + S[1, 1]) / 2)
        return {"IND": [np.log(s)], "CS": [np.log(s), np.arctanh(rho)],
                "UN": [np.log(s7), np.log(s28), np.arctanh(rho)],
                "INDH": [np.log(s7), np.log(s28)]}[self.struct]

    def _fit(self):
        f = lambda t: -self.loglik(t)
        best = None
        starts = (self._start(),) if not self._want_hessian else (self._start(), np.zeros(N_THETA[self.struct]))
        for start in starts:
            res = optimize.minimize(f, np.asarray(start, float), method="BFGS",
                                    options={"gtol": 1e-10, "maxiter": 2000})
            res2 = optimize.minimize(f, res.x, method="Nelder-Mead",
                                     options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
            cand = res2 if res2.fun < res.fun else res
            if best is None or cand.fun < best.fun:
                best = cand
        self.theta = best.x
        self.ll = -best.fun
        self.Sigma = _sigma(self.theta, self.struct)
        self.beta, self.XtVX, self.cov_beta, self.resid, _ = self._gls(self.Sigma)
        self.se = np.sqrt(np.diag(self.cov_beta))
        k_theta = N_THETA[self.struct]
        # information about theta: numerical Hessian of -REML loglik
        self.A_theta = (np.linalg.pinv(self._hessian(lambda t: -self.loglik(t), self.theta))
                        if self._want_hessian else None)
        self.k = self.p + k_theta
        # ML log-likelihood at the ML optimum (for AIC comparisons of fixed effects)
        if self.method == "ML":
            self.ll_ml = self.ll
        else:
            self.ll_ml = None

    @staticmethod
    def _hessian(f, x, h=1e-4):
        x = np.asarray(x, float)
        k = len(x)
        H = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ej = np.zeros(k)
                ei[i] = h; ej[j] = h
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
        return H

    # ---------------------------------------------------------------- inference
    def _cov_beta_at(self, theta):
        Si = np.linalg.inv(self._sigmas(_sigma(theta, self.struct)))
        XtVX = np.einsum("iap,iab,ibq->pq", self.X, Si, self.X)
        return np.linalg.inv(XtVX)

    def _grad_phi(self, L, h=1e-5):
        """gradient of vec(L Phi L') wrt theta (numerical)."""
        k = len(self.theta)
        out = []
        for j in range(k):
            e = np.zeros(k); e[j] = h
            Pp = L @ self._cov_beta_at(self.theta + e) @ L.T
            Pm = L @ self._cov_beta_at(self.theta - e) @ L.T
            out.append((Pp - Pm) / (2 * h))
        return out

    def satterthwaite_df(self, l):
        l = np.atleast_2d(np.asarray(l, float))
        phi = (l @ self.cov_beta @ l.T).item()
        g = np.array([G.item() for G in self._grad_phi(l)])
        denom = g @ self.A_theta @ g
        if denom <= 0:
            return np.inf
        return 2 * phi ** 2 / denom

    def t_test(self, l):
        l = np.asarray(l, float)
        est = float(l @ self.beta)
        se = float(np.sqrt(l @ self.cov_beta @ l))
        df = self.satterthwaite_df(l)
        t = est / se
        p = 2 * stats.t.sf(abs(t), df)
        return est, se, df, t, p

    def f_test(self, L):
        """Wald F test of L beta = 0 with Satterthwaite-type denominator df
        (lmerTest method: eigen-decomposition of L Phi L')."""
        L = np.atleast_2d(np.asarray(L, float))
        q = L.shape[0]
        LPL = L @ self.cov_beta @ L.T
        Lb = L @ self.beta
        F = float(Lb @ np.linalg.solve(LPL, Lb)) / q
        if q == 1:
            df = self.satterthwaite_df(L[0])
        else:
            w, P = np.linalg.eigh(LPL)
            nus = np.array([self.satterthwaite_df(P[:, m] @ L) for m in range(q)])
            E = np.sum(nus[nus > 2] / (nus[nus > 2] - 2))
            df = 2 * E / (E - q) if E > q else np.nan
            if not np.isfinite(df):
                df = np.min(nus)
        p = stats.f.sf(F, q, df)
        return F, q, df, p

    # ---------------------------------------------------------------- summaries
    def aicc_ml(self):
        """AICc from the ML fit of the same model (fixed-effects comparisons)."""
        m = PairedLMM(self.y, self.X, self.names, self.struct, method="ML", extra_var=self.extra_var)
        k = m.k
        return -2 * m.ll + 2 * k + 2 * k * (k + 1) / (self.N - k - 1), m.ll, k

    def variance_components(self):
        S = self.Sigma
        out = {"var_7": S[0, 0], "var_28": S[1, 1], "cov": S[0, 1],
               "rho": S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])}
        out["sd_7"], out["sd_28"] = np.sqrt(S[0, 0]), np.sqrt(S[1, 1])
        # random-intercept interpretation (valid when cov >= 0)
        out["tau2_mix"] = max(S[0, 1], 0.0)
        return out

    def whitened_residuals(self):
        """Residuals decorrelated within mixture (Cholesky of Sigma_i): iid N(0,1)
        under the model.  Column 0 = 7 d scaled residual, column 1 = 28 d residual
        conditional on the 7 d residual."""
        S = self._sigmas(self.Sigma)
        Lc = np.linalg.cholesky(S)
        return np.linalg.solve(Lc, self.resid[..., None])[..., 0]

    def hat_diag(self):
        """Diagonal of the GLS hat matrix H = X (X'V^-1 X)^-1 X' V^-1, per observation."""
        Si = np.linalg.inv(self._sigmas(self.Sigma))
        XC = np.einsum("iap,pq->iaq", self.X, self.cov_beta)
        XSi = np.einsum("iaq,iab->ibq", self.X, Si)  # (X' V^-1) rows
        return np.einsum("iaq,iaq->ia", XC, XSi)

    def predict(self, Xnew):
        """Xnew: (m, p) rows. Returns fitted value and SE of the mean."""
        Xnew = np.atleast_2d(Xnew)
        fit = Xnew @ self.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", Xnew, self.cov_beta, Xnew))
        return fit, se
