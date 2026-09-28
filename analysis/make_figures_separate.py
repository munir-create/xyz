"""
Figures for the two independent models (7-day and 28-day), from results/separate/separate_results.json.
Both models are drawn with the same axes so their trends can be compared.

Run:  python analysis/make_figures_separate.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

import design as dz
from make_figures import C7, C28, CARB_COL, CARB_MK, INK, INK2, MUTED, GRID, AXIS, panel, save as _save

RES = os.path.join(dz.ROOT, "results", "separate")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
COL = {"7": C7, "28": C28}
MK = {"7": "o", "28": "s"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_O = LinearSegmentedColormap.from_list("orange", ["#fde3d6", "#f8bfa3", "#f39a70", "#eb6834", "#c9501f", "#9c3c15", "#6e290d"])


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


class Model:
    """Prediction wrapper around an exported OLS model."""

    def __init__(self, A):
        e = A["model_export"]
        self.terms = A["final_terms"]
        self.names = e["names"]
        self.beta = np.array(e["beta"])
        self.cov = np.array(e["cov"])
        self.s2 = e["s2"]
        self.df = e["df"]
        self.t = stats.t.ppf(0.975, self.df)

    def X(self, pts):
        df = dz.new_points(pts)
        X, names = dz.mix_design([t for t in dz.LEVEL_TERMS if t in self.terms], df)
        assert names == self.names
        return X

    def pred(self, pts):
        X = self.X(pts)
        f = X @ self.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", X, self.cov, X))
        return np.exp(f), np.exp(f - self.t * se), np.exp(f + self.t * se)

    def ratio(self, p1, p2):
        l = self.X(p1) - self.X(p2)
        e = l @ self.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", l, self.cov, l))
        return np.exp(e), np.exp(e - self.t * se), np.exp(e + self.t * se)


def load():
    with open(os.path.join(RES, "separate_results.json")) as f:
        R = json.load(f)
    d = dz.load()
    return R, d, {a: Model(R[a]) for a in ("7", "28")}


# ------------------------------------------------------------------ figures
def fig_selection(R):
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    ax = axs[0]
    for a in ("7", "28"):
        b = R[a]["boxcox_final"]
        lam, ll = np.array(b["curve"]["lambda"]), np.array(b["curve"]["loglik"])
        ax.plot(lam, ll, color=COL[a], lw=1.8, label=f"{a} d: λ̂ = {b['lambda']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]")
    ax.axhline(R["7"]["boxcox_final"]["cut"], color=AXIS, lw=0.9)
    ax.axvline(0, color=INK2, lw=0.8); ax.text(0.04, 0.35, "ln", fontsize=7, color=INK2)
    ax.axvline(1, color=MUTED, lw=0.8); ax.text(0.96, 0.35, "none", fontsize=7, color=MUTED, ha="right")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-8, 0.8)
    ax.set_xlabel("Box–Cox λ"); ax.set_ylabel("Profile log-likelihood (relative)")
    ax.set_title("Response scale (selected models)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=7)
    panel(ax, "a")
    for ax, a, letter in ((axs[1], "7", "b"), (axs[2], "28", "c")):
        A = R[a]
        terms = dz.LEVEL_TERMS
        imp = np.array([A["importance"][t] for t in terms])
        rb = np.array([A["bootstrap"]["resid"]["inclusion"][t] for t in terms])
        sb = np.array([A["bootstrap"]["subsample"]["inclusion"][t] for t in terms])
        y = np.arange(len(terms))[::-1]
        ax.barh(y + 0.27, imp, height=0.26, color=INK, label="Akaike weight (716 models)")
        ax.barh(y, rb, height=0.26, color=COL[a], label="Residual bootstrap")
        ax.barh(y - 0.27, sb, height=0.26, color=COL[a], alpha=0.45, label="Subsamples (24 of 30)")
        ax.set_yticks(y)
        ax.set_yticklabels([("● " if t in A["final_terms"] else "   ") + dz.label(t) for t in terms], fontsize=7)
        ax.set_xlim(0, 1.02); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of weight / re-selections including the term")
        ax.set_title(f"{a}-day model: term support (● = selected)", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=6.8, ncol=2)
        panel(ax, letter)
    fig.tight_layout()
    save(fig, "H01_scale_and_term_support")


def fig_coefficients(R):
    """Coded coefficients of each model; terms a model does not contain are shown hollow,
    at the value they take when added (with hierarchy parents) to that model."""
    cols = ["A", "C", "D", "Carb[0.5 h]", "Carb[1 h]", "Carb[5 h]", "AC", "AD", "CD", "A2", "C2", "D2", "KA", "KC", "KD"]
    lab = {"A": "RCF", "C": "A/B", "D": "SS", "Carb[0.5 h]": "Carb 0.5 h", "Carb[1 h]": "Carb 1 h", "Carb[5 h]": "Carb 5 h",
           "AC": "RCF·A/B", "AD": "RCF·SS", "CD": "A/B·SS", "A2": "RCF²", "C2": "(A/B)²", "D2": "SS²", "KA": "K·RCF", "KC": "K·A/B", "KD": "K·SS"}
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    y = np.arange(len(cols))[::-1]
    for a, off in (("7", 0.18), ("28", -0.18)):
        A = R[a]
        coef = pd.DataFrame(A["coef"]).set_index("column")
        add = pd.DataFrame(A["added_terms"])
        tq = stats.t.ppf(0.975, A["fit"]["df_resid"])
        for j, cname in enumerate(cols):
            if cname in coef.index:
                r = coef.loc[cname]
                ax.plot([r.lo, r.hi], [y[j] + off] * 2, color=COL[a], lw=1.9, solid_capstyle="round")
                ax.scatter(r.coef, y[j] + off, color=COL[a], marker=MK[a], s=30, zorder=3, edgecolor="white", lw=0.7)
            else:
                est = None
                for _, rr in add.iterrows():
                    if cname in rr.estimates:
                        est = rr.estimates[cname]; df2 = rr.df2
                        if dz.term_of_column(cname) == rr.term:
                            break
                if est is None:
                    continue
                q = stats.t.ppf(0.975, df2)
                ax.plot([est[0] - q * est[1], est[0] + q * est[1]], [y[j] + off] * 2, color=COL[a], lw=1.0, alpha=0.55)
                ax.scatter(est[0], y[j] + off, facecolor="white", edgecolor=COL[a], marker=MK[a], s=30, zorder=3, lw=1.2)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([lab[c] for c in cols], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Coefficient on ln(strength), coded units (−1…+1 factor range)")
    ax.set_title("Coefficients of the two independent models, 95 % CI", loc="left")
    h = [Line2D([], [], color=C7, marker="o", lw=1.9, ms=5, label="7-day model"),
         Line2D([], [], color=C28, marker="s", lw=1.9, ms=5, label="28-day model"),
         Line2D([], [], color=INK2, marker="o", mfc="white", lw=1.0, ms=5, label="not selected: value if added")]
    ax.legend(handles=h, loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "H02_coefficients_both_models")


def fig_curves_ss(R, d, M):
    ss = np.linspace(0, 75, 151)
    fig, axs = plt.subplots(4, 3, figsize=(7.4, 8.6), sharex=True, sharey=True)
    for i, carb in enumerate(dz.CARB_LEVELS):
        for j, rcf in enumerate((10, 30, 50)):
            ax = axs[i, j]
            for a in ("7", "28"):
                pts = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ss, "AB": 0.45, "carbonation": carb})
                f, lo, hi = M[a].pred(pts)
                ax.fill_between(ss, lo, hi, color=COL[a], alpha=0.14, lw=0)
                ax.plot(ss, f, color=COL[a], lw=1.8)
                k = int(np.argmax(f))
                if 0 < k < len(ss) - 1:
                    ax.scatter(ss[k], f[k], s=22, facecolor="white", edgecolor=COL[a], lw=1.3, zorder=4)
            near = d[(d.carbonation == carb) & (np.abs(d.RCF_pct - rcf) <= 10)]
            for a in ("7", "28"):
                ax.scatter(near.SS_pct, near[f"f{a}_mean"], s=16, color=COL[a], marker=MK[a], edgecolor="white", lw=0.6, zorder=3)
            if i == 0:
                ax.set_title(f"RCF {rcf} %")
            if j == 0:
                ax.set_ylabel(f"{carb}\nStrength (MPa)")
            if i == 3:
                ax.set_xlabel("SS (%)")
            ax.set_ylim(0, 45)
    h = [Line2D([], [], color=C7, lw=1.8, label="7-day model (95 % CI)"), Line2D([], [], color=C28, lw=1.8, label="28-day model (95 % CI)"),
         Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="runs within ±10 % RCF"),
         Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=5, label="interior peak")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "H03_strength_vs_SS_both_models")


def fig_curves_rcf(R, d, M):
    rc = np.linspace(10, 50, 81)
    fig, axs = plt.subplots(4, 3, figsize=(7.4, 8.6), sharex=True, sharey=True)
    for i, carb in enumerate(dz.CARB_LEVELS):
        for j, ssv in enumerate((0, 37.5, 75)):
            ax = axs[i, j]
            for a in ("7", "28"):
                pts = pd.DataFrame({"RCF_pct": rc, "SS_pct": ssv, "AB": 0.45, "carbonation": carb})
                f, lo, hi = M[a].pred(pts)
                ax.fill_between(rc, lo, hi, color=COL[a], alpha=0.14, lw=0)
                ax.plot(rc, f, color=COL[a], lw=1.8)
            near = d[(d.carbonation == carb) & (np.abs(d.SS_pct - ssv) <= 15)]
            for a in ("7", "28"):
                ax.scatter(near.RCF_pct, near[f"f{a}_mean"], s=16, color=COL[a], marker=MK[a], edgecolor="white", lw=0.6, zorder=3)
            if i == 0:
                ax.set_title(f"SS {ssv:g} %")
            if j == 0:
                ax.set_ylabel(f"{carb}\nStrength (MPa)")
            if i == 3:
                ax.set_xlabel("RCF (%)")
            ax.set_ylim(0, 45)
    h = [Line2D([], [], color=C7, lw=1.8, label="7-day model (95 % CI)"), Line2D([], [], color=C28, lw=1.8, label="28-day model (95 % CI)"),
         Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="runs within ±15 % SS")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "H04_strength_vs_RCF_both_models")


def fig_ratios(R, M):
    ss = np.linspace(0, 75, 76)
    fig, axs = plt.subplots(2, 3, figsize=(10.0, 6.2), sharey="row")
    for j, rcf in enumerate((10, 30, 50)):
        ax = axs[0, j]
        for a in ("7", "28"):
            p1 = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ss, "AB": 0.45, "carbonation": "1 h"})
            p0 = p1.assign(carbonation="NC")
            r, lo, hi = M[a].ratio(p1, p0)
            ax.fill_between(ss, lo, hi, color=COL[a], alpha=0.14, lw=0)
            ax.plot(ss, r, color=COL[a], lw=1.8)
        ax.axhline(1, color=AXIS, lw=0.9)
        ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 0.8, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.5", "2", "3"])
        ax.minorticks_off(); ax.set_ylim(0.35, 3.2)
        ax.set_title(f"Carbonated (1 h) / NC, RCF {rcf} %", loc="left", fontsize=8.5)
        ax.set_xlabel("SS (%)")
        if j == 0:
            ax.set_ylabel("Strength ratio")
    for j, carb in enumerate(("NC", "1 h", "5 h")):
        ax = axs[1, j]
        for a in ("7", "28"):
            p1 = pd.DataFrame({"RCF_pct": 50, "SS_pct": ss, "AB": 0.45, "carbonation": carb})
            p0 = p1.assign(RCF_pct=10)
            r, lo, hi = M[a].ratio(p1, p0)
            ax.fill_between(ss, lo, hi, color=COL[a], alpha=0.14, lw=0)
            ax.plot(ss, r, color=COL[a], lw=1.8)
        ax.axhline(1, color=AXIS, lw=0.9)
        ax.set_yscale("log"); ax.set_yticks([0.6, 0.8, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3"])
        ax.minorticks_off(); ax.set_ylim(0.55, 3.2)
        ax.set_title(f"RCF 50 % / RCF 10 %, {carb}", loc="left", fontsize=8.5)
        ax.set_xlabel("SS (%)")
        if j == 0:
            ax.set_ylabel("Strength ratio")
    h = [Line2D([], [], color=C7, lw=1.8, label="7-day model"), Line2D([], [], color=C28, lw=1.8, label="28-day model")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("Bands: 95 % CI of the ratio. A ratio of 1 means no effect; a flat line at 1 means the term is not in that model.",
                 x=0.01, y=-0.01, ha="left", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save(fig, "H05_effect_ratios_both_models")


def fig_maps(R, d, M):
    Rg = np.linspace(10, 50, 81); Sg = np.linspace(0, 75, 151)
    RR, SS = np.meshgrid(Rg, Sg)
    fig, axs = plt.subplots(2, 4, figsize=(10.6, 5.4), sharex=True, sharey=True)
    for i, a in enumerate(("7", "28")):
        vmax = 26 if a == "7" else 40
        levels = np.arange(0, vmax + 0.01, 2)
        opt = {o["carbonation"]: o for o in R[a]["optimum"]}
        for j, carb in enumerate(dz.CARB_LEVELS):
            ax = axs[i, j]
            ab = opt[carb]["AB"]
            pts = pd.DataFrame({"RCF_pct": RR.ravel(), "SS_pct": SS.ravel(), "AB": ab, "carbonation": carb})
            Z = M[a].pred(pts)[0].reshape(RR.shape)
            cf = ax.contourf(RR, SS, Z, levels=levels, cmap=SEQ if a == "7" else SEQ_O, extend="max")
            cs = ax.contour(RR, SS, Z, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.8)
            ax.clabel(cs, fmt="%d", fontsize=6, colors="white")
            s = d[d.carbonation == carb]
            ax.scatter(s.RCF_pct, s.SS_pct, s=14, color="white", edgecolor=INK, lw=0.6, zorder=3, clip_on=False)
            o = opt[carb]
            ax.scatter(o["RCF"], o["SS"], s=70, marker="*", color="#eda100", edgecolor=INK, lw=0.6, zorder=4, clip_on=False)
            ax.grid(False)
            if i == 0:
                ax.set_title(carb)
            if j == 0:
                ax.set_ylabel(f"{a}-day model\nSS (%)")
            if i == 1:
                ax.set_xlabel("RCF (%)")
            if "C" in R[a]["final_terms"]:
                ax.text(0.02, 0.02, f"A/B {ab:.2f}", transform=ax.transAxes, fontsize=6.5, color="white")
        cb = fig.colorbar(cf, ax=axs[i, :], shrink=0.9, pad=0.012)
        cb.set_label(f"{a} d strength (MPa)"); cb.outline.set_visible(False)
    fig.suptitle("Median predicted strength over RCF × SS from each model (own colour scale per age). ○ runs at that level, ★ maximum",
                 x=0.02, ha="left", fontsize=8.5, color=INK2)
    save(fig, "H06_response_surfaces_both_models")


def fig_obs_pred(R):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
    for ax, a in zip(axs, ("7", "28")):
        t = pd.DataFrame(R[a]["diagnostics"]["table"])
        ax.plot([2.5, 45], [2.5, 45], color=AXIS, lw=0.9)
        ax.scatter(t.observed, t.fitted, color=COL[a], s=22, marker=MK[a], edgecolor="white", lw=0.6, label="fitted", zorder=3)
        ax.scatter(t.observed, t.loo, facecolor="white", edgecolor=COL[a], s=22, marker=MK[a], lw=1.1, label="left out of fit", zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ticks = [3, 5, 10, 20, 40]
        ax.set_xticks(ticks); ax.set_xticklabels(map(str, ticks)); ax.set_yticks(ticks); ax.set_yticklabels(map(str, ticks))
        ax.minorticks_off(); ax.set_xlim(2.5, 45); ax.set_ylim(2.5, 45)
        v = R[a]["validation"]
        ax.text(0.97, 0.04, f"R² {R[a]['fit']['R2']:.2f} · adj {R[a]['fit']['adjR2']:.2f} · pred {v['loo']['predR2']:.2f}\n"
                f"RMSE fitted {v['fit']['rmse_MPa']:.1f} · left out {v['loo']['rmse_MPa']:.1f} MPa",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
        ax.set_xlabel("Observed (MPa)"); ax.set_title(f"{a}-day model", loc="left")
        ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel("Model (MPa, median)")
    panel(axs[0], "a"); panel(axs[1], "b")
    fig.tight_layout()
    save(fig, "H07_observed_vs_predicted_both_models")


def fig_diag(R):
    fig, axs = plt.subplots(2, 4, figsize=(11.0, 5.2))
    for i, a in enumerate(("7", "28")):
        t = pd.DataFrame(R[a]["diagnostics"]["table"]); D = R[a]["diagnostics"]
        dfr = R[a]["fit"]["df_resid"]
        ax = axs[i, 0]
        ax.scatter(np.log(t.fitted), t.t_ext, color=COL[a], marker=MK[a], s=20, edgecolor="white", lw=0.6)
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.set_xlabel("Fitted ln(strength)"); ax.set_ylabel(f"{a} d\nStudentized residual")
        ax.set_title(f"vs fitted (Breusch–Pagan p = {D['bp_p']:.2f})", loc="left", fontsize=8)
        ax = axs[i, 1]
        r = np.sort(t.t_ext.values); q = stats.norm.ppf((np.arange(1, 31) - 0.375) / 30.25)
        ax.plot([-2.6, 2.6], [-2.6, 2.6], color=AXIS, lw=0.9)
        ax.scatter(q, r, color=COL[a], marker=MK[a], s=18, edgecolor="white", lw=0.5)
        ax.set_xlabel("Normal quantile"); ax.set_title(f"Q–Q (Shapiro–Wilk p = {D['shapiro_p']:.2f})", loc="left", fontsize=8)
        ax = axs[i, 2]
        crit = stats.t.ppf(1 - 0.025 / 30, dfr - 1)
        ax.vlines(t.mix, 0, t.t_ext, color=COL[a], lw=2)
        for yv in (-crit, crit):
            ax.axhline(yv, color=MUTED, lw=0.9)
        ax.axhline(0, color=AXIS, lw=0.9)
        k = int(np.argmax(np.abs(t.t_ext.values)))
        ax.annotate(f"mix {int(t.mix[k])}", (t.mix[k], t.t_ext[k]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_xlabel("Mix"); ax.set_title("Outliers (Bonferroni limits)", loc="left", fontsize=8)
        ax.set_xticks([1, 10, 20, 30]); ax.grid(axis="x", visible=False)
        ax = axs[i, 3]
        ax.vlines(t.mix, 0, t.cook, color=INK2, lw=2.2)
        ax.axhline(4 / 30, color=MUTED, lw=0.9)
        k = int(np.argmax(t.cook.values))
        ax.annotate(f"mix {int(t.mix[k])}", (t.mix[k], t.cook[k]), xytext=(4, 1), textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_xlabel("Mix"); ax.set_title("Cook's distance", loc="left", fontsize=8)
        ax.set_xticks([1, 10, 20, 30]); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "H08_residual_diagnostics_both_models")


def fig_robust(R):
    keys = ["RCF50/10, 1 h, SS 0", "RCF50/10, NC, SS 75", "1 h/NC, RCF 50, SS 0", "1 h/NC, RCF 10, SS 75", "SS 50/0, NC, RCF 30",
            "SS 75/50, NC, RCF 30", "A/B 0.42/0.48"]
    fig, axs = plt.subplots(2, len(keys), figsize=(12.4, 7.2), sharey="row")
    for i, a in enumerate(("7", "28")):
        rb = pd.DataFrame(R[a]["robustness"])
        labels = [s.replace("PRIMARY: final model (ln, 4-level carbonation, OLS)", "PRIMARY").replace("Same terms, ", "")[:44] for s in rb.analysis]
        y = np.arange(len(rb))[::-1]
        for j, k in enumerate(keys):
            ax = axs[i, j]
            v = rb[k].values
            ax.scatter(v, y, s=20, color=[INK] + [COL[a]] * (len(v) - 1), edgecolor="white", lw=0.6, zorder=3)
            ax.axvline(1, color=AXIS, lw=0.9); ax.axvline(v[0], color=INK, lw=0.8, alpha=0.35)
            ax.set_xscale("log"); ax.minorticks_off()
            lo, hi = np.nanmin(v), np.nanmax(v)
            tk = [t for t in (0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3, 5, 7) if lo * 0.8 <= t <= hi * 1.25] or [1]
            ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk], fontsize=6.5)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(k.replace(", ", "\n", 1), fontsize=7.2, loc="left")
        axs[i, 0].set_yticks(y); axs[i, 0].set_yticklabels(labels, fontsize=6.3)
        axs[i, 0].set_ylabel(f"{a}-day model", fontsize=8)
    fig.suptitle("Robustness of each model's trend ratios under alternative analyses (black = primary)", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, "H09_robustness_both_models")


def fig_boot(R):
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, a in zip(axs, ("7", "28")):
        A = R[a]
        coef = pd.DataFrame(A["coef"])
        names = coef.column.tolist()
        y = np.arange(len(names))[::-1]
        for j, nm in enumerate(names):
            r = coef.iloc[j]
            ax.plot([r.lo, r.hi], [y[j] + 0.2] * 2, color=INK, lw=1.8, solid_capstyle="round")
            ax.scatter(r.coef, y[j] + 0.2, s=16, color=INK, zorder=3)
            lo, hi = A["bootstrap"]["resid"]["beta_ci"][nm]
            ax.plot([lo, hi], [y[j]] * 2, color=COL[a], lw=1.8)
            lo, hi = A["bootstrap"]["case"]["beta_ci"][nm]
            ax.plot([lo, hi], [y[j] - 0.2] * 2, color=COL[a], lw=1.8, alpha=0.45)
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([n.replace("Carb[", "Carb [") for n in names], fontsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{a}-day model coefficients, 95 % intervals", loc="left")
        ax.set_xlabel("Coefficient (coded, ln scale)")
        h = [Line2D([], [], color=INK, lw=1.8, marker="o", ms=3.5, label="t interval"),
             Line2D([], [], color=COL[a], lw=1.8, label=f"residual bootstrap ({A['bootstrap']['resid']['n']})"),
             Line2D([], [], color=COL[a], lw=1.8, alpha=0.45, label=f"case bootstrap, stratified ({A['bootstrap']['case']['n']})")]
        ax.legend(handles=h, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "H10_coefficient_intervals_both_models")


def main():
    R, d, M = load()
    fig_selection(R)
    fig_coefficients(R)
    fig_curves_ss(R, d, M)
    fig_curves_rcf(R, d, M)
    fig_ratios(R, M)
    fig_maps(R, d, M)
    fig_obs_pred(R)
    fig_diag(R)
    fig_robust(R)
    fig_boot(R)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
