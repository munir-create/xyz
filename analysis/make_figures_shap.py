"""
SHAP figures (PNG 300 dpi + vector PDF) from results/shap/shap_results.json.

  Z01  mean |SHAP| importance with 95 % bootstrap intervals
  Z02  SHAP summary (beeswarm)
  Z03  SHAP dependence plots, 7-day and 28-day models on one scale
  Z04  SHAP interaction matrices (mean |interaction value|)
  Z05  SHAP interaction dependence plots for the key pairs
  Z06  waterfall decompositions (MPa) for selected mixtures
  Z07  paired model: curing age and its interaction with SS
  Z08  supplementary: summary and importance on the MPa scale

Run:  python analysis/make_figures_shap.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import design as dz
from make_figures import C7, C28, CARB_COL, CARB_MK, INK, INK2, MUTED, GRID, AXIS, panel

RES = os.path.join(dz.ROOT, "results", "shap")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
LEVELS = dz.CARB_LEVELS
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#6da7ec", "#c9c7c0", "#ec835a", "#c73a39"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
POS, NEG = "#c73a39", "#1c5cab"
TITLES = {"7d": "7-day model", "28d": "28-day model", "paired": "Paired 7/28-day model"}
RANGES = {"RCF": (10, 50), "SS": (0, 75), "A/B": (0.42, 0.48), "Curing age": (7, 28)}
UNITS = {"RCF": "RCF (%)", "SS": "SS (%)", "A/B": "A/B", "Carbonation": "Carbonation", "Curing age": "Curing age (d)"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def load():
    with open(os.path.join(RES, "shap_results.json")) as f:
        return json.load(f)


def _spaced_factors(lo, hi, min_gap=0.3):
    """Multiplicative-factor ticks inside [lo, hi] (ln units), at least min_gap apart, always including x1."""
    cands = [1, 1.25, 0.8, 1.5, 0.67, 2, 0.5, 3, 0.33, 4, 0.25]
    out = []
    for c in cands:
        v = np.log(c)
        if lo <= v <= hi and all(abs(v - np.log(o)) >= min_gap for o in out):
            out.append(c)
    return sorted(out)


def factor_axis(ax, lim=None):
    """Top axis showing the multiplicative factor exp(SHAP) for ln-scale SHAP values."""
    lo, hi = ax.get_xlim() if lim is None else lim
    top = ax.secondary_xaxis("top", functions=(np.exp, np.log))
    ticks = _spaced_factors(lo, hi)
    top.set_xticks(ticks)
    top.set_xticklabels([f"×{t:g}" for t in ticks], fontsize=7)
    top.tick_params(length=2, colors=INK2)
    return top


def factor_yaxis(ax):
    lo, hi = ax.get_ylim()
    right = ax.secondary_yaxis("right", functions=(np.exp, np.log))
    ticks = _spaced_factors(lo, hi, min_gap=0.18)
    right.set_yticks(ticks)
    right.set_yticklabels([f"×{t:g}" for t in ticks], fontsize=6.5)
    right.tick_params(length=2, colors=INK2)
    return right


def feat_color_values(key, R, j):
    """Normalised colour value in [0, 1] for a continuous feature (or age), None for carbonation."""
    X = np.array(R[key]["X"])
    fn = R[key]["features"][j]
    if fn == "Carbonation":
        return None
    lo, hi = RANGES[fn]
    return (X[:, j] - lo) / (hi - lo)


def swarm_offsets(x, width=0.36, nbins=40):
    """Simple beeswarm: stack points that share an x bin symmetrically around the row centre."""
    x = np.asarray(x)
    if np.ptp(x) == 0:
        edges = np.array([x[0] - 1, x[0] + 1])
    else:
        edges = np.linspace(x.min(), x.max(), nbins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    off = np.zeros(len(x))
    for k in np.unique(b):
        idx = np.where(b == k)[0]
        idx = idx[np.argsort(x[idx])]
        n = len(idx)
        step = min(width / max(n, 1), 0.06)
        pos = (np.arange(n) - (n - 1) / 2) * step * 2
        order = np.argsort(np.abs(pos))
        off[idx] = pos[order] if n > 1 else 0
    return off


# ================================================================== Z01 importance
def fig_importance(R, scale="ln", name="Z01_mean_abs_SHAP_importance"):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    feats = R["7d"]["features"]
    order = np.argsort(-(np.array(R["7d"][scale]["mean_abs"]) + np.array(R["28d"][scale]["mean_abs"])))
    y = np.arange(len(feats))[::-1]
    for k, (key, col, off) in enumerate((("7d", C7, 0.19), ("28d", C28, -0.19))):
        m = np.array(R[key][scale]["mean_abs"])[order]
        lo, hi = (np.array(v)[order] for v in R[key][scale]["mean_abs_ci"])
        ax.barh(y + off, m, height=0.34, color=col, label=TITLES[key])
        ax.errorbar(m, y + off, xerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        for yy, v, h in zip(y + off, m, hi):
            lab = ("0 (not in model)" if v < 1e-12 else (f"{v:.2f}" if scale == "ln" else f"{v:.1f}"))
            ax.text(h + 0.008 * (1 if scale == "ln" else 10), yy, lab, va="center", fontsize=6.8, color=MUTED if v < 1e-12 else INK2)
    ax.set_yticks(y); ax.set_yticklabels([feats[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value|  (ln strength)" if scale == "ln" else "Mean |SHAP value|  (MPa)")
    ax.set_title("Independent age models", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(0, None)
    panel(ax, "a")
    ax = axs[1]
    feats = R["paired"]["features"]
    m_all = np.array(R["paired"][scale]["mean_abs"])
    order = np.argsort(-m_all)
    y = np.arange(len(feats))[::-1]
    m = m_all[order]
    lo, hi = (np.array(v)[order] for v in R["paired"][scale]["mean_abs_ci"])
    ax.barh(y, m, height=0.5, color=INK2)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    for yy, v, h in zip(y, m, hi):
        lab = ("0 (not in model)" if v < 1e-12 else (f"{v:.2f}" if scale == "ln" else f"{v:.1f}"))
        ax.text(h + 0.008 * (1 if scale == "ln" else 10), yy, lab, va="center", fontsize=6.8, color=MUTED if v < 1e-12 else INK2)
    ax.set_yticks(y); ax.set_yticklabels([feats[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value|  (ln strength)" if scale == "ln" else "Mean |SHAP value|  (MPa)")
    ax.set_title(TITLES["paired"], loc="left")
    ax.set_xlim(0, None)
    panel(ax, "b")
    fig.text(0.01, -0.02, "Bars: mean |SHAP| over the 30 mixtures (60 mixture × age rows in b). Whiskers: 95 % interval over 1,000 case-bootstrap refits "
             "stratified by carbonation level.", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, name)


# ================================================================== Z02 beeswarm
def beeswarm(ax, R, key, scale="ln", show_cbar=False):
    E = R[key][scale]
    phi = np.array(E["phi"])
    X = np.array(R[key]["X"])
    feats = R[key]["features"]
    order = np.argsort(np.abs(phi).mean(0))           # least important at the bottom
    for row, j in enumerate(order):
        v = phi[:, j]
        if np.allclose(v, 0):
            continue
        off = swarm_offsets(v, width=0.34)
        fn = feats[j]
        if fn == "Carbonation":
            for li, lv in enumerate(LEVELS):
                s = X[:, 3].astype(int) == li
                ax.scatter(v[s], row + off[s], s=16, marker=CARB_MK[lv], color=CARB_COL[lv], edgecolor="white", lw=0.4, zorder=3)
        else:
            cv = feat_color_values(key, R, j)
            ax.scatter(v, row + off, s=16, c=cv, cmap=DIV, vmin=0, vmax=1, edgecolor="white", lw=0.4, zorder=3)
    for row, j in enumerate(order):
        if np.allclose(phi[:, j], 0):
            ax.text(0.03, row, "not in model (SHAP = 0)", transform=ax.get_yaxis_transform(), va="center", fontsize=6.8, color=MUTED)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feats[j] for j in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP value (change in ln strength)" if scale == "ln" else "SHAP value (MPa)")
    ax.set_title(TITLES[key], loc="left", pad=22 if scale == "ln" else 6)


def fig_beeswarm(R, scale="ln", name="Z02_SHAP_summary_beeswarm"):
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    lim = 0
    for key in ("7d", "28d", "paired"):
        lim = max(lim, np.abs(np.array(R[key][scale]["phi"])).max())
    lim *= 1.08
    for ax, key, letter in zip(axs, ("7d", "28d", "paired"), "abc"):
        beeswarm(ax, R, key, scale)
        ax.set_xlim(-lim, lim)
        if scale == "ln":
            factor_axis(ax)
        panel(ax, letter)
    # legends: feature value colour bar + carbonation levels
    sm = plt.cm.ScalarMappable(cmap=DIV, norm=Normalize(0, 1))
    cax = fig.add_axes([0.93, 0.25, 0.008, 0.5])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=7)
    cb.set_label("Feature value\n(RCF, SS, A/B, age)", fontsize=7)
    cb.outline.set_visible(False)
    h = [Line2D([], [], marker=CARB_MK[l], color=CARB_COL[l], ls="", ms=5, label=l) for l in LEVELS]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.2, title="Carbonation (categorical)", title_fontsize=7.2,
               bbox_to_anchor=(0.46, -0.08))
    fig.subplots_adjust(left=0.08, right=0.9, wspace=0.42, bottom=0.2, top=0.82)
    save(fig, name)


# ================================================================== Z03 dependence
def fig_dependence(R):
    feats = R["7d"]["features"]
    fig, axs = plt.subplots(2, 4, figsize=(11.2, 5.6), sharey=True)
    lim = max(np.abs(np.array(R[k]["ln"]["phi"])).max() for k in ("7d", "28d")) * 1.1
    for i, key in enumerate(("7d", "28d")):
        phi = np.array(R[key]["ln"]["phi"])
        X = np.array(R[key]["X"])
        for j, fn in enumerate(feats):
            ax = axs[i, j]
            if fn == "Carbonation":
                rng = np.random.default_rng(3)
                xs = X[:, 3] + rng.uniform(-0.16, 0.16, len(X))
                sc = ax.scatter(xs, phi[:, j], c=(X[:, 1] - 0) / 75, cmap=DIV, vmin=0, vmax=1, s=26, edgecolor="white", lw=0.5, zorder=3)
                for li in range(4):
                    s = X[:, 3] == li
                    if s.any():
                        ax.hlines(phi[s, j].mean(), li - 0.28, li + 0.28, color=INK, lw=1.4, zorder=4)
                ax.set_xticks(range(4)); ax.set_xticklabels(LEVELS)
                ax.set_xlim(-0.5, 3.5)
                ax.grid(axis="x", visible=False)
                if i == 0:
                    ax.text(0.98, 0.97, "colour: SS (low→high)\nbar: level mean", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
            else:
                for li, lv in enumerate(LEVELS):
                    s = X[:, 3] == li
                    ax.scatter(X[s, j], phi[s, j], marker=CARB_MK[lv], color=CARB_COL[lv], s=24, edgecolor="white", lw=0.5, zorder=3)
            ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
            ax.set_ylim(-lim, lim)
            if i == 1:
                ax.set_xlabel(UNITS[fn])
            if j == 0:
                ax.set_ylabel(f"{TITLES[key]}\nSHAP value (ln strength)")
            if j == 3:
                factor_yaxis(ax)
            if i == 0:
                ax.set_title(fn, loc="left")
            if np.allclose(phi[:, j], 0):
                ax.text(0.5, 0.85, "not in this model (SHAP = 0)", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=MUTED)
    h = [Line2D([], [], marker=CARB_MK[l], color=CARB_COL[l], ls="", ms=5, label=l) for l in LEVELS]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=7.2, title="Point colour and shape in RCF, SS and A/B panels: carbonation level",
               title_fontsize=7.2, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Z03_SHAP_dependence_7d_vs_28d")


# ================================================================== Z04 interaction matrices
def fig_inter_matrix(R):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [4, 4, 5]})
    vmax = max(np.array(R[k]["ln"]["mean_abs_inter"]).max() for k in ("7d", "28d", "paired"))
    for ax, key, letter in zip(axs, ("7d", "28d", "paired"), "abc"):
        Mx = np.array(R[key]["ln"]["mean_abs_inter"])
        feats = R[key]["features"]
        im = ax.imshow(Mx, cmap=SEQ, vmin=0, vmax=vmax)
        for a in range(len(feats)):
            for b in range(len(feats)):
                v = Mx[a, b]
                ax.text(b, a, f"{v:.3f}" if v > 1e-10 else "0", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 0.55 * vmax else INK)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=35, ha="right", fontsize=7.2)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7.2)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(TITLES[key], loc="left")
        panel(ax, letter)
    fig.subplots_adjust(wspace=0.55, left=0.07, right=0.88, bottom=0.22, top=0.88)
    cax = fig.add_axes([0.9, 0.3, 0.01, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean |SHAP interaction value| (ln strength)", fontsize=7.5)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.04, "Diagonal: main effect of each feature; off-diagonal: the interaction value of the pair (each half shown in both cells; "
             "a row sums to the feature's SHAP value). Zero cells: the model has no term linking the two features.", fontsize=7, color=INK2)
    save(fig, "Z04_SHAP_interaction_matrices")


# ================================================================== Z05 interaction dependence
def fig_inter_dep(R):
    pairs = [("7d", "SS", "Carbonation", "carb"), ("7d", "RCF", "Carbonation", "carb"), ("7d", "RCF", "SS", "SS"),
             ("paired", "SS", "Curing age", "age")]
    fig, axs = plt.subplots(1, 4, figsize=(12.0, 3.4), sharey=True)
    lim = 0
    for key, a, b, _ in pairs:
        f = R[key]["features"]
        I = np.array(R[key]["ln"]["inter"])
        lim = max(lim, np.abs(2 * I[:, f.index(a), f.index(b)]).max())
    lim *= 1.15
    for ax, (key, a, b, colby), letter in zip(axs, pairs, "abcd"):
        f = R[key]["features"]
        X = np.array(R[key]["X"])
        I = np.array(R[key]["ln"]["inter"])
        ia, ib = f.index(a), f.index(b)
        v = 2 * I[:, ia, ib]            # full interaction effect (both halves)
        if colby == "carb":
            for li, lv in enumerate(LEVELS):
                s = X[:, 3] == li
                ax.scatter(X[s, ia], v[s], marker=CARB_MK[lv], color=CARB_COL[lv], s=26, edgecolor="white", lw=0.5, zorder=3, label=lv)
        elif colby == "SS":
            ax.scatter(X[:, ia], v, c=X[:, 1] / 75, cmap=DIV, vmin=0, vmax=1, s=26, edgecolor="white", lw=0.5, zorder=3)
            ax.text(0.03, 0.95, "colour: SS (blue low → red high)", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
        else:
            for age, col, mk in ((7, C7, "o"), (28, C28, "s")):
                s = X[:, 4] == age
                ax.scatter(X[s, ia], v[s], color=col, marker=mk, s=26, edgecolor="white", lw=0.5, zorder=3, label=f"{age} d")
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(UNITS[a])
        ax.set_title(f"{a} × {b}\n{TITLES[key]}", loc="left", fontsize=8.3)
        panel(ax, letter)
    axs[0].set_ylabel("SHAP interaction value\n(ln strength, both halves)")
    factor_yaxis(axs[-1])
    h1 = [Line2D([], [], marker=CARB_MK[l], color=CARB_COL[l], ls="", ms=5, label=l) for l in LEVELS]
    h2 = [Line2D([], [], marker="o", color=C7, ls="", ms=5, label="7 d"), Line2D([], [], marker="s", color=C28, ls="", ms=5, label="28 d")]
    fig.legend(handles=h1, loc="upper left", ncol=4, fontsize=7, title="Carbonation (a, b)", title_fontsize=7, bbox_to_anchor=(0.05, 1.12))
    fig.legend(handles=h2, loc="upper right", ncol=2, fontsize=7, title="Curing age (d)", title_fontsize=7, bbox_to_anchor=(0.97, 1.12))
    fig.text(0.01, -0.05, "The 28-day model contains no interaction terms, so all of its interaction values are exactly zero on the ln scale "
             "(Z04 b).", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "Z05_SHAP_interaction_dependence")


# ================================================================== Z06 waterfalls (MPa)
def waterfall(ax, R, key, i, mixlabel):
    E = R[key]["MPa"]
    phi = np.array(E["phi"])[i]
    base = E["base"]
    fx = np.array(E["fx"])[i]
    feats = R[key]["features"]
    X = np.array(R[key]["X"])[i]
    vals = {"RCF": f"{X[0]:g} %", "SS": f"{X[1]:g} %", "A/B": f"{X[2]:.3f}", "Carbonation": LEVELS[int(X[3])]}
    order = np.argsort(-np.abs(phi))
    cum = base
    y = len(order)
    ax.axvline(base, color=AXIS, lw=0.9, zorder=1)
    for j in order:
        v = phi[j]
        ax.barh(y, v, left=cum, height=0.6, color=POS if v >= 0 else NEG, zorder=3)
        tx = cum + v
        ax.text(max(cum, tx), y, f" {v:+.1f}", va="center", fontsize=6.8, color=INK2)
        ax.plot([tx, tx], [y - 0.3, y - 0.7], color=MUTED, lw=0.6)
        cum = tx
        y -= 1
    ax.set_yticks(range(len(order), 0, -1))
    ax.set_yticklabels([f"{feats[j]} = {vals[feats[j]]}" for j in order], fontsize=7)
    ax.axvline(fx, color=INK, lw=0.9, ls="-", zorder=2)
    ax.grid(axis="y", visible=False)
    pts = np.concatenate([[base, fx], base + np.cumsum(phi[order])])
    lo, hi = pts.min(), pts.max()
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.12 * span, hi + 0.28 * span)
    ax.set_ylim(-0.45, len(order) + 0.6)
    ax.set_title(mixlabel, loc="left", fontsize=8)
    ax.text(base, -0.35, f"E[f] = {base:.1f}", fontsize=6.5, color=MUTED, ha="center", va="bottom")
    ax.text(fx, 0.2, f"f(x) = {fx:.1f}", fontsize=7, color=INK, ha="left" if fx >= base else "right", va="bottom", fontweight="bold")


def fig_waterfalls(R):
    d = dz.load()
    mixes = [25, 29, 30, 27]
    fig, axs = plt.subplots(2, 4, figsize=(12.4, 5.2))
    for r, key in enumerate(("7d", "28d")):
        for c, mx in enumerate(mixes):
            i = int(np.where(d.mix.values == mx)[0][0])
            row = d.iloc[i]
            obs = row[f"f{key[:-1]}_mean"]
            waterfall(axs[r, c], R, key, i, f"{TITLES[key]} · mix {mx} (observed {obs:.1f} MPa)")
            if r == 1:
                axs[r, c].set_xlabel("Predicted strength (MPa)")
    h = [plt.Rectangle((0, 0), 1, 1, color=POS, label="raises the prediction"), plt.Rectangle((0, 0), 1, 1, color=NEG, label="lowers the prediction")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.02, "Exact Shapley decomposition of each model's median prediction in MPa: E[f] is the mean prediction over the 30 mixtures, "
             "the bars add up to the prediction f(x).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "Z06_SHAP_waterfalls_selected_mixes")


# ================================================================== Z07 paired model
def fig_paired(R):
    key = "paired"
    f = R[key]["features"]
    X = np.array(R[key]["X"])
    phi = np.array(R[key]["ln"]["phi"])
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.4))
    ax = axs[0]
    j = f.index("Curing age")
    for age, col, mk in ((7, C7, "o"), (28, C28, "s")):
        s = X[:, 4] == age
        ax.scatter(X[s, 1], phi[s, j], color=col, marker=mk, s=26, edgecolor="white", lw=0.5, label=f"{age} d", zorder=3)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("SS (%)"); ax.set_ylabel("SHAP value of curing age (ln strength)")
    ax.set_title("Curing age contribution against SS", loc="left")
    ax.legend(fontsize=7, title="Curing age", title_fontsize=7)
    factor_yaxis(ax)
    panel(ax, "a")
    ax = axs[1]
    j = f.index("SS")
    for age, col, mk in ((7, C7, "o"), (28, C28, "s")):
        s = X[:, 4] == age
        ax.scatter(X[s, 1], phi[s, j], color=col, marker=mk, s=26, edgecolor="white", lw=0.5, label=f"{age} d", zorder=3)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("SS (%)"); ax.set_ylabel("SHAP value of SS (ln strength)")
    ax.set_title("SS contribution, coloured by age", loc="left")
    factor_yaxis(ax)
    panel(ax, "b")
    ax = axs[2]
    j = f.index("RCF")
    for li, lv in enumerate(LEVELS):
        s = X[:, 3] == li
        ax.scatter(X[s, 0], phi[s, j], marker=CARB_MK[lv], color=CARB_COL[lv], s=24, edgecolor="white", lw=0.5, label=lv, zorder=3)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("RCF (%)"); ax.set_ylabel("SHAP value of RCF (ln strength)")
    ax.set_title("RCF contribution, coloured by carbonation", loc="left")
    ax.legend(fontsize=6.5, title="Carbonation", title_fontsize=6.5)
    factor_yaxis(ax)
    panel(ax, "c")
    fig.tight_layout()
    save(fig, "Z07_SHAP_paired_model_age")


def main():
    R = load()
    fig_importance(R)
    fig_beeswarm(R)
    fig_dependence(R)
    fig_inter_matrix(R)
    fig_inter_dep(R)
    fig_waterfalls(R)
    fig_paired(R)
    fig_importance(R, "MPa", "Z08a_mean_abs_SHAP_importance_MPa")
    fig_beeswarm(R, "MPa", "Z08b_SHAP_summary_beeswarm_MPa")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
