"""
Publication figures (PNG 300 dpi + vector PDF) from results/model_results.json.

Run:  python analysis/make_figures.py
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
import model_tools as mt
from lmm import PairedLMM

RES = os.path.join(dz.ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- style (validated palette)
C7, C28 = "#2a78d6", "#eb6834"                  # age: 7 d / 28 d
CARB_COL = {"NC": "#3d3c39", "0.5 h": "#008300", "1 h": "#4a3aa7", "5 h": "#e87ba4"}
CARB_MK = {"NC": "o", "0.5 h": "^", "1 h": "s", "5 h": "D"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                 "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)


def carb_legend(ax, loc="upper left", **kw):
    h = [Line2D([], [], marker=CARB_MK[c], color=CARB_COL[c], ls="", ms=5, label=c) for c in dz.CARB_LEVELS]
    ax.legend(handles=h, loc=loc, title="Carbonation", title_fontsize=7.5, **kw)


def load():
    with open(os.path.join(RES, "model_results.json")) as f:
        R = json.load(f)
    d = dz.load()
    final = R["final_terms"]
    X, names = dz.design(final, d)
    m = PairedLMM(d[["l7", "l28"]].values, X, names, R["final_struct"], fit=False)
    m.beta = np.array(R["final"]["beta"]); m.cov_beta = np.array(R["final"]["cov_beta"])
    m.Sigma = np.array(R["final"]["Sigma"])
    return R, d, final, m


def pred_band(m, final, pts, df):
    X = mt.rows(final, pts)
    est, se, lo, hi = mt.lincomb_fast(m, X, df)
    return np.exp(est), np.exp(lo), np.exp(hi)


# ================================================================ figures
def fig_data(R, d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axs[0]
    lim = [2.5, 45]
    xs = np.array(lim)
    for r in (1, 2, 3, 4):
        ax.plot(xs, r * xs, color=GRID if r != 1 else AXIS, lw=0.9, zorder=1)
        xe = min(30.0, 45.0 / r) / 1.12
        ax.text(xe, r * xe * 1.02, f"f28 = {r}×f7" if r == 1 else f"×{r}", fontsize=6.8, color=MUTED,
                ha="right", va="bottom")
    for c in dz.CARB_LEVELS:
        s = d[d.carbonation == c]
        ax.scatter(s.f7_mean, s.f28_mean, marker=CARB_MK[c], s=26, color=CARB_COL[c], edgecolor="white",
                   linewidth=0.7, zorder=3, label=c)
    gr = (d.f28_mean / d.f7_mean).values
    lab = set(d.mix.values[np.argsort(gr)[:2]]) | set(d.mix.values[np.argsort(-gr)[:2]]) | {int(d.mix.values[np.argmax(d.f28_mean.values)])}
    for _, r in d.iterrows():
        if r.mix in lab:
            ax.annotate(f"{int(r.mix)}", (r.f7_mean, r.f28_mean), xytext=(4, 2), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2.5, 30); ax.set_ylim(6, 45)
    ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["3", "5", "10", "20", "30"])
    ax.set_yticks([7, 10, 20, 30, 40]); ax.set_yticklabels(["7", "10", "20", "30", "40"])
    ax.minorticks_off()
    ax.set_xlabel("7-day strength (MPa)"); ax.set_ylabel("28-day strength (MPa)")
    ax.set_title("Paired strengths of the 30 mixes (log axes)", loc="left")
    carb_legend(ax, loc="lower right")
    panel(ax, "a")

    ax = axs[1]
    gain = d.f28_mean / d.f7_mean
    for c in dz.CARB_LEVELS:
        s = d.carbonation == c
        ax.scatter(d.SS_pct[s], gain[s], marker=CARB_MK[c], s=26, color=CARB_COL[c], edgecolor="white",
                   linewidth=0.7, zorder=3)
    ax.set_yscale("log"); ax.set_yticks([1, 1.5, 2, 3, 4]); ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
    ax.minorticks_off()
    ax.set_xlabel("Sodium silicate, SS (%)"); ax.set_ylabel("Strength gain  f28 / f7")
    ax.set_title("Observed 7 → 28-day gain against SS", loc="left")
    ax.set_xlim(-3, 78)
    panel(ax, "b")
    fig.tight_layout()
    save(fig, "F01_data_overview")


def fig_scale_selection(R):
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    ax = axs[0]
    for key, col, lab in (("final_UN", INK, "Final model"), ("full_UN", MUTED, "Full candidate model")):
        b = R["boxcox"][key]
        lam, ll = np.array(b["curve"]["lambda"]), np.array(b["curve"]["loglik"])
        ax.plot(lam, ll, color=col, lw=1.6, label=f"{lab}: λ̂ = {b['lambda']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]")
        ax.axvspan(b["ci"][0], b["ci"][1], color=col, alpha=0.08, lw=0)
    ax.axhline(R["boxcox"]["final_UN"]["cut"], color=AXIS, lw=0.9)
    ax.axvline(0, color=C7, lw=0.9); ax.text(0.04, 0.35, "ln", color=C7, fontsize=7)
    ax.axvline(1, color=MUTED, lw=0.9); ax.text(0.96, 0.35, "none", color=MUTED, fontsize=7, ha="right")
    ax.set_ylim(-12.5, 0.8); ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("Box–Cox λ"); ax.set_ylabel("Profile log-likelihood (relative)")
    ax.set_title("Response scale", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=6.5)
    panel(ax, "a")

    ax = axs[1]
    terms = dz.ALL_TERMS
    imp = np.array([R["search"]["UN"]["importance"][t] for t in terms])
    bc = np.array([R["bootstrap"]["subsample"]["inclusion"][t] for t in terms])
    br = np.array([R["bootstrap"]["resid"]["inclusion"][t] for t in terms])
    order = np.argsort(-imp)[:16]
    y = np.arange(len(order))[::-1]
    fin = set(R["final_terms"])
    ax.barh(y + 0.27, imp[order], height=0.26, color=INK, label="Akaike weight (53,105 models)")
    ax.barh(y, br[order], height=0.26, color=C7, label="Residual bootstrap")
    ax.barh(y - 0.27, bc[order], height=0.26, color="#6da7ec", label="Subsamples (24 of 30 mixes)")
    ax.set_yticks(y)
    ax.set_yticklabels([("● " if terms[i] in fin else "   ") + dz.label(terms[i]) for i in order], fontsize=7)
    ax.set_xlim(0, 1.02); ax.set_xlabel("Share of weight / resamples including the term")
    ax.set_title("Term support (● = in final model)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=6.5, ncol=2)
    ax.grid(axis="y", visible=False)
    panel(ax, "b")

    ax = axs[2]
    ncv = pd.DataFrame(R["nested_cv"])
    ncv = ncv[ncv.cv == "leave-one-mixture-out"].reset_index(drop=True)
    y = np.arange(len(ncv))[::-1]
    ax.scatter(ncv.predR2_7, y, color=C7, s=28, zorder=3, label="7 d", edgecolor="white", lw=0.7)
    ax.scatter(ncv.predR2_28, y, color=C28, s=28, zorder=3, label="28 d", marker="s", edgecolor="white", lw=0.7)
    for yy, a, b in zip(y, ncv.predR2_7, ncv.predR2_28):
        ax.plot([a, b], [yy, yy], color=GRID, lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("Backward", "Backward elim.") for p in ncv.procedure], fontsize=6.5)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_xlim(min(-0.85, ncv[["predR2_7", "predR2_28"]].min().min() - 0.05), 1.0)
    ax.set_xlabel("Predicted R² for a left-out mixture (ln scale)")
    ax.set_title("Nested cross-validation of procedures", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=7, ncol=2)
    ax.grid(axis="y", visible=False)
    panel(ax, "c")
    fig.tight_layout()
    save(fig, "F02_scale_and_selection")


def fig_gain(R, d):
    g = pd.DataFrame(R["gain_curve"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(g.SS, g.boot_case_lo, g.boot_case_hi, color=C28, alpha=0.08, lw=0, label="95 % case-bootstrap band")
    ax.fill_between(g.SS, g.ratio_lo, g.ratio_hi, color=C28, alpha=0.18, lw=0, label="95 % CI (model)")
    ax.plot(g.SS, g.ratio_est, color=C28, lw=2, label="Model: exp(γ₀ + γ₁·SS + γ₂·SS²)")
    gain = d.f28_mean / d.f7_mean
    for c in dz.CARB_LEVELS:
        s = d.carbonation == c
        ax.scatter(d.SS_pct[s], gain[s], marker=CARB_MK[c], s=24, color=CARB_COL[c], edgecolor="white",
                   linewidth=0.7, zorder=3, label=f"{c} (observed)")
    c = R["contrasts"].get("gain_min_SS")
    if c:
        ax.annotate(f"minimum ×{c['ratio_at_min']:.2f}\nat SS ≈ {c['SS']:.0f} %", xy=(c["SS"], c["ratio_at_min"]),
                    xytext=(c["SS"] + 3, 1.08), fontsize=7, color=INK2, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_yscale("log"); ax.set_yticks([1, 1.5, 2, 3, 4]); ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
    ax.minorticks_off()
    ax.set_xlim(-2, 77)
    ax.set_xlabel("Sodium silicate, SS (%)"); ax.set_ylabel("Strength gain  f28 / f7")
    ax.set_title("7 → 28-day strength gain depends on SS only", loc="left")
    ax.legend(fontsize=6.6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "F03_gain_ratio_vs_SS")


def fig_forest(R):
    t = pd.DataFrame(R["age_specific"]["table"])
    coef = pd.DataFrame(R["final"]["coef"]).set_index("column")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(t))[::-1]
    for j, (_, r) in enumerate(t.iterrows()):
        yy = y[j]
        ax.plot([r.lo7, r.hi7], [yy + 0.2, yy + 0.2], color=C7, lw=1.8, solid_capstyle="round")
        ax.scatter(r.eff7, yy + 0.2, color=C7, s=26, zorder=3, edgecolor="white", lw=0.7)
        ax.plot([r.lo28, r.hi28], [yy - 0.2, yy - 0.2], color=C28, lw=1.8, solid_capstyle="round")
        ax.scatter(r.eff28, yy - 0.2, color=C28, s=26, zorder=3, marker="s", edgecolor="white", lw=0.7)
        # final model: common (level) effect +/- age term
        col = r.column
        if col in coef.index:
            b = coef.loc[col, "coef"]
            if r.in_final_as_age_term:
                ga = coef.loc[col + ".Age", "coef"]
                ax.scatter([b - ga / 2, b + ga / 2], [yy + 0.2, yy - 0.2], marker="|", s=90, color=INK, zorder=4, lw=1.4)
            else:
                ax.scatter([b], [yy], marker="|", s=240, color=INK, zorder=4, lw=1.4)
        ptxt = f"p = {r.p_change:.3f}" if r.p_change >= 0.001 else "p < 0.001"
        ax.text(1.02, yy, ptxt, transform=ax.get_yaxis_transform(), fontsize=6.8, va="center",
                color=INK if r.p_change < 0.05 else MUTED)
    ax.text(1.02, y[0] + 0.85, "7 vs 28 d", transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK2)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(t.label, fontsize=7.5)
    ax.set_xlabel("Effect on ln(strength), coded units (−1…+1 factor range)")
    ax.set_title("Age-specific effects (fully interacted fit) and final-model values", loc="left")
    h = [Line2D([], [], color=C7, marker="o", lw=1.8, ms=4, label="7 d, 95 % CI"),
         Line2D([], [], color=C28, marker="s", lw=1.8, ms=4, label="28 d, 95 % CI"),
         Line2D([], [], color=INK, marker="|", ls="", ms=9, mew=1.4, label="Final model")]
    ax.legend(handles=h, loc="lower left", fontsize=7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "F04_age_specific_effects")


def fig_curves(R, d, final, m):
    ss = np.linspace(0, 75, 151)
    df = 25
    carbs = ["NC", "0.5 h", "1 h", "5 h"]
    fig, axs = plt.subplots(len(carbs), 3, figsize=(7.4, 8.4), sharex=True, sharey=True)
    for i, carb in enumerate(carbs):
        for j, rcf in enumerate((10, 30, 50)):
            ax = axs[i, j]
            for age, col in ((7, C7), (28, C28)):
                pts = pd.DataFrame({"RCF_pct": rcf, "SS_pct": ss, "AB": 0.45, "carbonation": carb, "age": age})
                f, lo, hi = pred_band(m, final, pts, df)
                ax.fill_between(ss, lo, hi, color=col, alpha=0.14, lw=0)
                ax.plot(ss, f, color=col, lw=1.8)
                k = int(np.argmax(f))
                ax.scatter(ss[k], f[k], s=22, facecolor="white", edgecolor=col, lw=1.3, zorder=4)
            near = d[(d.carbonation == carb) & (np.abs(d.RCF_pct - rcf) <= 10)]
            for age, col, mk in ((7, C7, "o"), (28, C28, "s")):
                ax.scatter(near.SS_pct, near[f"f{age}_mean"], s=16, color=col, marker=mk, edgecolor="white",
                           lw=0.6, zorder=3, alpha=0.9)
            if i == 0:
                ax.set_title(f"RCF {rcf} %", loc="center")
            if j == 0:
                ax.set_ylabel(f"{carb}\nStrength (MPa)")
            if i == len(carbs) - 1:
                ax.set_xlabel("SS (%)")
            ax.set_ylim(0, 45)
    h = [Line2D([], [], color=C7, lw=1.8, label="7 d model (95 % CI band)"),
         Line2D([], [], color=C28, lw=1.8, label="28 d model (95 % CI band)"),
         Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="runs within ±10 % RCF (7 d ●, 28 d ■)"),
         Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=5, label="peak of each curve")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "F05_strength_vs_SS_by_age")


def fig_maps(R, d, final, m):
    Rg = np.linspace(10, 50, 81); Sg = np.linspace(0, 75, 151)
    RR, SS = np.meshgrid(Rg, Sg)
    fig, axs = plt.subplots(2, 4, figsize=(10.4, 5.2), sharex=True, sharey=True)
    vmax = 34
    levels = np.arange(0, 36, 2)
    opt = {(o["age"], o["carbonation"]): o for o in R["optimum"]}
    for i, age in enumerate((7, 28)):
        for j, carb in enumerate(dz.CARB_LEVELS):
            ax = axs[i, j]
            pts = pd.DataFrame({"RCF_pct": RR.ravel(), "SS_pct": SS.ravel(), "AB": 0.45, "carbonation": carb, "age": age})
            Z = np.exp(mt.rows(final, pts) @ m.beta).reshape(RR.shape)
            cf = ax.contourf(RR, SS, Z, levels=levels, cmap=SEQ, vmin=0, vmax=vmax, extend="max")
            cs = ax.contour(RR, SS, Z, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.8)
            ax.clabel(cs, fmt="%d", fontsize=6, colors="white")
            s = d[d.carbonation == carb]
            ax.scatter(s.RCF_pct, s.SS_pct, s=14, color="white", edgecolor=INK, lw=0.6, zorder=3, clip_on=False)
            o = opt[(age, carb)]
            ax.scatter(o["RCF"], o["SS"], s=70, marker="*", color="#eda100", edgecolor=INK, lw=0.6, zorder=4, clip_on=False)
            ax.grid(False)
            if i == 0:
                ax.set_title(carb)
            if j == 0:
                ax.set_ylabel(f"{age} d\nSS (%)")
            if i == 1:
                ax.set_xlabel("RCF (%)")
    cb = fig.colorbar(cf, ax=axs, shrink=0.85, pad=0.015)
    cb.set_label("Predicted strength (MPa, median)")
    cb.outline.set_visible(False)
    fig.suptitle("Predicted strength over RCF × SS (A/B not in model). ○ runs at that carbonation level, ★ maximum",
                 x=0.02, ha="left", fontsize=8.5, color=INK2)
    save(fig, "F06_response_surfaces")


def fig_effects(R):
    ce = pd.DataFrame(R["carb_effect"]); re = pd.DataFrame(R["rcf_effect"])
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4))
    fig.suptitle("Ratios are identical at 7 and 28 d in the final model (no age interaction for these terms); bands = 95 % CI",
                 x=0.01, ha="left", fontsize=7.5, color=INK2)
    ax = axs[0]
    ss = np.linspace(0, 75, 151)
    # carbonated / NC ratio: exp(Carb_l + b_KA A + b_KD D) - drawn from exact contrasts at RCF 10/30/50 for 1 h
    sub = ce[(ce.carbonation == "1 h") & (ce.age == 7)]
    for rcf, alpha in ((10, 0.45), (30, 0.7), (50, 1.0)):
        s = sub[sub.RCF == rcf]
        ax.plot(s.SS, s.ratio, color=CARB_COL["1 h"], alpha=alpha, lw=1.8, marker="o", ms=3.5)
        ax.fill_between(s.SS, s.lo, s.hi, color=CARB_COL["1 h"], alpha=0.07 * alpha, lw=0)
        ax.text(s.SS.values[-1] + 1.5, s.ratio.values[-1], f"RCF {rcf}", fontsize=6.8, va="center", color=INK2)
    ax.axhline(1, color=AXIS, lw=0.9)
    ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 0.8, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.5", "2", "3"])
    ax.minorticks_off(); ax.set_xlim(-2, 88)
    ax.set_xlabel("SS (%)"); ax.set_ylabel("Strength ratio, 1 h carbonated / NC")
    ax.set_title("Carbonated (1 h) vs uncarbonated RCF", loc="left")
    panel(ax, "a")
    ax = axs[1]
    sub = re[re.age == 7]
    for carb in ("NC", "1 h"):
        s = sub[sub.carbonation == carb]
        ax.plot(s.SS, s.ratio, color=CARB_COL[carb], lw=1.8, marker=CARB_MK[carb], ms=4, label=carb if carb == "NC" else "carbonated (1 h)")
        ax.fill_between(s.SS, s.lo, s.hi, color=CARB_COL[carb], alpha=0.1, lw=0)
    ax.axhline(1, color=AXIS, lw=0.9)
    ax.set_yscale("log"); ax.set_yticks([0.6, 0.8, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlabel("SS (%)"); ax.set_ylabel("Strength ratio, RCF 50 % / RCF 10 %")
    ax.set_title("RCF 50 % vs 10 %", loc="left")
    ax.legend(loc="upper right")
    panel(ax, "b")
    fig.tight_layout()
    save(fig, "F07_carbonation_and_RCF_effects")


def fig_obs_pred(R, d):
    t = pd.DataFrame(R["diagnostics"]["table"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for ax, (k7, k28, title) in zip(axs, (("fit7", "fit28", "Fitted (all 30 mixes)"),
                                          ("cv7", "cv28", "Leave-one-mixture-out prediction"))):
        ax.plot([2.5, 45], [2.5, 45], color=AXIS, lw=0.9)
        ax.scatter(t.obs7, t[k7], color=C7, s=22, edgecolor="white", lw=0.6, label="7 d", zorder=3)
        ax.scatter(t.obs28, t[k28], color=C28, s=22, marker="s", edgecolor="white", lw=0.6, label="28 d", zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ticks = [3, 5, 10, 20, 40]
        ax.set_xticks(ticks); ax.set_xticklabels(map(str, ticks)); ax.set_yticks(ticks); ax.set_yticklabels(map(str, ticks))
        ax.minorticks_off(); ax.set_xlim(2.5, 45); ax.set_ylim(2.5, 45)
        ax.set_xlabel("Observed (MPa)"); ax.set_title(title, loc="left")
        v = R["validation"]["fit" if k7 == "fit7" else "mix"]
        ax.text(0.97, 0.04, f"7 d: R² {'pred ' if k7 != 'fit7' else ''}{v['predR2_7']:.2f}, RMSE {v['rmse_MPa_7']:.1f} MPa\n"
                f"28 d: R² {'pred ' if k7 != 'fit7' else ''}{v['predR2_28']:.2f}, RMSE {v['rmse_MPa_28']:.1f} MPa",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
    lab28 = set(t.mix.values[np.argsort(-np.abs(np.log(t.obs28 / t.cv28)).values)[:4]])
    for _, r in t.iterrows():
        if r.mix in lab28:
            axs[1].annotate(str(int(r.mix)), (r.obs28, r.cv28), xytext=(3, -8), textcoords="offset points", fontsize=6.3, color=INK2)
    axs[0].set_ylabel("Model (MPa, median)")
    axs[0].legend(loc="upper left")
    panel(axs[0], "a"); panel(axs[1], "b")
    fig.tight_layout()
    save(fig, "F08_observed_vs_predicted")


def fig_diagnostics(R, d):
    t = pd.DataFrame(R["diagnostics"]["table"])
    D = R["diagnostics"]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    ax = axs[0, 0]
    ax.scatter(np.log(t.fit7), t.white7, color=C7, s=20, edgecolor="white", lw=0.6, label="7 d")
    ax.scatter(np.log(t.fit28), t.white28, color=C28, s=20, marker="s", edgecolor="white", lw=0.6, label="28 d | 7 d")
    ax.axhline(0, color=AXIS, lw=0.9)
    for yv in (-2, 2):
        ax.axhline(yv, color=GRID, lw=0.9)
    ax.set_xlabel("Fitted ln(strength)"); ax.set_ylabel("Whitened residual")
    ax.set_title(f"Residuals vs fitted (Breusch–Pagan p = {D['bp_fitted_age_p']:.2f})", loc="left")
    ax.legend(loc="lower right"); panel(ax, "a")
    ax = axs[0, 1]
    w = np.concatenate([t.white7, t.white28])
    ages = np.array([7] * 30 + [28] * 30)
    o = np.argsort(w)
    q = stats.norm.ppf((np.arange(1, 61) - 0.375) / 60.25)
    ax.plot([-2.6, 2.6], [-2.6, 2.6], color=AXIS, lw=0.9)
    cols = np.where(ages[o] == 7, C7, C28)
    ax.scatter(q, w[o], c=cols, s=18, edgecolor="white", lw=0.5)
    ax.set_xlabel("Normal quantile"); ax.set_ylabel("Whitened residual")
    ax.set_title(f"Normal Q–Q (Shapiro–Wilk p = {D['shapiro_p']:.2f})", loc="left"); panel(ax, "b")
    ax = axs[1, 0]
    crit = stats.t.ppf(1 - 0.025 / 60, D["df_deleted_t"])
    ax.vlines(t.mix - 0.18, 0, t.t_del7, color=C7, lw=2)
    ax.vlines(t.mix + 0.18, 0, t.t_del28, color=C28, lw=2)
    for yv in (-crit, crit):
        ax.axhline(yv, color=MUTED, lw=0.9)
    ax.text(30.6, crit, "Bonferroni 5 %", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("Mix"); ax.set_ylabel("Deleted studentized residual")
    ax.set_title("Outliers (mixture left out of the fit)", loc="left")
    k = np.argmax(np.abs(t.t_del28.values))
    ax.annotate(f"mix {int(t.mix[k])}, 28 d", (t.mix[k] + 0.18, t.t_del28[k]), xytext=(6, 0), textcoords="offset points",
                fontsize=6.5, color=INK2, va="center")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30]); ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], color=C7, lw=2, label="7 d"), Line2D([], [], color=C28, lw=2, label="28 d")],
              loc="lower left", ncol=2)
    panel(ax, "c")
    ax = axs[1, 1]
    ax.vlines(t.mix, 0, t.cook, color=INK2, lw=2.2)
    ax.axhline(4 / 30, color=MUTED, lw=0.9)
    ax.text(30.6, 4 / 30, "4/n", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    k = int(np.argmax(t.cook.values))
    ax.annotate(f"mix {int(t.mix[k])}", (t.mix[k], t.cook[k]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xlabel("Mix"); ax.set_ylabel("Cook's distance (whole mixture)")
    ax.set_title("Influence of each mixture on the coefficients", loc="left")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30]); ax.grid(axis="x", visible=False)
    panel(ax, "d")
    fig.tight_layout()
    save(fig, "F09_residual_diagnostics")


def fig_robustness(R):
    rb = pd.DataFrame(R["robustness"])
    def shorten(a):
        rules = [("Primary", "PRIMARY: final model"), ("Same terms, CS", "CS covariance, same terms"),
                 ("AICc-best model under CS", "AICc-best model under CS"), ("Same terms, carbonation coded on/off", "Carbonation on/off, same terms"),
                 ("Same terms, carbonation coded log2", "Carbonation log-duration, same terms"),
                 ("Same terms, carbonation coded numeric", "Carbonation numeric duration, same terms"),
                 ("AICc-best model with log-duration", "AICc-best, log-duration coding"), ("Same terms, raw MPa", "Raw MPa scale, same terms"),
                 ("Same terms, Box-Cox lambda = -0.5", "Box–Cox λ = −0.5, same terms"), ("Same terms, Box-Cox lambda = +0.5", "Box–Cox λ = +0.5, same terms"),
                 ("Selection repeated", "Re-selected on λ = −0.5 scale"), ("Same terms, single-specimen", "n = 1 means down-weighted"),
                 ("Same terms, Huber", "Huber robust fit"), ("Final terms, every term", "Age-specific coefficients (all × Age)"),
                 ("Backward elimination", "Backward elimination α = 0.10"), ("Separate per-age", "Separate per-age OLS models")]
        for k, v in rules:
            if a.startswith(k):
                return v
        if "excluded" in a:
            return a.split(",")[1].strip().split("(")[0].strip().capitalize() + " (" + a.split("(")[1].split(")")[0] + ")"
        return a[:45]
    short = [shorten(a) for a in rb.analysis]
    y = np.arange(len(rb))[::-1]
    cols = [("gain_SS0", "Gain f28/f7\nat SS 0 %", None), ("gain_SS75", "Gain f28/f7\nat SS 75 %", None),
            ("1h/NC ratio RCF50 SS0 28d", "1 h / NC at 28 d\nRCF 50, SS 0", 1), ("1h/NC ratio RCF10 SS75 28d", "1 h / NC at 28 d\nRCF 10, SS 75", 1),
            ("RCF50/10 ratio 1h SS0 28d", "RCF 50 / 10 at 28 d\n1 h, SS 0", 1)]
    fig, axs = plt.subplots(1, len(cols), figsize=(11.0, 5.2), sharey=True)
    for ax, (c, lab, ref) in zip(axs, cols):
        v = rb[c].values
        ax.scatter(v, y, s=22, color=[INK if i == 0 else C7 for i in range(len(v))], zorder=3, edgecolor="white", lw=0.6)
        ax.axvline(v[0], color=INK, lw=0.8, alpha=0.4)
        if ref:
            ax.axvline(ref, color=AXIS, lw=0.9)
        ax.set_title(lab, fontsize=7.5, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xscale("log"); ax.minorticks_off()
        lo, hi = np.nanmin(v), np.nanmax(v)
        ticks = [t for t in (0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4) if lo * 0.85 <= t <= hi * 1.15]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks])
    axs[0].set_yticks(y); axs[0].set_yticklabels(short, fontsize=7)
    fig.suptitle("Robustness: key model quantities under alternative analyses (black = primary)", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, "F10_robustness")


def fig_bootstrap(R):
    names = R["final"]["names"]
    coef = pd.DataFrame(R["final"]["coef"])
    B = R["bootstrap"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    y = np.arange(len(names))[::-1]
    for j, nm in enumerate(names):
        r = coef.iloc[j]
        ax.plot([r.lo, r.hi], [y[j] + 0.2] * 2, color=INK, lw=1.8, solid_capstyle="round")
        lo, hi = B["resid"]["beta_ci"][nm]
        ax.plot([lo, hi], [y[j]] * 2, color=C7, lw=1.8, solid_capstyle="round")
        lo, hi = B["case"]["beta_ci"][nm]
        ax.plot([lo, hi], [y[j] - 0.2] * 2, color="#6da7ec", lw=1.8, solid_capstyle="round")
        ax.scatter(r.coef, y[j] + 0.2, s=18, color=INK, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y)
    lab = [n.replace("Carb[", "Carb [").replace(".Age", " × Age") for n in names]
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("Coefficient (coded units, ln scale)")
    ax.set_title("Final-model coefficients: 95 % intervals", loc="left")
    h = [Line2D([], [], color=INK, lw=1.8, marker="o", ms=3.5, label="Wald (Satterthwaite df)"),
         Line2D([], [], color=C7, lw=1.8, label=f"Residual bootstrap (n = {B['resid']['n']})"),
         Line2D([], [], color="#6da7ec", lw=1.8, label=f"Case bootstrap, stratified (n = {B['case']['n']})")]
    ax.legend(handles=h, loc="lower right", fontsize=6.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "F11_coefficient_intervals")


def main():
    R, d, final, m = load()
    fig_data(R, d)
    fig_scale_selection(R)
    fig_gain(R, d)
    fig_forest(R)
    fig_curves(R, d, final, m)
    fig_maps(R, d, final, m)
    fig_effects(R)
    fig_obs_pred(R, d)
    fig_diagnostics(R, d)
    fig_robustness(R)
    fig_bootstrap(R)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
