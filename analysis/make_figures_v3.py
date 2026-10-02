"""
Figures for the revised-data (v3) bivariate model: PNG (300 dpi) + vector PDF in
results/v3/figures.  Every curve is recomputed from the final model refitted to the
data with the terms stored in results/v3/v3_results.json.

Run:  python analysis/make_figures_v3.py
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

import bivariate as bv
import design as dz

RES = os.path.join(dz.ROOT, "results", "v3")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- style (palettes validated with
# the dataviz validator: age pair adjacent; carbonation set all-pairs, CVD dE >= 16; SS/RCF ordinal ramp)
C7, C28 = "#2a78d6", "#eb6834"                         # curing age: 7 d / 28 d
AGE_COL = {7: C7, 28: C28}
CARB_COL = {"NC": "#4a3aa7", "0.5 h": "#008300", "1 h": "#eda100", "5 h": "#e87ba4"}
CARB_MK = {"NC": "o", "0.5 h": "^", "1 h": "s", "5 h": "D"}       # secondary encoding (two hues < 3:1)
RAMP3 = ["#6da7ec", "#2a78d6", "#104281"]               # ordinal levels (low -> high)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                 "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF})

REF = {"RCF_pct": 30.0, "SS_pct": 37.5, "AB": 0.45, "carbonation": "NC"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def panel(ax, letter, x=-0.12):
    ax.text(x, 1.03, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)


def grid(**kw):
    base = dict(REF)
    base.update({k: v for k, v in kw.items() if np.ndim(v) == 0})
    var = [(k, np.asarray(v)) for k, v in kw.items() if np.ndim(v) > 0]
    n = len(var[0][1]) if var else 1
    out = pd.DataFrame({k: [v] * n for k, v in base.items()})
    for k, v in var:
        out[k] = v
    return out


class Model:
    def __init__(self):
        with open(os.path.join(RES, "v3_results.json")) as f:
            self.R = json.load(f)
        self.d = dz.load(os.path.join(dz.ROOT, self.R["meta"]["data"]))
        self.t7, self.t28 = self.R["final_terms"]["7"], self.R["final_terms"]["28"]
        self.m, self.X, self.names = bv.fit(self.d, self.t7, self.t28)
        # one Satterthwaite df per age (at the reference point) for pointwise curve intervals
        self.df = {a: self.m.satterthwaite_df(bv.rows(self.t7, self.t28, grid(age=a))[0]) for a in bv.AGES}

    def curve(self, pts, age, level=0.95):
        p = pts.copy()
        p["age"] = age
        X = bv.rows(self.t7, self.t28, p)
        f = X @ self.m.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", X, self.m.cov_beta, X))
        q = stats.t.ppf(0.5 + level / 2, self.df[age])
        return np.exp(f), np.exp(f - q * se), np.exp(f + q * se)

    def gain(self, pts, level=0.95):
        L = bv.gain_rows(self.t7, self.t28, pts)
        f = L @ self.m.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", L, self.m.cov_beta, L))
        q = stats.t.ppf(0.5 + level / 2, min(self.df.values()))
        return np.exp(f), np.exp(f - q * se), np.exp(f + q * se)


# ============================================================================ G01 data revision
def g01(M):
    rv = pd.read_csv(os.path.join(TAB, "V00_revision_log_v2_to_v3.csv"))
    rv = rv[rv.change_MPa.abs() > 0.011].copy()
    pe2, pe3 = M.R["revision"]["pure_error_v2"], M.R["revision"]["pure_error_v3"]
    n7, n28 = (rv.age_d == 7).sum(), (rv.age_d == 28).sum()
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 1.3 + 0.27 * (n7 + n28) + 0.9),
                             gridspec_kw={"height_ratios": [n7 + 1.2, n28 + 1.2], "hspace": 0.55})
    for ax, age, letter in zip(axes, (7, 28), "ab"):
        r = rv[rv.age_d == age].sort_values("mix", ascending=False).reset_index(drop=True)
        y = np.arange(len(r))
        for i, row in r.iterrows():
            ax.plot([row.mean_v2, row.mean_v3], [i, i], color=AXIS, lw=2.0, zorder=1)
            ax.annotate("", xy=(row.mean_v3, i), xytext=(row.mean_v2, i),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=4, shrinkB=5), zorder=2)
            ax.scatter(row.mean_v2, i, s=34, facecolor=SURF, edgecolor=INK2, lw=1.2, zorder=3)
            mk = "D" if row.kind == "mean shifted, SD unchanged" else "o"
            ax.scatter(row.mean_v3, i, s=40 if mk == "o" else 34, marker=mk, color=AGE_COL[age],
                       edgecolor=SURF, lw=1.5, zorder=4)
            ax.text(max(row.mean_v2, row.mean_v3) + 0.5, i, f"{row.change_MPa:+.2f}", va="center",
                    fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Mix {int(m)}" + (f" ({g})" if isinstance(g, str) and g else "")
                            for m, g in zip(r.mix, r.replicate_group)])
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(r) - 0.3)
        a = str(age)
        ax.set_title(f"{age}-day means changed: {len(r)}   ·   replicate (pure-error) SD "
                     f"{pe2[a + '_MPa']['sd']:.2f} → {pe3[a + '_MPa']['sd']:.2f} MPa", loc="left", fontsize=8.5)
        ax.set_xlabel(f"{age}-day strength (MPa)")
        panel(ax, letter, x=-0.2)
    h = [Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=6, label="previous data (v2)"),
         Line2D([], [], marker="o", ls="", color=INK2, ms=6, label="revised data (v3)"),
         Line2D([], [], marker="D", ls="", color=INK2, ms=5.5, label="v3: mean moved, SD unchanged")]
    axes[0].legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.22), ncol=3, fontsize=7.2)
    save(fig, "G01_data_revision")


# ============================================================================ G02 strategy comparison
def g02(M):
    cv = pd.read_csv(os.path.join(TAB, "V04_strategy_nested_cv.csv"))
    keep = ["Bivariate, per-age AICc selection (final procedure)",
            "Shared-effects paired model, AICc (UN)",
            "Shared-effects paired model, AICc (CS + UN; earlier protocol)",
            "Full candidate model at both ages (no selection)",
            "SS + SS² only at both ages (reference)"]
    short = {keep[0]: "Age-specific terms (bivariate) — chosen",
             keep[1]: "Shared effects, AICc (UN)",
             keep[2]: "Shared effects, AICc (CS + UN, earlier protocol)",
             keep[3]: "Full candidate model, no selection",
             keep[4]: "SS + SS² only (reference)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, scheme, letter in zip(axes, ("leave one mixture out", "leave one design point out"), "ab"):
        s = cv[(cv.cv == scheme) & cv.procedure.isin(keep)].set_index("procedure").loc[keep]
        y = np.arange(len(keep))[::-1]
        h = 0.34
        for off, age, col in ((h / 2 + 0.02, 7, C7), (-h / 2 - 0.02, 28, C28)):
            v = s[f"Q2_ln_{age}"].values
            ax.barh(y + off, v, height=h, color=col, label=f"{age} d", zorder=2)
            for yy, vv in zip(y + off, v):
                ax.text(vv + (0.03 if vv >= 0 else -0.03), yy, f"{0.0 if abs(vv) < 0.005 else vv:.2f}", va="center",
                        ha="left" if vv >= 0 else "right", fontsize=6.8, color=INK2)
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        ax.set_xlim(-1.2, 1.15)
        ax.set_yticks(y)
        ax.set_yticklabels([short[k] for k in keep])
        if letter == "a":
            ax.get_yticklabels()[0].set_fontweight("bold")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{letter}   {scheme.capitalize()}", loc="left")
    fig.supxlabel("Predicted R² (ln scale); the model selection is repeated inside every cross-validation fold",
                  fontsize=8.5, y=0.02)
    axes[1].legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.08))
    fig.tight_layout()
    save(fig, "G02_strategy_comparison")


# ============================================================================ G03 main trends
def g03(M):
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.75), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 0.9]})
    specs = [("SS_pct", np.linspace(0, 75, 151), "SS (%)", "a"),
             ("RCF_pct", np.linspace(10, 50, 121), "RCF (%)", "b"),
             ("AB", np.linspace(0.42, 0.48, 61), "A/B", "c")]
    for ax, (var, xs, lab, letter) in zip(axes, specs):
        for age in bv.AGES:
            med, lo, hi = M.curve(grid(**{var: xs}), age)
            ax.fill_between(xs, lo, hi, color=AGE_COL[age], alpha=0.12, lw=0)
            ax.plot(xs, med, color=AGE_COL[age], label=f"{age} d")
            ax.text(xs[-1], med[-1], f" {age} d", color=INK, fontsize=7, va="center")
        ax.set_xlabel(lab)
        ax.set_ylim(0, None)
        panel(ax, letter, x=-0.16)
    axes[0].set_ylabel("Predicted strength (MPa)")
    axes[2].set_xticks([0.42, 0.44, 0.46, 0.48])
    ax = axes[3]
    xs = np.arange(4)
    for k, age in enumerate(bv.AGES):
        med, lo, hi = M.curve(grid(carbonation=np.array(dz.CARB_LEVELS)), age)
        xx = xs + (-0.12 if age == 7 else 0.12)
        ax.errorbar(xx, med, yerr=[med - lo, hi - med], fmt="none", ecolor=AGE_COL[age], elinewidth=1.4, capsize=0)
        ax.scatter(xx, med, s=40, color=AGE_COL[age], edgecolor=SURF, lw=1.5, zorder=3, label=f"{age} d")
    ax.set_xticks(xs)
    ax.set_xticklabels(dz.CARB_LEVELS)
    ax.set_xlabel("Carbonation of RCF")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    panel(ax, "d", x=-0.16)
    axes[3].legend(loc="lower right", fontsize=7)
    fig.suptitle("One factor at a time; the others at RCF 30 %, SS 37.5 %, A/B 0.45, uncarbonated (NC). Bands and bars: 95 % CI of the mean",
                 fontsize=8, color=INK2, y=1.02, x=0.01, ha="left", fontweight="normal")
    fig.tight_layout()
    save(fig, "G03_main_trends_7d_vs_28d")


# ============================================================================ G04 carbonation interactions
def g04(M):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    specs = [("SS_pct", np.linspace(0, 75, 151), "SS (%)", {"RCF_pct": 30.0}, np.arange(0, 76, 15)),
             ("RCF_pct", np.linspace(10, 50, 121), "RCF (%)", {"SS_pct": 37.5}, np.arange(10, 51, 10))]
    letters = iter("abcd")
    for r, age in enumerate(bv.AGES):
        ymax = 0
        for c, (var, xs, lab, fixed, mk_at) in enumerate(specs):
            ax = axes[r, c]
            step = mk_at[1] - mk_at[0]
            for j, carb in enumerate(dz.CARB_LEVELS):
                med, _, _ = M.curve(grid(**fixed, **{var: xs, "carbonation": carb}), age)
                ax.plot(xs, med, color=CARB_COL[carb], zorder=2)
                at = (mk_at[:-1] + j * step / 4).astype(float)      # staggered: coinciding lines stay readable
                mk, _, _ = M.curve(grid(**fixed, **{var: at, "carbonation": carb}), age)
                ax.scatter(at, mk, marker=CARB_MK[carb], s=30, color=CARB_COL[carb], edgecolor=SURF, lw=1.2, zorder=3)
                ymax = max(ymax, med.max())
            if r == 1:
                ax.set_xlabel(lab)
            fixed_txt = "RCF 30 %" if var == "SS_pct" else "SS 37.5 %"
            ax.set_title(f"{age} d — strength vs {lab.split()[0]} ({fixed_txt}, A/B 0.45)", loc="left", fontsize=8.5)
            panel(ax, next(letters), x=-0.15)
        for c in range(2):
            axes[r, c].set_ylim(0, ymax * 1.08)
        axes[r, 0].set_ylabel(f"Predicted {age}-day strength (MPa)")
    h = [Line2D([], [], color=CARB_COL[c], marker=CARB_MK[c], ms=5.5, mec=SURF, label=c) for c in dz.CARB_LEVELS]
    axes[0, 0].legend(handles=h, title="Carbonation of RCF", title_fontsize=7.5, ncol=4, loc="lower left",
                      bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    save(fig, "G04_carbonation_interactions")


# ============================================================================ G05 response surfaces
def g05(M):
    rcf = np.linspace(10, 50, 81)
    ss = np.linspace(0, 75, 151)
    RR, SS = np.meshgrid(rcf, ss)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.32})
    d = M.d
    letters = iter("abcd")
    for r, age in enumerate(bv.AGES):
        Z = {}
        for carb in ("NC", "1 h"):
            med, _, _ = M.curve(grid(RCF_pct=RR.ravel(), SS_pct=SS.ravel(), carbonation=carb), age)
            Z[carb] = med.reshape(RR.shape)
        vmin = min(z.min() for z in Z.values())
        vmax = max(z.max() for z in Z.values())
        levels = matplotlib.ticker.MaxNLocator(nbins=12).tick_values(vmin, vmax)
        for c, carb in enumerate(("NC", "1 h")):
            ax = axes[r, c]
            cf = ax.contourf(RR, SS, Z[carb], levels=levels, cmap=SEQ)
            cs = ax.contour(RR, SS, Z[carb], levels=levels[::2], colors=[SURF], linewidths=0.6)
            ax.clabel(cs, fmt="%.0f", fontsize=6.5, colors=INK)
            k = np.unravel_index(np.argmax(Z[carb]), Z[carb].shape)
            ax.scatter(RR[k], SS[k], marker="*", s=130, color=SURF, edgecolor=INK, lw=0.8, zorder=5, clip_on=False)
            pts = d[d.carbonation == "NC"] if carb == "NC" else d[d.carbonation != "NC"]
            for cl, g in pts.groupby("carbonation", observed=True):
                ax.scatter(g.RCF_pct, g.SS_pct, marker=CARB_MK[cl], s=26, facecolor=SURF, edgecolor=INK, lw=0.8,
                           zorder=4, clip_on=False)
            ax.grid(False)
            lab = "uncarbonated RCF (NC)" if carb == "NC" else "carbonated RCF (1 h)"
            ax.set_title(f"{next(letters)}   {age} d — {lab}\n      ★ max {Z[carb][k]:.1f} MPa (RCF {RR[k]:.0f} %, SS {SS[k]:.0f} %)",
                         loc="left", fontsize=8.2)
            if r == 1:
                ax.set_xlabel("RCF (%)")
            if c == 0:
                ax.set_ylabel("SS (%)")
        cb = fig.colorbar(cf, ax=axes[r, :].tolist(), shrink=0.92, pad=0.02)
        cb.set_label(f"Predicted {age}-day strength (MPa)")
        cb.outline.set_visible(False)
    h = [Line2D([], [], marker=CARB_MK[c], ls="", mfc=SURF, mec=INK, ms=5, label=f"mix, {c}") for c in dz.CARB_LEVELS]
    axes[0, 0].legend(handles=h, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.2), fontsize=7)
    fig.text(0.01, -0.01, "A/B fixed at 0.45. Open symbols: the mixtures of that carbonation state (left: NC; right: all "
             "carbonated levels). ★ maximum within the panel.", fontsize=7, color=INK2)
    save(fig, "G05_response_surfaces")


# ============================================================================ G06 strength gain
def g06(M):
    d = M.d.copy()
    d["gain"] = d.f28_mean / d.f7_mean
    ss = np.linspace(0, 75, 151)
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
    for ax, carb, letter in zip(axes, dz.CARB_LEVELS, "abcd"):
        for rcf, col in zip((10, 30, 50), RAMP3):
            med, lo, hi = M.gain(grid(SS_pct=ss, RCF_pct=float(rcf), carbonation=carb))
            if rcf == 30:
                ax.fill_between(ss, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(ss, med, color=col, label=f"RCF {rcf} %")
        ax.axhline(1, color=INK2, lw=0.8, zorder=1)
        g = d[d.carbonation == carb]
        ax.scatter(g.SS_pct, g.gain, s=30, facecolor=SURF, edgecolor=INK, lw=1.0, zorder=4, clip_on=False)
        lab = g[(g.gain < 1.05) | (g.gain > 3.6)].sort_values("SS_pct")
        clusters = []
        for _, r in lab.iterrows():                      # one label per cluster of coinciding points
            if clusters and abs(r.SS_pct - clusters[-1][0]) < 3 and abs(np.log(r.gain / clusters[-1][1])) < 0.2:
                clusters[-1][2].append(int(r.mix))
            else:
                clusters.append([r.SS_pct, r.gain, [int(r.mix)]])
        for ssv, gv, mixes in clusters:
            ax.annotate(", ".join(map(str, sorted(mixes))), (ssv, gv), xytext=(5, -9 if gv < 1.05 else 3),
                        textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4, 6])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0.5, 6.5)
        ax.set_xlabel("SS (%)")
        ax.set_title(f"{carb}", loc="left")
        panel(ax, letter, x=-0.12)
    axes[0].set_ylabel("Strength gain  f28 / f7")
    h = [Line2D([], [], color=c, label=f"RCF {r} %") for r, c in zip((10, 30, 50), RAMP3)] + \
        [Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK, ms=5.5, label="observed mixture")]
    fig.legend(handles=h, ncol=4, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    fig.text(0.01, 0.0, "A/B 0.45. Band: 95 % CI at RCF 30 %. Gain below 1 means the 28-day strength is lower than the "
             "7-day strength. Numbers: mixtures with gain < 1.05 or > 3.6.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, "G06_strength_gain")


# ============================================================================ G07 effects 7 d vs 28 d
def g07(M):
    cmp_ = pd.read_csv(os.path.join(TAB, "V11_effect_7d_vs_28d.csv")).set_index("term")
    coef = pd.read_csv(os.path.join(TAB, "V06_final_coefficients_coded.csv"))
    rows = [("D", "SS"), ("D2", "SS²"), ("A", "RCF"), ("A2", "RCF²"), ("C", "A/B"), ("C2", "(A/B)²"),
            ("AD", "RCF × SS"), ("KA", "Carbonated × RCF"), ("KD", "Carbonated × SS"),
            ("Carb[0.5 h]", "Carbonation 0.5 h vs NC"), ("Carb[1 h]", "Carbonation 1 h vs NC"),
            ("Carb[5 h]", "Carbonation 5 h vs NC")]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(rows))[::-1]
    for yy, (key, lab) in zip(y, rows):
        for age, off in ((7, 0.17), (28, -0.17)):
            if key.startswith("Carb["):
                r = coef[(coef.age_d == age) & (coef.column == key)].iloc[0]
                est, lo, hi, inmod = r.coef, r.lo, r.hi, True
            else:
                r = cmp_.loc[key]
                est, lo, hi = r[f"coef_{age}d"], r[f"lo_{age}d"], r[f"hi_{age}d"]
                inmod = bool(r[f"in_{age}d_model"])
            ax.plot([lo, hi], [yy + off] * 2, color=AGE_COL[age], lw=2.0, zorder=2, alpha=1 if inmod else 0.55)
            ax.scatter(est, yy + off, s=40, zorder=3, color=AGE_COL[age] if inmod else SURF,
                       edgecolor=AGE_COL[age] if not inmod else SURF, lw=1.5)
        pkey = "Carb" if key.startswith("Carb[") else key
        if not key.startswith("Carb[") or key == "Carb[0.5 h]":
            p = cmp_.loc[pkey, "p_diff"]
            txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
            yt = yy if not key.startswith("Carb[") else y[-3] - 1.0
            ax.text(1.02, yt, txt + (" (3 levels)" if key.startswith("Carb[") else ""), transform=ax.get_yaxis_transform(),
                    fontsize=7, color=INK if p < 0.05 else INK2, va="center", fontweight="bold" if p < 0.05 else "normal")
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Coefficient on ln(strength), per coded unit (±1 = ends of the design range), 95 % CI")
    ax.text(1.02, 1.0, "7 d vs 28 d\ndifferent?", transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom", fontweight="bold")
    h = [Line2D([], [], color=C7, marker="o", ms=6, mec=SURF, label="7 d (in final model)"),
         Line2D([], [], color=C28, marker="o", ms=6, mec=SURF, label="28 d (in final model)"),
         Line2D([], [], color=INK2, marker="o", ms=6, mfc=SURF, mec=INK2, alpha=0.7, label="hollow: not selected at that age")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7)
    save(fig, "G07_effects_7d_vs_28d")


# ============================================================================ G08 validation
def g08(M):
    dg = pd.read_csv(os.path.join(TAB, "V09_diagnostics_per_mix.csv"))
    pr = pd.read_csv(os.path.join(TAB, "V10_predictions_all_mixes.csv"))
    fs = M.R["fit_stats"]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.7))
    lim = (0, 40)
    for age in bv.AGES:
        p = pr[pr.age_d == age]
        axes[0].scatter(p["median"], p.observed, s=26, color=AGE_COL[age], edgecolor=SURF, lw=1.2, zorder=3, label=f"{age} d")
        axes[1].scatter(p.loo_pred, p.observed, s=26, color=AGE_COL[age], edgecolor=SURF, lw=1.2, zorder=3, label=f"{age} d")
        res = np.log(dg[f"f{age}_mean"]) - np.log(dg[f"fit{age}"])
        axes[2].scatter(np.log(dg[f"fit{age}"]), res, s=26, color=AGE_COL[age], edgecolor=SURF, lw=1.2, zorder=3)
        t = np.sort(dg[f"t_del{age}"].values)
        q = stats.norm.ppf((np.arange(1, len(t) + 1) - 0.5) / len(t))
        axes[3].scatter(q, t, s=26, color=AGE_COL[age], edgecolor=SURF, lw=1.2, zorder=3)
    for ax in axes[:2]:
        ax.plot(lim, lim, color=AXIS, lw=1.0, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_ylabel("Observed (MPa)")
    axes[0].set_xlabel("Fitted (MPa)")
    axes[1].set_xlabel("Predicted with that mixture left out (MPa)")
    axes[0].set_title(f"R²(ln) {fs['7']['R2_ln']:.2f} (7 d) · {fs['28']['R2_ln']:.2f} (28 d)", loc="left", fontsize=8)
    axes[1].set_title(f"Pred. R²(ln) {fs['7']['predR2_ln_LOO']:.2f} · {fs['28']['predR2_ln_LOO']:.2f}", loc="left", fontsize=8)
    axes[2].axhline(0, color=AXIS, lw=1.0, zorder=1)
    axes[2].set_xlabel("Fitted ln(strength)")
    axes[2].set_ylabel("Residual (ln)")
    axes[2].set_title("Residuals vs fitted", loc="left", fontsize=8)
    axes[3].plot([-2.5, 2.5], [-2.5, 2.5], color=AXIS, lw=1.0, zorder=1)
    axes[3].set_xlabel("Normal quantile")
    axes[3].set_ylabel("Studentized deleted residual")
    axes[3].set_title("Normal Q–Q", loc="left", fontsize=8)
    axes[0].legend(loc="upper left")
    for ax, l in zip(axes, "abcd"):
        panel(ax, l, x=-0.2)
    fig.tight_layout()
    save(fig, "G08_validation_diagnostics")


# ============================================================================ G09 design-point influence at 28 d
def g09(M):
    ldp = pd.read_csv(os.path.join(TAB, "V22_leave_design_point_out.csv"))
    ldp = ldp.sort_values("err_ln28").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    y = np.arange(len(ldp))
    for i, r in ldp.iterrows():
        hl = not r.carb_in_28_without_it
        ax.barh(i, r.err_ln28, height=0.62, color=C28 if hl else MUTED, zorder=2)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Mix {int(r.mix)} · {r.carbonation}" + (f" · {r.design_point}" if r.design_point.startswith("R") else "")
                        for _, r in ldp.iterrows()], fontsize=6.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ln(observed / predicted) at 28 d, its design point left out and the 28-d terms re-selected")
    ax.set_title("Which compositions carry the 28-day carbonation effect?", loc="left")
    h = [plt.Rectangle((0, 0), 1, 1, color=C28, label="without this design point, carbonation drops out of the 28-d model"),
         plt.Rectangle((0, 0), 1, 1, color=MUTED, label="carbonation stays in the 28-d model")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, ncol=1)
    ax.set_title("Which compositions carry the 28-day carbonation effect?", loc="left", pad=34)
    save(fig, "G09_28d_design_point_influence")


def main():
    M = Model()
    for f in (g01, g02, g03, g04, g05, g06, g07, g08, g09):
        f(M)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
