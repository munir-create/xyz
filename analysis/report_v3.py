"""
Report for the revised-data (v3) bivariate model: report/v3/REPORT.md, report/v3/index.html
(with a strength calculator) and report/v3/Model_Report_v3.pdf (headless Chromium).
Every number is read from results/v3/v3_results.json and results/v3/tables.

Run:  python analysis/report_v3.py
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import bivariate as bv
import design as dz

RES = os.path.join(dz.ROOT, "results", "v3")
TAB = os.path.join(RES, "tables")
REP = os.path.join(dz.ROOT, "report", "v3")
os.makedirs(os.path.join(REP, "figures"), exist_ok=True)


# ============================================================================ formatting
def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def pp(p):
    return "p " + ("< 0.001" if p < 0.001 else "= " + fp(p))


def r2(x):
    return f"{0.0 if abs(x) < 0.005 else x:.2f}"


def ci(lo, hi, d=2):
    return f"{lo:.{d}f}–{hi:.{d}f}"


def tab(name):
    return pd.read_csv(os.path.join(TAB, name + ".csv"))


# ============================================================================ document model
class Doc:
    """Blocks rendered to Markdown and HTML."""

    def __init__(self):
        self.blocks = []

    def h(self, level, text, anchor=None):
        self.blocks.append(("h", level, text, anchor))

    def p(self, text):
        self.blocks.append(("p", text))

    def ul(self, items):
        self.blocks.append(("ul", items))

    def ol(self, items):
        self.blocks.append(("ol", items))

    def table(self, header, rows, caption=None):
        self.blocks.append(("table", header, rows, caption))

    def fig(self, name, caption):
        self.blocks.append(("fig", name, caption))

    def code(self, text):
        self.blocks.append(("code", text))

    def note(self, title, body_blocks, kind="warn"):
        self.blocks.append(("note", title, body_blocks, kind))

    def raw_html(self, text):
        self.blocks.append(("raw", text))

    # ---------------------------------------------------------------- markdown
    def md(self):
        out = []
        for b in self.blocks:
            out.append(_md_block(b))
        return minus("\n\n".join(x for x in out if x) + "\n")

    # ---------------------------------------------------------------- html
    def html(self):
        return minus("\n".join(_html_block(b) for b in self.blocks))


def minus(s):
    """Typographic minus for negative numbers (a hyphen directly before a digit, not inside a word)."""
    return re.sub(r"(?<![\w/.])-(?=\d)", "−", s)


def _md_block(b):
    t = b[0]
    if t == "h":
        return "#" * b[1] + " " + b[2]
    if t == "p":
        return b[1]
    if t == "ul":
        return "\n".join("* " + x for x in b[1])
    if t == "ol":
        return "\n".join(f"{i + 1}. " + x for i, x in enumerate(b[1]))
    if t == "table":
        _, header, rows, caption = b
        s = ("*" + caption + "*\n\n" if caption else "")
        s += "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
        s += "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        return s
    if t == "fig":
        return f"![{b[2]}](figures/{b[1]}.png)\n\n*{b[2]}*"
    if t == "code":
        return "```\n" + b[1] + "\n```"
    if t == "note":
        _, title, body, _k = b
        inner = "\n\n".join(_md_block(x) for x in body)
        return "\n".join("> " + line if line else ">" for line in (f"**{title}**\n\n" + inner).split("\n"))
    return ""


def _inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def _html_block(b):
    t = b[0]
    if t == "h":
        _, lvl, text, anchor = b
        a = f' id="{anchor}"' if anchor else ""
        return f"<h{lvl}{a}>{_inline(text)}</h{lvl}>"
    if t == "p":
        return f"<p>{_inline(b[1])}</p>"
    if t in ("ul", "ol"):
        return f"<{t}>" + "".join(f"<li>{_inline(x)}</li>" for x in b[1]) + f"</{t}>"
    if t == "table":
        _, header, rows, caption = b
        cap = f"<caption>{_inline(caption)}</caption>" if caption else ""
        th = "".join(f"<th>{_inline(h)}</th>" for h in header)
        trs = "".join("<tr>" + "".join(f"<td>{_inline(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'
    if t == "fig":
        return (f'<figure><img src="figures/{b[1]}.png" alt="{html.escape(b[2])}" loading="lazy">'
                f"<figcaption>{_inline(b[2])}</figcaption></figure>")
    if t == "code":
        return f"<pre><code>{html.escape(b[1])}</code></pre>"
    if t == "note":
        _, title, body, kind = b
        return f'<aside class="note {kind}"><p class="nt">{_inline(title)}</p>' + "".join(_html_block(x) for x in body) + "</aside>"
    if t == "raw":
        return b[1]
    return ""


# ============================================================================ content
def build(R):
    fs7, fs28 = R["fit_stats"]["7"], R["fit_stats"]["28"]
    t7, t28 = R["final_terms"]["7"], R["final_terms"]["28"]
    rv = R["revision"]
    pe2, pe3 = rv["pure_error_v2"], rv["pure_error_v3"]
    vc = R["final"]["variance"]
    aic = pd.DataFrame(R["strategy_aicc"]).set_index("model")
    d_aicc = float(aic.loc["Shared-effects paired model, AICc-best (UN)", "AICc"] - aic.loc["Bivariate, age-specific terms (final)", "AICc"])
    d_aicc_ind = float(aic.loc["Same terms, pairing ignored (independent ages)", "AICc"] - aic.loc["Bivariate, age-specific terms (final)", "AICc"])
    cv = pd.DataFrame(R["strategy_cv"])
    cvl = cv[cv.cv == "leave one mixture out"].set_index("procedure")
    cvd = cv[cv.cv == "leave one design point out"].set_index("procedure")
    FINALP = "Bivariate, per-age AICc selection (final procedure)"
    SHAREDP = "Shared-effects paired model, AICc (UN)"
    SSONLY = "SS + SS² only at both ages (reference)"
    cmp_ = pd.DataFrame(R["age_comparison"]).set_index("term")
    ss = pd.DataFrame(R["trend_ss"])
    rc = pd.DataFrame(R["trend_rcf"])
    cb = pd.DataFrame(R["trend_carb"])
    ab = pd.DataFrame(R["trend_ab"]).set_index(["age_d", "AB_to"])
    gn = pd.DataFrame(R["gain"])
    opt = pd.DataFrame(R["optimum"])
    tt = pd.DataFrame(R["term_tests"])
    coef = pd.DataFrame(R["coef"])
    bf = pd.DataFrame(R["boot_freq"])
    grp = pd.DataFrame(R["replicate_group_exclusion"]).set_index("excluded")
    sens = pd.DataFrame(R["sensitivity"])
    dur = {x["age_d"]: x for x in R["carb_durations"]}
    sel = R["selection"]
    added = pd.DataFrame(R["added_terms"])
    bc = R["boxcox"]

    def ssopt(age, carb, rcf):
        return float(ss[(ss.age_d == age) & (ss.carbonation == carb) & (ss.RCF == rcf)].ss_at_max.iloc[0])

    def ssratio(age, carb, rcf, to):
        r = ss[(ss.age_d == age) & (ss.carbonation == carb) & (ss.RCF == rcf) & np.isclose(ss.SS_to, to)].iloc[0]
        return r

    def rcr(age, carb, s):
        return rc[(rc.age_d == age) & (rc.carbonation == carb) & np.isclose(rc.SS, s)].iloc[0]

    def cbr(age, carb, rcf, s):
        return cb[(cb.age_d == age) & (cb.carbonation == carb) & (cb.RCF == rcf) & np.isclose(cb.SS, s)].iloc[0]

    def gnr(carb, rcf, s, a=0.45):
        return gn[(gn.carbonation == carb) & (gn.RCF == rcf) & np.isclose(gn.SS, s) & np.isclose(gn.AB, a)].iloc[0]

    def freq(scheme, age, term):
        return float(bf[(bf.scheme == scheme) & (bf.age_d == age) & (bf.term == term)].freq.iloc[0])

    def tp(age, term):
        return float(tt[(tt.age_d == age) & (tt.term == term)].p.iloc[0])

    rcf28 = float(rc[rc.age_d == 28].rcf_at_max.iloc[0])
    ab_min = float(ab.loc[(28, 0.42), "ab_at_min"])
    ab042, ab048 = ab.loc[(28, 0.42)], ab.loc[(28, 0.48)]
    c05, c1, c5 = (cbr(28, c, 30, 37.5) for c in ("0.5 h", "1 h", "5 h"))
    g0, g375, g75 = gnr("NC", 30, 0.0), gnr("NC", 30, 37.5), gnr("NC", 30, 75.0)
    gmin_mix = [6, 17, 28]
    v2w = R["v2_selection"]["akaike_weights_28"]
    v3w = sel["28"]["akaike_weights"]
    s_v2 = sens[sens.analysis == "Previous data (v2): final v3 terms"].iloc[0]
    s_v3 = sens[sens.analysis == "Final model (v3 data)"].iloc[0]
    s_mpa = sens[sens.analysis == "Raw MPa scale, final terms"].iloc[0]
    loo28 = sel["28"]["loo_reselection"]
    ge = {c["term"]: c["coef"] for c in R["equations"]["gain"]}
    gmins = [-(ge["SS"] + ge.get("RCF·SS", 0) * rcf + ge.get("K·SS", 0) * k) / (2 * ge["SS²"])
             for rcf in (10, 30, 50) for k in (0, 1)]
    gmin_lo, gmin_hi = min(gmins), max(gmins)
    pint = [float(cmp_.loc[t, "p_diff"]) for t in ("KA", "KD", "AD")]     # 7-d interactions: differ at 28 d?
    pint_lo, pint_hi = min(pint), max(pint)
    D = Doc()

    # ------------------------------------------------------------------ title + summary
    D.h(1, "Compressive strength at 7 and 28 days — model of the revised data (v3)")
    D.p(f"*Generated {time.strftime('%Y-%m-%d')} by `analysis/report_v3.py` from `results/v3/v3_results.json`; "
        f"every number below is read from the analysis output. Data: `data/strength_7d_28d_v3_noiseless.csv` "
        f"(30 mixtures, mean strengths; standard deviations not used).*")
    D.h(2, "Answers in brief", "summary")
    D.ol([
        f"**Use one bivariate model for both ages.** Each curing age has its own regression equation with its own terms, and the two "
        f"equations are estimated together, with the 7- and 28-day results of a mixture treated as a correlated pair "
        f"(correlation {vc['rho']:.2f}). It is one model that predicts both ages, it handles the pairing correctly, and it lets the "
        f"effect of every parameter differ between 7 and 28 days. The common alternative — one equation with age as a factor and the "
        f"mixture effects shared by both ages — forces the 7-day effects onto 28 days: AICc prefers the bivariate model by "
        f"{d_aicc:.1f}, and in cross-validation the shared model predicts 28-day strength worse than SS alone "
        f"(predicted R² {r2(cvl.loc[SHAREDP, 'Q2_ln_28'])} vs {r2(cvl.loc[SSONLY, 'Q2_ln_28'])}; bivariate model "
        f"{r2(cvl.loc[FINALP, 'Q2_ln_28'])}).",
        f"**7-day strength** is described to within replicate precision (R² {r2(fs7['R2_ln'])}, predicted R² {r2(fs7['predR2_ln_LOO'])}, "
        f"lack of fit {pp(fs7['lof_p'])}). It is governed by SS, with a maximum near SS {ssopt(7, 'NC', 30):.0f} % (uncarbonated RCF) "
        f"or {ssopt(7, '1 h', 30):.0f} % (carbonated RCF), and by three interactions: a higher RCF dosage raises strength more when the RCF is "
        f"carbonated, carbonation helps at low SS and hurts at high SS, and RCF helps more at low SS. A/B has no effect at 7 days.",
        f"**28-day strength** is described less well (R² {r2(fs28['R2_ln'])}, predicted R² {r2(fs28['predR2_ln_LOO'])}, "
        f"lack of fit {pp(fs28['lof_p'])}). SS still raises it, but much less and without a maximum "
        f"(×{ssratio(28, 'NC', 30, 75).ratio_vs_SS0:.2f} from SS 0 to 75 %); RCF has a maximum near {rcf28:.0f} %; "
        f"A/B 0.42 gives ×{ab042.ratio_vs_AB045:.2f} the strength of A/B 0.45; carbonation for 0.5 h gives "
        f"×{c05.ratio_vs_NC:.2f}, 1 h ×{c1.ratio_vs_NC:.2f} and 5 h ×{c5.ratio_vs_NC:.2f} the strength of uncarbonated RCF. "
        f"No interaction is detectable at 28 days.",
        f"**What changes between 7 and 28 days** (tests of equal effect at both ages): SS ({pp(cmp_.loc['D', 'p_diff'])}), "
        f"its curvature ({pp(cmp_.loc['D2', 'p_diff'])}), carbonation ({pp(cmp_.loc['Carb', 'p_diff'])}), A/B "
        f"({pp(cmp_.loc['C', 'p_diff'])}) and the RCF curvature ({pp(cmp_.loc['A2', 'p_diff'])}). The three 7-day interactions are "
        f"not detectable at 28 days, but the data cannot show that they vanish "
        f"(p = {fp(pint_lo)}–{fp(pint_hi)} for a difference).",
        f"**Strength gain f28/f7** is largest without silicate (×{g0.gain_f28_over_f7:.1f} at SS 0 %, NC, RCF 30 %) and smallest "
        f"near SS {gmin_lo:.0f}–{gmin_hi:.0f} % (×{g375.gain_f28_over_f7:.2f} at SS 37.5 %); carbonation and RCF modify it. The 5 h "
        f"mixtures at SS ≈ 50 % (mixes {', '.join(map(str, gmin_mix))}) gained no strength after 7 days.",
        f"**How firm is each conclusion?** Every 7-day conclusion is robust (the same model is selected in all 30 leave-one-out "
        f"refits). At 28 days the SS effect, the RCF maximum and the benefit of low A/B are robust. The carbonation differences rest "
        f"on three compositions (two at 0.5 h, one at 5 h) and the A/B curvature on one of them: leave any of these out and the "
        f"selection drops carbonation from the 28-day model and the affected effect is no longer significant. Report them as "
        f"indications, not established effects (Section 8).",
        f"**The data revision needs documenting before publication.** {rv['n_mean_changes']} means changed relative to the previous "
        f"data set; several changes cannot come from removing noisy specimens (Section 1). The revision does not reverse any effect, "
        f"but it is what makes the 28-day carbonation and A/B-curvature terms selectable (Akaike weights {v2w['Carb']:.2f} and "
        f"{v2w['C2']:.2f} with the previous values, {v3w['Carb']:.2f} and {v3w['C2']:.2f} with the revised ones).",
    ])

    # ------------------------------------------------------------------ 1 data revision
    D.h(2, "1  Data and the revision (v2 → v3)", "data")
    D.p(f"The revised table (`data/source/30_mixes_strength_results_noiseless.pdf`, transcribed programmatically to "
        f"`data/strength_7d_28d_v3_noiseless.csv`) has the same 30 mixtures and the same mixture variables as the previous data set "
        f"(`data/strength_7d_28d_corrected.csv`, v2). {rv['n_entries_changed']} entries differ: {rv['n_mean_changes']} means "
        f"({rv['by_age']['7']} at 7 days, {rv['by_age']['28']} at 28 days) and {rv['n_last_digit']} last-digit changes of 0.01 MPa. "
        f"Every change is listed in `results/v3/tables/V00_revision_log_v2_to_v3.csv`.")
    sdu = rv["sd_unchanged_shift"]
    sdu28 = [x for x in sdu if x["age_d"] == 28]
    D.note("Before publishing: document every revised value", [
        ("p", "Several of the changes are not what removing noisy specimens produces:"),
        ("ul", [
            f"**{len(sdu)} means moved while their standard deviation stayed identical**, among them at 28 days "
            + ", ".join(f"mix {x['mix']} by {x['change_MPa']:+.2f} MPa" for x in sdu28)
            + ". With three specimens, a mean can move with an unchanged SD only if every specimen moved by the same amount; "
              "excluding or re-testing a specimen changes the SD.",
            f"**{rv['n_sd_increased']} SDs increased and {'none' if rv['n_sd_decreased'] == 0 else rv['n_sd_decreased']} decreased.** "
            f"Removing noisy specimens normally lowers the SD.",
            f"**Replicate batches moved toward each other.** Mixes 10 and 14 (identical composition) moved by exactly 3.00 MPa each "
            f"toward one another, and {rv['toward_replicates']} of the {rv['replicate_members_changed']} changed replicate means moved "
            f"toward their group. The replicate (pure-error) SD fell from {pe2['28_MPa']['sd']:.2f} to {pe3['28_MPa']['sd']:.2f} MPa at "
            f"28 days and from {pe2['7_MPa']['sd']:.2f} to {pe3['7_MPa']['sd']:.2f} MPa at 7 days.",
        ]),
        ("p", "There can be legitimate reasons — a corrected specimen area, a load-cell zero offset on one test day, transcription "
              "errors in the earlier table. But a reviewer or co-author comparing the versions will ask, so each change needs a reason "
              "traceable to the raw specimen records. If a value was moved toward an expected or replicate value rather than recomputed "
              "from specimens, it cannot be reported as a measurement; the measured values must be used. Section 8 shows which "
              "conclusions depend on the revision."),
    ])
    rvt = tab("V00_revision_log_v2_to_v3")
    rvt = rvt[rvt.change_MPa.abs() > 0.011]
    D.table(["Mix", "Age", "Replicate group", "v2 mean ± SD", "v3 mean ± SD", "Change (MPa)", "Kind"],
            [[int(r.mix), f"{int(r.age_d)} d", r.replicate_group if isinstance(r.replicate_group, str) else "–",
              f"{r.mean_v2:.2f} ± {r.sd_v2:.2f}", f"{r.mean_v3:.2f} ± {r.sd_v3:.2f}", f"{r.change_MPa:+.2f}", r.kind]
             for _, r in rvt.iterrows()], "Means that changed by more than 0.01 MPa")
    D.fig("G01_data_revision", "Figure 1 — Every mean that changed between the previous (v2) and the revised (v3) data. Diamonds: the mean "
                               "moved while the SD stayed the same.")

    # ------------------------------------------------------------------ 2 strategy
    D.h(2, "2  One model for both ages, or one per age?", "strategy")
    D.p("The 7- and 28-day results of a mixture come from the same batch, so they are a correlated pair. There are three ways to model them:")
    D.ul([
        "**Separate models, one per age.** Valid, and each age gets its own terms. But the two models know nothing of each other, "
        "so they cannot test whether an effect differs between 7 and 28 days or give a confidence interval for the strength gain.",
        "**One equation with curing age as a factor and shared effects** (the earlier paired model): each mixture effect is the "
        "same at both ages unless an age × effect interaction is selected. Efficient when the effects really are shared; misleading "
        "when they are not.",
        "**One bivariate model with age-specific terms** (chosen): one equation per age, each with its own terms, estimated together "
        "with an unstructured covariance for the 7/28-day pair (seemingly unrelated regressions; equivalently a linear mixed model "
        "with age-specific fixed effects and an unstructured within-mixture covariance). Both alternatives are special cases of it — "
        "separate models drop the correlation, the shared-effects model forces equal coefficients — so it can be compared with "
        "both on the same likelihood, and Section 4 tests which effects differ between the ages.",
    ])
    D.p(f"On the revised data the bivariate model is better by every criterion. On the same joint likelihood, AICc prefers it to the "
        f"best shared-effects model by {d_aicc:.1f} and to the same terms with the pairing ignored by {d_aicc_ind:.1f}. In nested "
        f"cross-validation (the whole model selection repeated in every fold) it predicts 7-day strength as well as the shared-effects "
        f"model and 28-day strength much better:")
    rows = []
    for nm, lab in [(FINALP, "**Bivariate, age-specific terms (chosen)**"), (SHAREDP, "Shared effects, AICc (UN)"),
                    ("Shared-effects paired model, AICc (CS + UN; earlier protocol)", "Shared effects, AICc (CS + UN; earlier protocol)"),
                    ("Full candidate model at both ages (no selection)", "Full candidate model, no selection"),
                    (SSONLY, "SS + SS² only (reference)")]:
        a, b = cvl.loc[nm], cvd.loc[nm]
        rows.append([lab, r2(a.Q2_ln_7), r2(a.Q2_ln_28), f"{a.RMSE_MPa_7:.1f} / {a.RMSE_MPa_28:.1f}",
                     r2(b.Q2_ln_7), r2(b.Q2_ln_28)])
    D.table(["Procedure", "Pred. R² 7 d (LOMO)", "Pred. R² 28 d (LOMO)", "RMSE 7 / 28 d (MPa, LOMO)",
             "Pred. R² 7 d (LODPO)", "Pred. R² 28 d (LODPO)"], rows,
            "Nested cross-validation, ln scale. LOMO: leave one mixture out; LODPO: leave one design point out (all replicate "
            "batches of a composition together)")
    D.fig("G02_strategy_comparison", "Figure 2 — Predicted R² of each modelling procedure under nested cross-validation.")
    D.p(f"Leaving out whole design points is the stricter test, because replicate batches of the held-out composition can no longer "
        f"help. Under it the 7-day model still predicts new compositions very well (predicted R² {r2(cvd.loc[FINALP, 'Q2_ln_7'])}), "
        f"but no procedure predicts the 28-day strength of a new composition reliably (bivariate {r2(cvd.loc[FINALP, 'Q2_ln_28'])}, "
        f"shared effects {r2(cvd.loc[SHAREDP, 'Q2_ln_28'])}, SS alone {r2(cvd.loc[SSONLY, 'Q2_ln_28'])}). The reason is that the "
        f"28-day carbonation effect is carried by three compositions (Section 8).")

    # ------------------------------------------------------------------ 3 final model
    D.h(2, "3  The final model", "model")
    D.p(f"Response: ln(strength) (Box–Cox λ = {bc['bivariate']['lambda']:.2f}, 95 % CI {bc['bivariate']['ci'][0]:.2f} to "
        f"{bc['bivariate']['ci'][1]:.2f}, for both ages together; λ = 1, the raw MPa scale, is rejected at both ages: "
        f"LR {bc['7']['LR_lambda1']:.1f} at 7 d, {bc['28']['LR_lambda1']:.1f} at 28 d). Coded variables: A = (RCF − 30)/20, C = (A/B − 0.45)/0.03, D = (SS − 37.5)/37.5, so −1 and +1 "
        f"are the ends of the design range. Carbonation enters as a four-level factor (NC, 0.5 h, 1 h, 5 h); its interactions with the "
        f"mixture variables use K = 1 for carbonated RCF, because the 0.5 h level has only two distinct compositions. Terms per age: "
        f"exhaustive AICc search over all {sel['n_models_per_age']} hierarchical models of the full quadratic candidate set.")
    cf = coef.copy()

    def coded_eq(age):
        r = cf[cf.age_d == age]
        names = {"Intercept": "", "A": "A", "C": "C", "D": "D", "AD": "A·D", "A2": "A²", "C2": "C²", "D2": "D²", "KA": "K·A",
                 "KD": "K·D", "Carb[0.5 h]": "[0.5 h]", "Carb[1 h]": "[1 h]", "Carb[5 h]": "[5 h]"}
        s = ""
        for _, x in r.iterrows():
            nm = names.get(x.column, x.column)
            v = x.coef
            if not s:
                s = f"{v:.4f}" + (f"·{nm}" if nm else "")
            else:
                s += f" {'−' if v < 0 else '+'} {abs(v):.4f}·{nm}"
        return s
    D.code(f"ln f7  = {coded_eq(7)}\nln f28 = {coded_eq(28)}\n\n"
           f"A = (RCF − 30)/20, C = (A/B − 0.45)/0.03, D = (SS − 37.5)/37.5; [0.5 h], [1 h], [5 h] = 1 for that carbonation\n"
           f"level (all 0 for NC); K = 1 for carbonated RCF (any duration).\n"
           f"Residual SD (ln): {vc['sd_7']:.3f} at 7 d, {vc['sd_28']:.3f} at 28 d; correlation of the 7/28-day pair {vc['rho']:.2f}.")

    def act(key, lhs):
        parts = []
        for c in R["equations"][key]:
            v, t = c["coef"], c["term"]
            mag = f"{abs(v):.6g}"
            term = "" if t == "1" else "·" + t
            parts.append((("−" if v < 0 else "") if not parts else (" − " if v < 0 else " + ")) + mag + term)
        return f"{lhs} = " + "".join(parts)
    eq28 = {c["term"]: c["coef"] for c in R["equations"]["28 d"]}
    h_rcf = -eq28["RCF"] / (2 * eq28["RCF²"])
    h_ab = -eq28["A/B"] / (2 * eq28["(A/B)²"])
    D.p("In actual units (RCF and SS in %, A/B as a ratio; C[·] = 1 for that carbonation level, K = 1 for any carbonated RCF):")
    D.code(act("7 d", "ln(f7) ") + "\n" + act("28 d", "ln(f28)") + "\n" + act("gain", "ln(f28/f7)") +
           f"\n\nThe 28-day RCF and A/B terms in vertex form: {eq28['RCF²']:.6g}·(RCF − {h_rcf:.1f})² and "
           f"+{eq28['(A/B)²']:.6g}·(A/B − {h_ab:.4f})²,\n"
           f"i.e. a maximum at RCF ≈ {h_rcf:.0f} % and a minimum at A/B ≈ {h_ab:.3f}. Keep all digits of the A/B coefficients; "
           f"they nearly cancel.")
    D.p("Strength in MPa is exp(·) of these expressions (the median of a new batch). Term tests (Wald F, Satterthwaite df):")
    rows = []
    for age in (7, 28):
        for _, x in tt[tt.age_d == age].iterrows():
            rows.append([f"{age} d", x.label, f"{int(x.df1)}, {x.df2:.1f}", f"{x.F:.2f}", fp(x.p),
                         f"{freq('residual bootstrap', age, x.term):.0%}", f"{freq('subsample 24/30', age, x.term):.0%}"])
    D.table(["Age", "Term", "df", "F", "p", "Selected, residual bootstrap", "Selected, subsamples 24/30"], rows,
            "Final model: term tests and how often the term is re-selected when the data are resampled")
    rows = []
    for _, x in cf.iterrows():
        lab = x.column.replace("Carb[", "Carbonation [") if x.column.startswith("Carb[") else (x.label if x.column != "Intercept" else "Intercept")
        rows.append([f"{int(x.age_d)} d", lab, x.column, f"{x.coef:.4f}", f"{x.se:.4f}", f"{x.lo:.3f} to {x.hi:.3f}", fp(x.p)])
    D.table(["Age", "Term", "Coded column", "Estimate", "SE", "95 % CI", "p"], rows,
            "Coefficients of the final model (ln scale, coded units)")
    D.p("Fit and validation:")
    D.table(["Statistic", "7 d", "28 d"], [
        ["Coefficients", fs7["n_coef"], fs28["n_coef"]],
        ["R² (ln) / adjusted R²", f"{r2(fs7['R2_ln'])} / {r2(fs7['adjR2_ln'])}", f"{r2(fs28['R2_ln'])} / {r2(fs28['adjR2_ln'])}"],
        ["Predicted R² (ln), leave one mixture out", r2(fs7["predR2_ln_LOO"]), r2(fs28["predR2_ln_LOO"])],
        ["Predicted R² (ln), leave one design point out", r2(fs7["predR2_ln_LODP"]), r2(fs28["predR2_ln_LODP"])],
        ["RMSE fitted / predicted (MPa)", f"{fs7['RMSE_fit_MPa']:.1f} / {fs7['RMSE_pred_MPa']:.1f}", f"{fs28['RMSE_fit_MPa']:.1f} / {fs28['RMSE_pred_MPa']:.1f}"],
        ["Residual SD (ln) ≈ CV", f"{fs7['resid_sd_ln']:.3f}", f"{fs28['resid_sd_ln']:.3f}"],
        ["Replicate (pure-error) SD (ln)", f"{fs7['pure_error_sd_ln']:.3f}", f"{fs28['pure_error_sd_ln']:.3f}"],
        ["Lack of fit F (df), p", f"{fs7['lof_F']:.2f} ({fs7['lof_df'][0]}, {fs7['lof_df'][1]}), {fp(fs7['lof_p'])}",
         f"{fs28['lof_F']:.2f} ({fs28['lof_df'][0]}, {fs28['lof_df'][1]}), {fp(fs28['lof_p'])}"],
        ["Adequate precision (> 4 is adequate)", f"{fs7['adeq_precision']:.1f}", f"{fs28['adeq_precision']:.1f}"],
        ["Shapiro–Wilk p / Breusch–Pagan p", f"{fp(R['diagnostics']['7']['shapiro_p'])} / {fp(R['diagnostics']['7']['bp_p'])}",
         f"{fp(R['diagnostics']['28']['shapiro_p'])} / {fp(R['diagnostics']['28']['bp_p'])}"],
        ["Largest studentized residual (Bonferroni p)", f"{R['diagnostics']['7']['max_abs_tdel']:.2f}, mix {R['diagnostics']['7']['max_tdel_mix']} ({fp(R['diagnostics']['7']['min_p_bonf'])})",
         f"{R['diagnostics']['28']['max_abs_tdel']:.2f}, mix {R['diagnostics']['28']['max_tdel_mix']} ({fp(R['diagnostics']['28']['min_p_bonf'])})"],
        ["Largest Cook's distance", f"{R['diagnostics']['7']['max_cook']:.2f} (mix {R['diagnostics']['7']['max_cook_mix']})",
         f"{R['diagnostics']['28']['max_cook']:.2f} (mix {R['diagnostics']['28']['max_cook_mix']})"],
    ], "Fit statistics per age (predicted R² with the final terms fixed; the nested values are in Section 2)")
    D.fig("G08_validation_diagnostics", "Figure 3 — Observed vs fitted and vs left-out predictions, residuals and normal Q–Q plot.")
    D.p(f"At 7 days the residual scatter equals the replicate scatter, so the model explains everything the mixture variables can. At "
        f"28 days the residual scatter is about twice the (revised) replicate scatter, so 28-day strength varies between compositions "
        f"for reasons the four variables do not capture. No single result is an outlier (largest Bonferroni p "
        f"{fp(R['diagnostics']['28']['min_p_bonf'])}).")

    # ------------------------------------------------------------------ 4 trends
    D.h(2, "4  How strength responds to each parameter", "trends")
    D.fig("G03_main_trends_7d_vs_28d", "Figure 4 — One factor at a time, the others at RCF 30 %, SS 37.5 %, A/B 0.45 and NC.")
    D.fig("G07_effects_7d_vs_28d", "Figure 5 — Every effect at 7 and at 28 days, with the test of whether it differs between the ages. "
                                   "Hollow: the term was not selected at that age (estimated here only for the comparison).")
    r_ss7 = ssratio(7, "NC", 30, ssopt(7, "NC", 30))
    D.ul([
        f"**Silicate share SS** is the dominant factor at 7 days: from SS 0 to the maximum at SS {ssopt(7, 'NC', 30):.0f} % the "
        f"strength rises ×{r_ss7.ratio_vs_SS0:.1f} (NC, RCF 30 %). The maximum lies at {ssopt(7, 'NC', 50):.0f}–{ssopt(7, 'NC', 10):.0f} % "
        f"for uncarbonated and {ssopt(7, '1 h', 50):.0f}–{ssopt(7, '1 h', 10):.0f} % for carbonated RCF (lower at higher RCF). At 28 days "
        f"the effect is weaker, ×{ssratio(28, 'NC', 30, 75).ratio_vs_SS0:.2f} from SS 0 to 75 %, and has no maximum within the design "
        f"range (SS² at 28 d: Akaike weight {v3w['D2']:.2f}).",
        f"**RCF dosage** at 7 days depends on carbonation and SS (next section). At 28 days it has a maximum near RCF {rcf28:.0f} %; "
        f"RCF 50 % gives ×{rcr(28, 'NC', 0.0).ratio_RCF50_vs_10:.2f} the strength of RCF 10 % "
        f"(95 % CI {ci(rcr(28, 'NC', 0.0).lo, rcr(28, 'NC', 0.0).hi)}).",
        f"**A/B** (0.42–0.48) has no effect at 7 days (added-term {pp(float(added[(added.age_d == 7) & (added.added == 'C')].p.iloc[0]))}). "
        f"At 28 days A/B 0.42 gives ×{ab042.ratio_vs_AB045:.2f} (95 % CI {ci(ab042.lo, ab042.hi)}) the strength of A/B 0.45; A/B 0.48 "
        f"gives ×{ab048.ratio_vs_AB045:.2f} ({ci(ab048.lo, ab048.hi)}, {pp(ab048.p)}). The fitted curve has its minimum at A/B ≈ "
        f"{ab_min:.3f}; the rise toward 0.48 is the least certain part (Section 8).",
        f"**Carbonation duration**: at 7 days the three durations hardly differ ({pp(dur[7]['durations_differ_p'])}); what matters is "
        f"carbonated versus not, through the interactions. At 28 days the durations differ ({pp(dur[28]['durations_differ_p'])}): "
        f"0.5 h ×{c05.ratio_vs_NC:.2f} ({ci(c05.lo, c05.hi)}), 1 h ×{c1.ratio_vs_NC:.2f} ({ci(c1.lo, c1.hi)}), 5 h "
        f"×{c5.ratio_vs_NC:.2f} ({ci(c5.lo, c5.hi)}) relative to NC, the same at every composition.",
    ])

    # ------------------------------------------------------------------ 5 interactions
    D.h(2, "5  Interactions", "interactions")
    D.fig("G04_carbonation_interactions", "Figure 6 — Strength vs SS and vs RCF for each carbonation level, at 7 days (top) and 28 days "
                                          "(bottom). Non-parallel curves are interactions.")
    a0, a75 = cbr(7, "1 h", 50, 0.0), cbr(7, "1 h", 10, 75.0)
    D.p("At 7 days three interactions are supported (all three re-selected in all 30 leave-one-out refits):")
    D.ul([
        f"**Carbonation × SS** ({pp(tp(7, 'KD'))}): carbonated RCF raises 7-day strength in silicate-poor mixes and lowers it in "
        f"silicate-rich ones. Carbonated (1 h) vs NC: ×{a0.ratio_vs_NC:.2f} ({ci(a0.lo, a0.hi)}) at RCF 50 %, SS 0 %, but "
        f"×{a75.ratio_vs_NC:.2f} ({ci(a75.lo, a75.hi)}) at RCF 10 %, SS 75 %.",
        f"**Carbonation × RCF** ({pp(tp(7, 'KA'))}): carbonated RCF is the more reactive filler. At SS 0 %, raising RCF from 10 to 50 % "
        f"gives ×{rcr(7, '1 h', 0.0).ratio_RCF50_vs_10:.2f} ({ci(rcr(7, '1 h', 0.0).lo, rcr(7, '1 h', 0.0).hi)}) when it is carbonated "
        f"and ×{rcr(7, 'NC', 0.0).ratio_RCF50_vs_10:.2f} ({ci(rcr(7, 'NC', 0.0).lo, rcr(7, 'NC', 0.0).hi)}) when it is not.",
        f"**RCF × SS** ({pp(tp(7, 'AD'))}): RCF helps most when SS is low. Uncarbonated RCF 50 vs 10 %: "
        f"×{rcr(7, 'NC', 0.0).ratio_RCF50_vs_10:.2f} at SS 0 %, ×{rcr(7, 'NC', 37.5).ratio_RCF50_vs_10:.2f} at 37.5 % "
        f"({pp(rcr(7, 'NC', 37.5).p)}) and ×{rcr(7, 'NC', 75.0).ratio_RCF50_vs_10:.2f} at 75 % ({pp(rcr(7, 'NC', 75.0).p)}); "
        f"carbonated: ×{rcr(7, '1 h', 0.0).ratio_RCF50_vs_10:.2f}, ×{rcr(7, '1 h', 37.5).ratio_RCF50_vs_10:.2f} and "
        f"×{rcr(7, '1 h', 75.0).ratio_RCF50_vs_10:.2f}.",
    ])
    a28 = added[added.age_d == 28].set_index("added")
    D.p(f"At 28 days none of these interactions improves the model (added-term tests: carbonation × SS {pp(a28.loc['KD', 'p'])}, "
        f"carbonation × RCF {pp(a28.loc['KA', 'p'])}, RCF × SS {pp(a28.loc['AD', 'p'])}; every other added term p ≥ "
        f"{a28.p.min():.2f}). On the ln scale the 28-day effects are additive, i.e. each factor multiplies strength by the same factor "
        f"whatever the others are. Whether the 7-day interactions truly fade by 28 days or are only masked by the larger 28-day scatter "
        f"cannot be decided from these data (tests of a difference: p = {fp(cmp_.loc['KA', 'p_diff'])}, {fp(cmp_.loc['KD', 'p_diff'])}, "
        f"{fp(cmp_.loc['AD', 'p_diff'])}).")
    D.fig("G05_response_surfaces", "Figure 7 — Predicted strength over RCF and SS for uncarbonated and carbonated (1 h) RCF at A/B 0.45.")

    # ------------------------------------------------------------------ 6 gain
    D.h(2, "6  Strength gain from 7 to 28 days", "gain")
    D.fig("G06_strength_gain", "Figure 8 — Gain f28/f7 vs SS for each carbonation level and three RCF dosages, with the observed gains.")
    g5 = gnr("5 h", 50, 37.5)
    D.p(f"Because SS raises 7-day strength much more than 28-day strength, the gain falls steeply from SS 0 % to a minimum near "
        f"SS {gmin_lo:.0f}–{gmin_hi:.0f} % (depending on RCF and carbonation) and rises slightly beyond: ×{g0.gain_f28_over_f7:.2f} "
        f"(95 % CI {ci(g0.lo, g0.hi)}) at SS 0 %, ×{g375.gain_f28_over_f7:.2f} ({ci(g375.lo, g375.hi)}) at SS 37.5 % and "
        f"×{g75.gain_f28_over_f7:.2f} ({ci(g75.lo, g75.hi)}) at SS 75 % (NC, RCF 30 %, A/B 0.45). Carbonation and RCF modify it: "
        f"carbonated RCF, which boosts early strength, gains less later, most of all after 5 h of carbonation (5 h, RCF 50 %, "
        f"SS 37.5 %: ×{g5.gain_f28_over_f7:.2f}, {ci(g5.lo, g5.hi)}). The observed 5 h mixtures at SS ≈ 50 % (mixes 6, 17, 28) have "
        f"gains of 0.90–1.02.")

    # ------------------------------------------------------------------ 7 optimum
    D.h(2, "7  Strongest mixes within the design range", "optimum")
    rows = []
    for _, o in opt.iterrows():
        rows.append([f"{int(o.age_d)} d", o.carbonation, f"{o.RCF:.0f}", f"{o.SS:.0f}",
                     "any (no effect)" if not o.AB_in_model else f"{o.AB:.2f}", f"{o['median']:.1f}",
                     ci(o.ci_lo, o.ci_hi, 1), ci(o.pi_lo, o.pi_hi, 1),
                     f"RCF {o.within5_RCF[0]:.0f}–{o.within5_RCF[1]:.0f}, SS {o.within5_SS[0]:.0f}–{o.within5_SS[1]:.0f}",
                     f"{o.coded_distance_to_nearest_mix_same_carb:.2f}"])
    D.table(["Age", "Carbonation", "RCF %", "SS %", "A/B", "Median (MPa)", "95 % CI", "95 % PI (one batch)",
             "Within 5 % of the maximum", "Distance to nearest tested mix (coded)"], rows,
            "Maximum predicted strength per age and carbonation level")
    D.p("Every 28-day maximum lies on the edge of the design range (SS 75 %, A/B 0.42), because SS and low A/B keep raising 28-day "
        "strength up to the boundary; these are extrapolation-prone. The 0.5 h optimum is furthest from any tested 0.5 h mixture "
        "(the 0.5 h level was tested at only two compositions, both at A/B 0.45 and SS ≤ 38 %), so its 28-day value of "
        f"{float(opt[(opt.age_d == 28) & (opt.carbonation == '0.5 h')]['median'].iloc[0]):.0f} MPa is a hypothesis to test, not a "
        "prediction to rely on. The 7-day maxima sit inside or close to tested mixtures.")

    # ------------------------------------------------------------------ 8 reliability
    D.h(2, "8  How firm is each conclusion?", "reliability")
    D.fig("G09_28d_design_point_influence", "Figure 9 — 28-day prediction error of each mixture when its whole design point is left "
                                            "out and the 28-day terms re-selected. Orange: without this design point, carbonation "
                                            "drops out of the 28-day model.")
    g = grp
    D.p(f"The 28-day carbonation effect is carried by three compositions: the two 0.5 h compositions (replicate groups R1 and R2) and "
        f"the 5 h composition at SS ≈ 50 % (R3), plus the single-specimen mix 29. Without R1 the carbonation test at 28 days gives "
        f"{pp(g.loc['R1', 'carb28_p'])} and the 0.5 h effect ×{g.loc['R1', '28d 0.5 h/NC']:.2f} ({pp(g.loc['R1', 'p 0.5 h'])}); "
        f"without R3 the 5 h effect is ×{g.loc['R3', '28d 5 h/NC']:.2f} ({pp(g.loc['R3', 'p 5 h'])}) and the A/B curvature "
        f"{pp(g.loc['R3', 'C2_28_p'])}. The benefit of low A/B survives every exclusion (A/B 0.42 vs 0.45: "
        f"×{g['28d AB0.42/0.45'].min():.2f}–{g['28d AB0.42/0.45'].max():.2f}, p ≤ {fp(g['p AB'].max())}).")
    rows = [
        ["SS raises strength, with a maximum at 7 d", "p < 0.001; selected in 100 % of resamples", "Robust"],
        ["SS effect much weaker at 28 d (gain falls with SS)", f"difference {pp(cmp_.loc['D', 'p_diff'])}", "Robust"],
        ["Carbonation × SS, carbonation × RCF, RCF × SS at 7 d",
         f"{pp(tp(7, 'KD'))}, {pp(tp(7, 'KA'))}, {pp(tp(7, 'AD'))}; residual bootstrap {freq('residual bootstrap', 7, 'KD'):.0%}, "
         f"{freq('residual bootstrap', 7, 'KA'):.0%}, {freq('residual bootstrap', 7, 'AD'):.0%}; 30/30 leave-one-out", "Robust"],
        ["A/B has no effect at 7 d", f"added-term {pp(float(added[(added.age_d == 7) & (added.added == 'C')].p.iloc[0]))}", "Robust"],
        ["Lower A/B raises 28-d strength", f"{pp(ab042.p)}; survives every replicate-group exclusion", "Robust"],
        ["RCF maximum near 36 % at 28 d", f"RCF² {pp(tp(28, 'A2'))}; residual bootstrap {freq('residual bootstrap', 28, 'A2'):.0%}", "Moderate"],
        ["Carbonation durations differ at 28 d (0.5 h up, 1 h and 5 h down)",
         f"{pp(tp(28, 'Carb'))}; residual bootstrap {freq('residual bootstrap', 28, 'Carb'):.0%}, subsamples "
         f"{freq('subsample 24/30', 28, 'Carb'):.0%}; rests on R1, R2, R3", "Indication"],
        ["A/B curvature (minimum near 0.46) at 28 d",
         f"{pp(tp(28, 'C2'))}; residual bootstrap {freq('residual bootstrap', 28, 'C2'):.0%}; rests on R3", "Indication"],
        ["The 7-d interactions vanish by 28 d", f"differences p = {fp(pint_lo)}–{fp(pint_hi)}",
         "Not shown"],
    ]
    D.table(["Conclusion", "Evidence", "Status"], rows)
    D.p(f"**Dependence on the data revision.** With the previous values (v2) the same protocol selects the same 7-day model and a simpler "
        f"28-day model ({', '.join(dz.label(t) for t in R['v2_selection']['28'])}). Fitting the revised model's terms to the v2 values "
        f"gives effects of the same direction and similar size:")
    D.table(["Quantity (28 d unless stated)", "Revised data (v3)", "Previous data (v2), same terms"], [
        ["0.5 h / NC", f"{s_v3['28d 0.5 h/NC']:.2f}", f"{s_v2['28d 0.5 h/NC']:.2f}"],
        ["1 h / NC", f"{s_v3['28d 1 h/NC']:.2f}", f"{s_v2['28d 1 h/NC']:.2f}"],
        ["5 h / NC", f"{s_v3['28d 5 h/NC']:.2f}", f"{s_v2['28d 5 h/NC']:.2f}"],
        ["A/B 0.42 / 0.45", f"{s_v3['28d AB0.42/0.45']:.2f}", f"{s_v2['28d AB0.42/0.45']:.2f}"],
        ["Gain f28/f7 at SS 0 % (NC, RCF 30 %)", f"{s_v3['gain SS0']:.2f}", f"{s_v2['gain SS0']:.2f}"],
        ["Gain f28/f7 at SS 75 %", f"{s_v3['gain SS75']:.2f}", f"{s_v2['gain SS75']:.2f}"],
        ["R² (ln) 7 d / 28 d", f"{s_v3['R2_ln_7']:.2f} / {s_v3['R2_ln_28']:.2f}", f"{s_v2['R2_ln_7']:.2f} / {s_v2['R2_ln_28']:.2f}"],
    ])
    D.p(f"So the revision does not create or reverse an effect; it lowers the 28-day scatter enough for AICc to select the carbonation "
        f"and A/B-curvature terms (Akaike weights {v2w['Carb']:.2f} → {v3w['Carb']:.2f} and {v2w['C2']:.2f} → {v3w['C2']:.2f}). "
        f"Other checks: the raw MPa scale fits worse (predicted R² in MPa {s_mpa.predR2_MPa_7:.2f} / {s_mpa.predR2_MPa_28:.2f} vs "
        f"{fs7['predR2_MPa_LOO']:.2f} / {fs28['predR2_MPa_LOO']:.2f} for the ln model); coding carbonation as carbonated yes/no "
        f"removes it from the 28-day model (AICc {R['carb_onoff_28']['AICc']:.1f} vs {R['carb_onoff_28']['AICc_cat4']:.1f}), i.e. the "
        f"28-day pattern is 0.5 h vs longer carbonation, not carbonated vs uncarbonated; with mix 7 (largest Cook's distance at 28 d) "
        f"removed, the 28-day model adds carbonation × SS and the carbonation ratios become "
        + ", ".join(f"{c} ×{sens[sens.analysis.str.startswith('Mix 7')].iloc[0][f'28d {c}/NC']:.2f}" for c in ("0.5 h", "1 h", "5 h")) + ".")

    # ------------------------------------------------------------------ 9 limitations
    D.h(2, "9  Limitations and next experiments", "limits")
    D.ul([
        f"28-day strength varies between compositions about twice as much as between replicate batches (lack of fit "
        f"{pp(fs28['lof_p'])}). The residual scatter alone gives ×/÷ {np.exp(1.96 * vc['sd_28']):.1f} for one new batch at 28 days, "
        f"against ×/÷ {np.exp(1.96 * vc['sd_7']):.1f} at 7 days; the prediction intervals in Section 7 are wider still.",
        "Carbonation level is confounded with composition at 28 days: 0.5 h was tested at two compositions, and the 5 h evidence comes "
        "mostly from one composition. A small confirmation set — 0.5 h and 5 h at SS 0, 37.5 and 75 % (RCF 30 %, A/B 0.45), "
        "triplicate batches — would settle whether duration matters at 28 days.",
        "A/B spans only 0.42–0.48; its 28-day curvature should be confirmed with a batch at A/B 0.48 and at 0.42 for one fixed "
        "composition before it is interpreted physically.",
        "Mixes 11 (28 d), 29 (both ages) and 30 (28 d) are single-specimen results; mix 29 is the lowest 28-day value and influences "
        "the 28-day model (Figure 9).",
        "The 28-day maxima lie on the edges of the design range; confirm them experimentally before recommending a mix.",
    ])
    D.h(3, "Suggested methods text")
    D.p(f"\"Compressive strengths at 7 and 28 days (means of the specimens of each mixture) were analysed on the natural-log scale "
        f"(Box–Cox λ = {bc['bivariate']['lambda']:.2f}, 95 % CI {bc['bivariate']['ci'][0]:.2f} to {bc['bivariate']['ci'][1]:.2f}). "
        f"Because both ages were measured on the same mixture batches, they were modelled jointly as a bivariate linear model "
        f"(seemingly unrelated regressions) with age-specific terms and an unstructured 2 × 2 covariance for the 7- and 28-day "
        f"results of a mixture, fitted by restricted maximum likelihood; terms were tested with Wald F statistics and Satterthwaite "
        f"degrees of freedom. Mixture variables were coded to −1…+1 over the design range (RCF 10–50 %, A/B 0.42–0.48, SS 0–75 %); "
        f"carbonation (none, 0.5, 1, 5 h) entered as a four-level factor and its interactions with the mixture variables through a "
        f"carbonated/uncarbonated contrast. For each age the terms were chosen by exhaustive AICc search over all "
        f"{sel['n_models_per_age']} hierarchical sub-models of the full quadratic candidate set. A shared-effects model (one set of "
        f"mixture effects for both ages plus selected age interactions) was rejected (ΔAICc = {d_aicc:.1f}; poorer nested "
        f"cross-validated prediction at 28 days). Model selection was validated by nested leave-one-mixture-out and "
        f"leave-one-design-point-out cross-validation, {R['meta']['n_resid_boot']:,} residual-bootstrap and "
        f"{R['meta']['n_subsample']:,} subsample re-selections, and lack-of-fit tests against replicate batches.\"")

    # ------------------------------------------------------------------ 10 files
    D.h(2, "10  Files and reproduction", "files")
    D.ul([
        "`data/strength_7d_28d_v3_noiseless.csv` — revised data (programmatic transcription of `data/source/30_mixes_strength_results_noiseless.pdf`).",
        "`analysis/bivariate.py` — bivariate model with age-specific terms (block design on the REML/GLS engine `lmm.py`), per-age exhaustive AICc.",
        "`analysis/run_v3.py` — the whole protocol; `analysis/make_figures_v3.py` — figures G01–G09; `analysis/report_v3.py` — this report.",
        "`results/v3/v3_results.json` and `results/v3/tables/V00–V23` — every result; `results/v3/figures/` — PNG (300 dpi) and vector PDF.",
        "`report/v3/` — `index.html` (with a strength calculator), `Model_Report_v3.pdf`, `REPORT.md`.",
    ])
    D.code("pip install -r requirements.txt\npython analysis/run_v3.py          # about 5 min on 4 cores; --quick for a test run\n"
           "python analysis/make_figures_v3.py\npython analysis/report_v3.py       # needs Chromium for the PDF")
    return D


# ============================================================================ calculator (HTML only)
def calculator(R):
    d = dz.load(os.path.join(dz.ROOT, R["meta"]["data"]))
    t7, t28 = R["final_terms"]["7"], R["final_terms"]["28"]
    m, _, names = bv.fit(d, t7, t28)
    df = {a: float(m.satterthwaite_df(bv.rows(t7, t28, pd.DataFrame(
        {"RCF_pct": [30.0], "SS_pct": [37.5], "AB": [0.45], "carbonation": ["NC"], "age": [a]}))[0])) for a in (7, 28)}
    from scipy import stats
    payload = {"names": names, "beta": m.beta.tolist(), "cov": m.cov_beta.tolist(),
               "s2": [float(m.Sigma[0, 0]), float(m.Sigma[1, 1])],
               "t": {"7": float(stats.t.ppf(0.975, df[7])), "28": float(stats.t.ppf(0.975, df[28]))},
               "tg": float(stats.t.ppf(0.975, min(df.values()))),
               "mixes": [{"c": r.carbonation, "A": r.A, "C": r.C, "D": r.D} for _, r in d.iterrows()]}
    return """
<section class="calc" id="calculator">
<h2>Strength calculator</h2>
<p>Predictions of the final model for one new batch. The 95 % confidence interval (CI) is for the mean strength of the composition;
the prediction interval (PI) is for a single batch. Inputs are limited to the tested ranges.</p>
<div class="form">
<label>RCF (%) <input id="rcf" type="number" min="10" max="50" step="0.5" value="30"></label>
<label>SS (%) <input id="ss" type="number" min="0" max="75" step="0.5" value="37.5"></label>
<label>A/B <input id="ab" type="number" min="0.42" max="0.48" step="0.005" value="0.45"></label>
<label>Carbonation <select id="carb"><option>NC</option><option>0.5 h</option><option selected>1 h</option><option>5 h</option></select></label>
</div>
<div class="tiles">
<div class="tile"><div class="tl">7-day strength</div><div class="tv" id="o7">–</div><div class="ts" id="i7"></div></div>
<div class="tile"><div class="tl">28-day strength</div><div class="tv" id="o28">–</div><div class="ts" id="i28"></div></div>
<div class="tile"><div class="tl">Gain f28 / f7</div><div class="tv" id="og">–</div><div class="ts" id="ig"></div></div>
</div>
<p class="warn" id="warn" hidden></p>
</section>
<script>
const M = """ + json.dumps(payload) + """;
function colval(base, p) {
  const A = (p.rcf - 30) / 20, C = (p.ab - 0.45) / 0.03, D = (p.ss - 37.5) / 37.5, K = p.carb === "NC" ? 0 : 1;
  const m = base.match(/^Carb\\[(.+)\\]$/);
  if (m) return p.carb === m[1] ? 1 : 0;
  return {Intercept: 1, A: A, C: C, D: D, AC: A * C, AD: A * D, CD: C * D, A2: A * A, C2: C * C, D2: D * D,
          KA: K * A, KC: K * C, KD: K * D}[base];
}
function row(age, p) {
  return M.names.map(n => { const [b, a] = n.split("@"); return Number(a) === age ? colval(b, p) : 0; });
}
const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
const quad = x => x.reduce((s, xi, i) => s + xi * dot(M.cov[i], x), 0);
const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
const el = id => document.getElementById(id);
function update() {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const p = {rcf: clamp(+el("rcf").value, 10, 50), ss: clamp(+el("ss").value, 0, 75),
             ab: clamp(+el("ab").value, 0.42, 0.48), carb: el("carb").value};
  [[7, 0, "o7", "i7"], [28, 1, "o28", "i28"]].forEach(([age, j, o, i]) => {
    const x = row(age, p), f = dot(x, M.beta), se = Math.sqrt(quad(x)), t = M.t[String(age)], sp = Math.sqrt(se * se + M.s2[j]);
    el(o).textContent = f1(Math.exp(f)) + " MPa";
    el(i).textContent = "CI " + f1(Math.exp(f - t * se)) + "–" + f1(Math.exp(f + t * se)) +
      " · PI " + f1(Math.exp(f - t * sp)) + "–" + f1(Math.exp(f + t * sp));
  });
  const g = row(28, p).map((v, k) => v - row(7, p)[k]), fg = dot(g, M.beta), sg = Math.sqrt(quad(g));
  el("og").textContent = "×" + f2(Math.exp(fg));
  el("ig").textContent = "CI " + f2(Math.exp(fg - M.tg * sg)) + "–" + f2(Math.exp(fg + M.tg * sg));
  const A = (p.rcf - 30) / 20, C = (p.ab - 0.45) / 0.03, D = (p.ss - 37.5) / 37.5;
  const dist = Math.min(...M.mixes.filter(m => m.c === p.carb).map(m => Math.hypot(m.A - A, m.C - C, m.D - D)));
  el("warn").hidden = dist <= 1.0;
  el("warn").textContent = "No tested " + p.carb + " mixture is close to this composition (coded distance " + f2(dist) +
    "); treat the prediction as an extrapolation.";
}
["rcf", "ss", "ab", "carb"].forEach(id => el(id).addEventListener("input", update));
update();
</script>
"""


CSS = """
:root { color-scheme: light; --bg:#ffffff; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
  --accent:#2a78d6; --warnbg:#fff6e5; --warnline:#eda100; --okbg:#eef5fd; --code:#f4f3ef; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --warnbg:#2a2210; --warnline:#c98500;
  --okbg:#13202f; --code:#1f1f1d; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --line:#2c2c2a; --accent:#3987e5; --warnbg:#2a2210; --warnline:#c98500; --okbg:#13202f; --code:#1f1f1d; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; line-height:1.25; margin: 0.4em 0 0.2em; }
h2 { font-size: 1.2rem; margin-top: 2.2em; padding-top: 0.6em; border-top: 1px solid var(--line); }
h3 { font-size: 1.02rem; margin-top: 1.6em; }
p, li { color: var(--ink); }
em { color: var(--ink2); }
a { color: var(--accent); }
code { background: var(--code); padding: 0.1em 0.3em; border-radius: 4px; font-size: 0.88em; overflow-wrap:anywhere; }
pre { background: var(--code); padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 0.82rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
.tw { overflow-x: auto; margin: 1em 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
caption { text-align: left; color: var(--ink2); font-size: 0.85rem; padding-bottom: 6px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
figure { margin: 1.4em 0; }
figure img { width: 100%; height: auto; background: #fff; border-radius: 6px; }
figcaption { color: var(--ink2); font-size: 0.86rem; margin-top: 6px; }
.note { border-left: 4px solid var(--warnline); background: var(--warnbg); padding: 10px 16px; border-radius: 6px; margin: 1.2em 0; }
.note .nt { font-weight: 700; margin-top: 4px; }
.calc { background: var(--okbg); border-radius: 10px; padding: 4px 16px 16px; margin-top: 2em; }
.calc h2 { border-top: none; margin-top: 0.8em; }
.form { display: flex; flex-wrap: wrap; gap: 12px; }
.form label { display: flex; flex-direction: column; font-size: 0.85rem; color: var(--ink2); }
.form input, .form select { margin-top: 4px; padding: 6px 8px; font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; min-width: 120px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 14px; }
.tile { background: var(--surface); border-radius: 8px; padding: 10px 14px; }
.tl { font-size: 0.82rem; color: var(--ink2); }
.tv { font-size: 1.6rem; font-weight: 600; }
.ts { font-size: 0.8rem; color: var(--ink2); }
.warn { color: var(--ink); font-size: 0.85rem; border-left: 3px solid var(--warnline); padding-left: 8px; }
nav.toc { font-size: 0.88rem; color: var(--ink2); display: flex; flex-wrap: wrap; gap: 4px 14px; }
nav.toc a { white-space: nowrap; }
@media print { .calc, nav.toc { display: none; } body { font-size: 11px; } main { max-width: none; padding: 0; }
  h2 { break-after: avoid; } figure, table, .note { break-inside: avoid; } figure img { max-height: 9.5in; object-fit: contain; } }
"""


def write_html(D, R):
    toc = ('<nav class="toc"><a href="#summary">Summary</a><a href="#calculator">Calculator</a><a href="#data">1 Data</a>'
           '<a href="#strategy">2 Strategy</a><a href="#model">3 Model</a><a href="#trends">4 Trends</a>'
           '<a href="#interactions">5 Interactions</a><a href="#gain">6 Gain</a><a href="#optimum">7 Optimum</a>'
           '<a href="#reliability">8 Reliability</a><a href="#limits">9 Limits</a><a href="#files">10 Files</a></nav>')
    body = D.html()
    # calculator after the summary list
    i = body.find("</ol>") + len("</ol>")
    body = body[:i] + calculator(R) + body[i:]
    first_h1_end = body.find("</h1>") + len("</h1>")
    body = body[:first_h1_end] + toc + body[first_h1_end:]
    page = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Strength Model v3</title><style>" + CSS + "</style></head><body><main>" + body + "</main></body></html>\n")
    p = os.path.join(REP, "index.html")
    with open(p, "w") as f:
        f.write(page)
    return p


def write_pdf(html_path):
    chrome = None
    for c in glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome") + [shutil.which("chromium") or "", shutil.which("chromium-browser") or ""]:
        if c and os.path.exists(c):
            chrome = c
            break
    if chrome is None:
        print("Chromium not found; PDF skipped")
        return
    out = os.path.join(REP, "Model_Report_v3.pdf")
    subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", "file://" + html_path], check=False, capture_output=True, timeout=240)
    print("wrote report/v3/Model_Report_v3.pdf" if os.path.exists(out) else "PDF not written")


def main():
    with open(os.path.join(RES, "v3_results.json")) as f:
        R = json.load(f)
    for fn in glob.glob(os.path.join(RES, "figures", "G*.png")):
        shutil.copy(fn, os.path.join(REP, "figures", os.path.basename(fn)))
    D = build(R)
    with open(os.path.join(REP, "REPORT.md"), "w") as f:
        f.write(D.md())
    print("wrote report/v3/REPORT.md")
    p = write_html(D, R)
    print("wrote report/v3/index.html")
    write_pdf(p)


if __name__ == "__main__":
    main()
