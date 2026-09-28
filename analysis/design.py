"""
Data loading, factor coding and hierarchical term bookkeeping.

Coded factors (-1 .. +1 over the design range, as in the 7-day model):
    A = RCF      (10 - 50 %)        A = (RCF - 30) / 20
    C = A/B      (0.42 - 0.48)      C = (A/B - 0.45) / 0.03
    D = SS       (0 - 75 %)         D = (SS - 37.5) / 37.5
    Carb         4-level categorical: NC (reference), 0.5 h, 1 h, 5 h
    K            carbonated indicator (1 for 0.5, 1 and 5 h); used only for the
                 carbonation x mixture-variable slopes, which are not estimable
                 level-by-level because the 0.5 h level holds only two distinct
                 mixture compositions
    Age          two-level factor, effect coded g = -1/2 (7 d), +1/2 (28 d)

With the +/-1/2 age coding, the coefficient of a term T is its effect averaged
over the two ages ("level"), and the coefficient of T.Age is the change of that
effect from 7 to 28 days, i.e. its effect on ln(f28 / f7) ("gain").
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "strength_7d_28d_corrected.csv")

CARB_LEVELS = ["NC", "0.5 h", "1 h", "5 h"]
AGE_G = np.array([-0.5, 0.5])

LEVEL_TERMS = ["A", "C", "D", "Carb", "AC", "AD", "CD", "A2", "C2", "D2", "KA", "KC", "KD"]
PARENTS = {"A": [], "C": [], "D": [], "Carb": [],
           "AC": ["A", "C"], "AD": ["A", "D"], "CD": ["C", "D"],
           "A2": ["A"], "C2": ["C"], "D2": ["D"],
           "KA": ["Carb", "A"], "KC": ["Carb", "C"], "KD": ["Carb", "D"]}
AGE_TERMS = [t + ".Age" for t in LEVEL_TERMS]
ALL_TERMS = LEVEL_TERMS + AGE_TERMS

LABEL = {"A": "RCF", "C": "A/B", "D": "SS", "Carb": "Carb", "AC": "RCF·A/B", "AD": "RCF·SS",
         "CD": "A/B·SS", "A2": "RCF²", "C2": "(A/B)²", "D2": "SS²", "KA": "K·RCF",
         "KC": "K·A/B", "KD": "K·SS", "Age": "Age"}


def label(term):
    if term.endswith(".Age"):
        return LABEL[term[:-4]] + " × Age"
    return LABEL.get(term, term)


def parents(term):
    """Parent terms required by (strong) model hierarchy."""
    if term == "Age":
        return []
    if term.endswith(".Age"):
        base = term[:-4]
        return [base, "Age"] + [p + ".Age" for p in PARENTS[base]]
    return PARENTS[term]


def is_hierarchical(terms):
    s = set(terms)
    return all(set(parents(t)) <= s for t in terms)


def removable(terms):
    """Terms that can be dropped without breaking hierarchy."""
    s = set(terms)
    return [t for t in terms if t != "Age" and not any(t in parents(u) for u in s if u != t)]


def load(path=DATA):
    d = pd.read_csv(path)
    d["A"] = (d.RCF_pct - 30) / 20
    d["C"] = (d.AB - 0.45) / 0.03
    d["D"] = (d.SS_pct - 37.5) / 37.5
    d["K"] = (d.carb_h > 0).astype(float)
    d["carb"] = pd.Categorical(d.carbonation, categories=CARB_LEVELS)
    d["l7"] = np.log(d.f7_mean)
    d["l28"] = np.log(d.f28_mean)
    d["M"] = (d.l7 + d.l28) / 2          # mixture level (mean ln strength)
    d["G"] = d.l28 - d.l7                # ln strength gain 7 -> 28 d
    d["group"] = d.replicate_group.fillna(d.mix.map(lambda m: f"S{m}"))
    return d


CODING = {"value": "cat4"}   # carbonation coding used by term_columns


def set_coding(coding):
    """'cat4'   : Carb = 4-level categorical, slopes via K (primary)
       'onoff'  : Carb = K (carbonated yes/no), slopes via K
       'numeric': Carb = duration in h (coded -1..1 over 0-5 h), slopes via duration
       'log'    : Carb = log2(1 + h) (coded -1..1), slopes via log duration"""
    assert coding in ("cat4", "onoff", "numeric", "log")
    CODING["value"] = coding


def term_columns(term, df, coding=None):
    """Columns (name -> array over mixtures) of a level term."""
    coding = coding or CODING["value"]
    A, C, D = (np.asarray(df[c], float) for c in ("A", "C", "D"))
    h = np.asarray(df["carb_h"], float) if "carb_h" in df else None
    if coding == "numeric":
        K = (h - 2.5) / 2.5
    elif coding == "log":
        lb = np.log2(1 + h)
        K = (lb - np.log2(6) / 2) / (np.log2(6) / 2)
    else:
        K = np.asarray(df["K"], float)
    carb = df["carb"].astype(str).values if "carb" in df else None
    if term == "A": return {"A": A}
    if term == "C": return {"C": C}
    if term == "D": return {"D": D}
    if term == "AC": return {"AC": A * C}
    if term == "AD": return {"AD": A * D}
    if term == "CD": return {"CD": C * D}
    if term == "A2": return {"A2": A ** 2}
    if term == "C2": return {"C2": C ** 2}
    if term == "D2": return {"D2": D ** 2}
    if term == "KA": return {"KA": K * A}
    if term == "KC": return {"KC": K * C}
    if term == "KD": return {"KD": K * D}
    if term == "Carb":
        if coding == "cat4":
            return {f"Carb[{lv}]": (carb == lv).astype(float) for lv in CARB_LEVELS[1:]}
        return {"K" if coding == "onoff" else "B": K}
    raise KeyError(term)


def design(terms, df):
    """Paired design array X (n, 2, p) and column names for a hierarchical term set.
    'Age' is always included; the intercept is always included."""
    n = len(df)
    cols, names = [np.ones((n, 2))], ["Intercept"]
    tset = list(terms)
    for t in LEVEL_TERMS:
        if t in tset:
            for nm, v in term_columns(t, df).items():
                cols.append(np.repeat(v[:, None], 2, axis=1)); names.append(nm)
    cols.append(np.tile(AGE_G, (n, 1))); names.append("Age")
    for t in LEVEL_TERMS:
        if t + ".Age" in tset:
            for nm, v in term_columns(t, df).items():
                cols.append(v[:, None] * AGE_G[None, :]); names.append(nm + ".Age")
    X = np.stack(cols, axis=2)
    return X, names


def term_of_column(col):
    """Map a design column name back to its term."""
    if col in ("Intercept", "Age"):
        return col
    age = col.endswith(".Age")
    base = col[:-4] if age else col
    if base.startswith("Carb[") or base in ("K", "B"):
        base = "Carb"
    return base + (".Age" if age else "")


def split_level_gain(terms):
    """Level (M) terms and gain (G) terms of a joint term set."""
    lev = [t for t in terms if not t.endswith(".Age") and t != "Age"]
    gain = [t[:-4] for t in terms if t.endswith(".Age")]
    return lev, gain


def mix_design(terms, df):
    """Single-row design (n, p) for a level-type term list (for M or G OLS)."""
    cols, names = [np.ones(len(df))], ["Intercept"]
    for t in LEVEL_TERMS:
        if t in terms:
            for nm, v in term_columns(t, df).items():
                cols.append(v); names.append(nm)
    return np.column_stack(cols), names


def new_points(df_new):
    """Add coded columns to a frame of new points with RCF_pct, AB, SS_pct, carbonation."""
    df = df_new.copy()
    df["A"] = (df.RCF_pct - 30) / 20
    df["C"] = (df.AB - 0.45) / 0.03
    df["D"] = (df.SS_pct - 37.5) / 37.5
    df["carb"] = pd.Categorical(df.carbonation, categories=CARB_LEVELS)
    df["K"] = (df["carb"].astype(str) != "NC").astype(float)
    df["carb_h"] = df["carb"].astype(str).map({"NC": 0.0, "0.5 h": 0.5, "1 h": 1.0, "5 h": 5.0}).astype(float)
    return df
