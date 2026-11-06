"""Junk filter for resting labels (generator 1.3.0): a small classifier on the tracker's own per-label meta,
trained on the 2026-11-06 gold set (1,000 test-day crops judged by eye; real = mosquito, false = junk / shadow /
part of a mosquito). Features are rig-independent: contrasts, support, the dark component's size in cage widths,
and how much of its segment's sampled frames the track spans. Position and zone are left out on purpose.

    features(L, cw)   -> DataFrame of FEATS for every row of a labels meta table
    score(L, cw)      -> P(real) per row (NaN when the model file is missing)
    train(rows, out)  -> fits on a features table with a 0/1 'real' column and saves model + threshold
"""
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "junk_filter_model.pkl")
META = os.path.join(HERE, "junk_filter.json")
FEATS = ["supported", "contrast", "rcontrast", "mcontrast", "edge_suspect", "comp_w_cw", "comp_h_cw",
         "comp_area_cw2", "comp_nlabels", "track_frac"]
_cache = {}


def features(L, cw):
    """Rig-independent features of every label row. `cw` = the cage width in px of that experiment."""
    import pandas as pd
    cw = float(cw)
    stem = L.frame_id.str.split("__").str[1]
    n_f = L.groupby(stem).frame_id.transform("nunique").astype(float)
    n_t = L.assign(_s=stem).groupby(["_s", "track_id"]).frame_id.transform("nunique").astype(float)
    return pd.DataFrame({
        "supported": L.supported.astype(float).values,
        "contrast": L.contrast.astype(float).values,
        "rcontrast": L.rcontrast.astype(float).values,
        "mcontrast": L.mcontrast.astype(float).values,
        "edge_suspect": L.edge_suspect.astype(float).values if "edge_suspect" in L else np.zeros(len(L)),
        "comp_w_cw": L.comp_w.astype(float).values / cw,
        "comp_h_cw": L.comp_h.astype(float).values / cw,
        "comp_area_cw2": L.comp_area.astype(float).values / cw ** 2,
        "comp_nlabels": L.comp_nlabels.astype(float).values,
        "track_frac": (n_t / n_f).values,
    }, index=L.index)


def _load():
    if "m" not in _cache:
        if not os.path.exists(MODEL):
            _cache["m"] = None
        else:
            try:
                with open(MODEL, "rb") as f:
                    m = pickle.load(f)
            except Exception as exc:  # noqa: BLE001 -- a scikit-learn pickle loads only in a matching version
                import sklearn
                raise RuntimeError(
                    "The junk-filter model %s could not be loaded with scikit-learn %s (%s: %s). It was saved "
                    "with the scikit-learn pinned in environment.yml (1.3.2): run yolo_export in buzzwatch_env, "
                    "or retrain it here with junk_filter.train()." % (MODEL, sklearn.__version__,
                                                                      type(exc).__name__, exc))
            with open(META) as f:
                _cache["meta"] = json.load(f)
            _cache["m"] = m
    return _cache["m"], _cache.get("meta")


def available():
    """True when the trained model file is present (it ships with the repository)."""
    return _load()[0] is not None


def require():
    """Raise when the model is missing: a build that claims the junk filter must not quietly skip it."""
    if not available():
        raise RuntimeError(
            "policy.junk_filter is on but the junk-filter model %s is missing, so false resting labels "
            "would go into the dataset unfiltered. Restore it from the repository (git checkout -- "
            "yolo_export/junk_filter_model.pkl) or retrain it with junk_filter.train()." % MODEL)


def threshold():
    return _load()[1]["threshold"]


def score(L, cw):
    m, _ = _load()
    if m is None or not len(L):
        return np.full(len(L), np.nan)
    return m.predict_proba(features(L, cw)[FEATS].values)[:, 1]


def train(rows, keep_real=0.85, groups=None):
    """Fit on `rows` (FEATS + 'real'), pick the threshold that keeps `keep_real` of real labels in grouped
    cross-validation, save the model and a json with the threshold and CV numbers. Returns that json."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X, y = rows[FEATS].values.astype(float), rows.real.values.astype(int)
    make = lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=150, random_state=0)  # noqa: E731
    p = cross_val_predict(make(), X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    thr = float(np.quantile(p[y == 1], 1 - keep_real))
    m = make().fit(X, y)
    meta = {"threshold": thr, "keep_real_cv": float((p[y == 1] >= thr).mean()),
            "false_removed_cv": float((p[y == 0] < thr).mean()), "auc_cv": float(roc_auc_score(y, p)),
            "n_real": int(y.sum()), "n_false": int((y == 0).sum()), "features": FEATS,
            "trained_on": "gold set 2026-11-06 (test days of Penzance_Aedes_M_20260911 and Tbambusa_M_510Hz_20261010)"}
    with open(MODEL, "wb") as f:
        pickle.dump(m, f)
    with open(META, "w") as f:
        json.dump(meta, f, indent=1)
    _cache.clear()
    return meta
