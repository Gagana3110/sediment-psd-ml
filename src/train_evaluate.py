"""Train and evaluate models that predict d50 and PSD spread from water-column data.

Usage:  python src/train_evaluate.py

Follows the reference paper (Random Forest feature selection + OOB tuning, SVR
with a C / epsilon sweep) and extends it with gradient boosting, a small neural
net and - the paper's open question - tests of how well the models generalise to
unseen casts, unseen days and an unseen site.

Writes results/*.csv, results/summary.json, figures/*.png, models/psd_models.joblib
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut,
                                     ShuffleSplit)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config as cfg
import plots

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_IMPORTANCE_RUNS = 200        # paper: 1000 runs of a 10-tree forest
TREE_GRID = [10, 15, 20, 25, 32, 40, 50, 65, 80, 100, 150, 200]
C_GRID = 2.0 ** np.arange(-5, 9)
EPS_GRID = 2.0 ** np.arange(-7, 2)


def scaled(estimator):
    """Standardise inputs and the target (SVR and the neural net need both)."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler())


def make_models(n_trees, svr_params):
    return {
        "Ridge (linear baseline)": scaled(Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=2, n_jobs=-1, random_state=cfg.SEED),
        "SVR (RBF)": scaled(SVR(kernel="rbf", **svr_params)),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=cfg.SEED),
        "Neural net (10 units)": scaled(MLPRegressor(
            hidden_layer_sizes=(10,), alpha=1e-2, max_iter=2000, random_state=cfg.SEED)),
    }


def metrics(y, pred):
    return {"r2": r2_score(y, pred), "rmse": root_mean_squared_error(y, pred),
            "mae": mean_absolute_error(y, pred)}


# --- evaluation protocols --------------------------------------------------
def protocol_splits(df):
    """name -> list of (train_idx, test_idx). Ordered from easiest to hardest."""
    idx = np.arange(len(df))
    casts, days = df["profile"].values, df["date"].values
    channel = df["site_channel"].values == 1
    return {
        "Random 70/30 (paper)": list(ShuffleSplit(5, test_size=0.3, random_state=cfg.SEED).split(idx)),
        "Unseen casts (5-fold)": list(GroupKFold(5).split(idx, groups=casts)),
        "Unseen day (leave-one-out)": list(LeaveOneGroupOut().split(idx, groups=days)),
        "Unseen site": [(idx[channel], idx[~channel]), (idx[~channel], idx[channel])],
    }


def evaluate(model, X, y, splits):
    """Mean test metrics over splits, plus pooled out-of-sample predictions."""
    rows, true, pred, where = [], [], [], []
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append(metrics(y[te], p))
        true.append(y[te]); pred.append(p); where.append(te)
    out = pd.DataFrame(rows).mean().to_dict()
    out["r2_min"], out["r2_max"] = min(r["r2"] for r in rows), max(r["r2"] for r in rows)
    return out, np.concatenate(true), np.concatenate(pred), np.concatenate(where)


# --- steps taken from the paper -------------------------------------------
def rf_feature_importance(X, Y, features):
    """Impurity importance averaged over many small forests on random 70/30 splits."""
    rng = np.random.RandomState(cfg.SEED)
    table = {}
    for j, target in enumerate(cfg.TARGETS):
        imp = np.zeros(len(features))
        for _ in range(N_IMPORTANCE_RUNS):
            tr = rng.permutation(len(X))[: int(0.7 * len(X))]
            rf = RandomForestRegressor(n_estimators=10, random_state=rng.randint(1 << 30))
            imp += rf.fit(X[tr], Y[tr, j]).feature_importances_
        table[target] = imp / N_IMPORTANCE_RUNS
    return pd.DataFrame(table, index=features)


def grouped_permutation_importance(X, Y, features, groups, n_trees):
    """How much held-out R2 drops when one feature is shuffled (unseen casts)."""
    table = {}
    for j, target in enumerate(cfg.TARGETS):
        drops = []
        for tr, te in GroupKFold(5).split(X, groups=groups):
            rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, n_jobs=-1,
                                       random_state=cfg.SEED).fit(X[tr], Y[tr, j])
            r = permutation_importance(rf, X[te], Y[te, j], n_repeats=10, scoring="r2",
                                       random_state=cfg.SEED)
            drops.append(r.importances_mean)
        table[target] = np.mean(drops, axis=0)
    return pd.DataFrame(table, index=features)


def oob_curve(X, Y):
    """OOB R2 as trees are added; training set is 70 % of the data as in the paper."""
    tr, _ = next(ShuffleSplit(1, test_size=0.3, random_state=cfg.SEED).split(X))
    curve = {t: [] for t in cfg.TARGETS}
    for n in TREE_GRID:
        for j, target in enumerate(cfg.TARGETS):
            rf = RandomForestRegressor(n_estimators=n, oob_score=True, min_samples_leaf=2,
                                       n_jobs=-1, random_state=cfg.SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # tiny forests leave some rows without OOB votes
                curve[target].append(rf.fit(X[tr], Y[tr, j]).oob_score_)
    curve = pd.DataFrame(curve, index=pd.Index(TREE_GRID, name="n_trees"))
    # smallest forest within 0.01 of the best mean OOB score
    mean = curve.mean(axis=1)
    n_trees = int(mean.index[(mean >= mean.max() - 0.01).values][0])
    return curve, n_trees


def tune_svr(X, Y, groups):
    """C and epsilon sweeps on a validation set made of casts the model never saw."""
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=cfg.SEED).split(X, groups=groups))

    def val_r2(C, eps):
        return [r2_score(Y[va, j], scaled(SVR(C=C, epsilon=eps)).fit(X[tr], Y[tr, j]).predict(X[va]))
                for j in range(Y.shape[1])]

    grid = np.array([[val_r2(C, e) for e in EPS_GRID] for C in C_GRID])   # (C, eps, target)
    mean = grid.mean(axis=2)
    ci, ei = np.unravel_index(mean.argmax(), mean.shape)
    best = {"C": float(C_GRID[ci]), "epsilon": float(EPS_GRID[ei])}
    sweep_c = pd.DataFrame(grid[:, ei, :], index=pd.Index(C_GRID, name="C"), columns=cfg.TARGETS)
    sweep_e = pd.DataFrame(grid[ci, :, :], index=pd.Index(EPS_GRID, name="epsilon"), columns=cfg.TARGETS)
    return best, sweep_c, sweep_e


def main():
    for d in (cfg.FIG_DIR, cfg.RESULTS_DIR, cfg.MODEL_DIR):
        d.mkdir(exist_ok=True)

    df = pd.read_csv(cfg.DATASET_CSV)
    has_ctd = all(c in df.columns and df[c].notna().mean() > 0.8 for c in cfg.CTD_FEATURES)
    features = cfg.HYDRO_FEATURES + cfg.OPTICAL_FEATURES + (cfg.CTD_FEATURES if has_ctd else [])
    df = df.dropna(subset=features + cfg.TARGETS).reset_index(drop=True)
    X, Y, casts = df[features].values, df[cfg.TARGETS].values, df["profile"].values
    print(f"{len(df)} samples, {df['profile'].nunique()} casts, {len(features)} features"
          f" (CTD {'included' if has_ctd else 'not available'})")

    psd = np.load(cfg.PSD_NPZ)
    plots.example_psd(psd["vconc"], psd["centers"], pd.read_csv(cfg.DATASET_CSV))
    plots.data_overview(df)

    print("feature importance ...")
    impurity = rf_feature_importance(X, Y, features)
    print("OOB curve ...")
    oob, n_trees = oob_curve(X, Y)
    permutation = grouped_permutation_importance(X, Y, features, casts, n_trees)
    importance = impurity.add_prefix("impurity_").join(permutation.add_prefix("permutation_"))
    importance.to_csv(cfg.RESULTS_DIR / "feature_importance.csv", index_label="feature")
    plots.feature_importance(impurity, permutation)
    oob.to_csv(cfg.RESULTS_DIR / "oob_curve.csv")
    plots.oob_curve(oob, n_trees)
    print(f"  chosen number of trees: {n_trees}")

    print("SVR tuning ...")
    svr_params, sweep_c, sweep_e = tune_svr(X, Y, casts)
    plots.svr_tuning(sweep_c, sweep_e, svr_params)
    print(f"  chosen {svr_params}")

    print("model comparison ...")
    models = make_models(n_trees, svr_params)
    splits = protocol_splits(df)
    rows, scatter = [], {}
    for protocol, sp in splits.items():
        for name, model in models.items():
            for j, target in enumerate(cfg.TARGETS):
                m, true, pred, where = evaluate(model, X, Y[:, j], sp)
                rows.append({"protocol": protocol, "model": name, "target": target, **m})
                scatter[(protocol, name, target)] = (true, pred, df["site"].values[where])
    results = pd.DataFrame(rows)
    results.to_csv(cfg.RESULTS_DIR / "metrics.csv", index=False)
    table = results.pivot_table(index=["protocol", "model"], columns="target", values="r2", sort=False)
    print(table.round(3).to_string())

    protocols = list(splits)
    plots.pred_vs_true(scatter, protocols[0], "fig6_pred_vs_true_random.png")
    plots.pred_vs_true(scatter, protocols[1], "fig7_pred_vs_true_unseen_casts.png")
    plots.protocol_comparison(results, protocols, list(models))

    print("feature-set ablation (Random Forest, unseen casts) ...")
    sets = {"Position + tide + temperature": cfg.HYDRO_FEATURES,
            "+ beam attenuation": cfg.HYDRO_FEATURES + cfg.OPTICAL_FEATURES}
    if has_ctd:
        sets["+ salinity + turbidity (CTD)"] = features
    top4 = list(impurity.mean(axis=1).sort_values(ascending=False).index[:4])
    sets["Top-4 by RF importance"] = top4
    ablation = []
    for label, cols in sets.items():
        for j, target in enumerate(cfg.TARGETS):
            m, *_ = evaluate(models["Random Forest"], df[cols].values, Y[:, j], splits[protocols[1]])
            ablation.append({"feature_set": label, "n_features": len(cols), "target": target, **m})
    ablation = pd.DataFrame(ablation)
    ablation.to_csv(cfg.RESULTS_DIR / "ablation.csv", index=False)
    print(ablation.pivot_table(index="feature_set", columns="target", values="r2", sort=False).round(3).to_string())

    # final models for the demo: trained on everything
    final = {t: clone(models["Random Forest"]).fit(X, Y[:, j]) for j, t in enumerate(cfg.TARGETS)}
    joblib.dump({"models": final, "features": features,
                 "ranges": df[features].agg(["min", "median", "max"]).to_dict()},
                cfg.MODEL_DIR / "psd_models.joblib", compress=3)

    def r2(protocol, model, target):
        q = results.query("protocol == @protocol and model == @model and target == @target")
        return round(float(q["r2"].iloc[0]), 3)

    summary = {
        "n_samples": len(df), "n_casts": int(df["profile"].nunique()),
        "n_days": int(df["date"].nunique()), "features": features, "has_ctd": has_ctd,
        "n_trees": n_trees, "svr_params": svr_params, "top4_features": top4,
        "r2": {p: {m: {t: r2(p, m, t) for t in cfg.TARGETS} for m in models} for p in protocols},
    }
    (cfg.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done - see results/ and figures/")


if __name__ == "__main__":
    main()
