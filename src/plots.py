"""All figures. Each function saves one PNG into figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

import config as cfg

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
TARGET_COLOR = {"d50_um": BLUE, "sigma_um": ORANGE}
SITE_COLOR = {"channel": BLUE, "shallows": ORANGE}
MODEL_COLOR = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(cfg.FIG_DIR / name)
    plt.close(fig)


def example_psd(vconc, centers, df):
    """Two measured size distributions with their d50 marked (paper Fig. 1)."""
    picks = {
        "channel": df[(df.site == "channel") & (df.height_above_bed_m < 2)].d50_um.sub(150).abs().idxmin(),
        "shallows": df[df.site == "shallows"].d50_um.sub(50).abs().idxmin(),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for site, i in picks.items():
        pct = 100 * vconc[i] / vconc[i].sum()
        c = SITE_COLOR[site]
        ax.plot(centers, pct, color=c, lw=2, marker="o", ms=3.5,
                label=f"{site}: d50 = {df.d50_um[i]:.0f} µm, σ = {df.sigma_um[i]:.0f} µm")
        ax.axvline(df.d50_um[i], color=c, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("particle diameter (µm)")
    ax.set_ylabel("volume in size class (%)")
    ax.set_title("Two measured particle size distributions (dashed line = d50)")
    ax.legend(loc="upper left")
    save(fig, "fig1_example_psd.png")


def data_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, target in zip(axes[:2], cfg.TARGETS):
        bins = np.linspace(df[target].min(), df[target].quantile(0.995), 40)
        for site, c in SITE_COLOR.items():
            ax.hist(df.loc[df.site == site, target], bins=bins, color=c, alpha=0.75, label=site)
        ax.set_xlabel(cfg.TARGET_LABELS[target])
        ax.set_ylabel("bursts")
        ax.set_title(f"Distribution of {cfg.TARGET_LABELS[target]}")
        ax.legend()
    ax = axes[2]
    for site, c in SITE_COLOR.items():
        s = df[df.site == site]
        ax.scatter(s.d50_um, s.depth_m, s=6, color=c, alpha=0.35, label=site, linewidths=0)
    ax.invert_yaxis()
    ax.set_xlabel("d50 (µm)")
    ax.set_ylabel("depth below surface (m)")
    ax.set_title("d50 through the water column")
    ax.legend(markerscale=3)
    fig.tight_layout()
    save(fig, "fig2_data_overview.png")


def feature_importance(impurity, permutation):
    order = impurity.mean(axis=1).sort_values().index
    labels = [cfg.FEATURE_LABELS[f] for f in order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(order) + 1.6), sharey=True)
    panels = [(impurity, "Random-forest importance (paper's method)", "share of impurity reduction"),
              (permutation, "Permutation importance on unseen casts", "drop in R² when the feature is shuffled")]
    for ax, (table, title, xlabel) in zip(axes, panels):
        for k, target in enumerate(cfg.TARGETS):
            ax.barh(y + (0.19 if k == 0 else -0.19), table.loc[order, target], height=0.34,
                    color=TARGET_COLOR[target], label=cfg.TARGET_LABELS[target])
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#c3c2b7", lw=1)
    axes[0].legend(title="target", loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_feature_importance.png")


def oob_curve(oob, n_trees):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for target in cfg.TARGETS:
        ax.plot(oob.index, oob[target], color=TARGET_COLOR[target], lw=2, marker="o", ms=4,
                label=cfg.TARGET_LABELS[target])
    ax.axvline(n_trees, color=MUTED, lw=1, ls="--")
    ax.annotate(f"chosen: {n_trees} trees", (n_trees, ax.get_ylim()[0]), xytext=(6, 8),
                textcoords="offset points", color=MUTED)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("out-of-bag R²")
    ax.set_title("Out-of-bag score levels off as trees are added")
    ax.legend(loc="lower right")
    save(fig, "fig4_oob_trees.png")


def svr_tuning(sweep_c, sweep_e, best):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, sweep, key, other in [(axes[0], sweep_c, "C", "epsilon"), (axes[1], sweep_e, "epsilon", "C")]:
        for target in cfg.TARGETS:
            ax.plot(sweep.index, sweep[target], color=TARGET_COLOR[target], lw=2, marker="o", ms=4,
                    label=cfg.TARGET_LABELS[target])
        ax.axvline(best[key], color=MUTED, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"{key}   (with {other} = {best[other]:g})")
        ax.set_title(f"SVR: validation R² against {key}")
    axes[0].set_ylabel("R² on validation casts")
    axes[0].set_ylim(bottom=max(-0.2, axes[0].get_ylim()[0]))
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_svr_tuning.png")


def pred_vs_true(scatter, protocol, filename, models=("Random Forest", "SVR (RBF)")):
    rng = np.random.RandomState(cfg.SEED)
    fig, axes = plt.subplots(len(models), 2, figsize=(8.4, 4 * len(models)))
    for r, model in enumerate(models):
        for c, target in enumerate(cfg.TARGETS):
            ax = axes[r, c]
            true, pred, site = scatter[(protocol, model, target)]
            show = rng.permutation(len(true))[:2500]
            for s, color in SITE_COLOR.items():
                m = site[show] == s
                ax.scatter(true[show][m], pred[show][m], s=7, color=color, alpha=0.4,
                           linewidths=0, label=s)
            lim = [0, np.quantile(true, 0.998) * 1.05]
            ax.plot(lim, lim, color=INK, lw=1)
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
            ax.set_xlabel(f"measured {cfg.TARGET_LABELS[target]}")
            ax.set_ylabel(f"predicted {cfg.TARGET_LABELS[target]}")
            ax.set_title(f"{model}: R² = {r2_score(true, pred):.2f}")
    axes[0, 0].legend(markerscale=3, loc="upper left")
    fig.suptitle(f"Predicted against measured - test protocol: {protocol}", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, filename)


def protocol_comparison(results, protocols, models):
    """R2 of every model under each test protocol, one panel per target."""
    floor = -0.3
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    width = 0.8 / len(models)
    for ax, target in zip(axes, cfg.TARGETS):
        for k, model in enumerate(models):
            sub = results[(results.model == model) & (results.target == target)].set_index("protocol")
            vals = sub.loc[protocols, "r2"].values
            x = np.arange(len(protocols)) + (k - (len(models) - 1) / 2) * width
            ax.bar(x, np.clip(vals, floor, None), width=width * 0.9, color=MODEL_COLOR[k], label=model)
            for xi, v in zip(x, vals):
                ax.text(xi, max(v, floor) + (0.015 if v >= 0 else -0.015), f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7.5, color=MUTED)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_ylim(floor - 0.08, 1.0)
        ax.set_ylabel("R² on test data")
        ax.set_title(f"Target: {cfg.TARGET_LABELS[target]}")
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(np.arange(len(protocols)), protocols)
    axes[0].legend(ncol=len(models), loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig8_protocol_comparison.png")
