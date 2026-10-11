"""Live demo.

1) Hold out one whole cast, train on the rest, predict its profile:
       python demo.py                  (random cast)
       python demo.py --cast C21       (a specific cast; C = channel, S = shallows)

2) Predict for conditions you type in, using the saved model:
       python demo.py --predict --depth 6 --water-depth 12.5 --temp 21.5 --attn 20 --site channel
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import config as cfg
import plots


def holdout_cast(cast):
    df = pd.read_csv(cfg.DATASET_CSV)
    bundle = joblib.load(cfg.MODEL_DIR / "psd_models.joblib")
    features = bundle["features"]
    df = df.dropna(subset=features + cfg.TARGETS)
    if cast is None:
        cast = np.random.default_rng().choice(df["profile"].unique())
    if cast not in set(df["profile"]):
        raise SystemExit(f"Unknown cast {cast!r}. Available: {', '.join(sorted(df['profile'].unique()))}")

    test, train = df[df["profile"] == cast].sort_values("depth_m"), df[df["profile"] != cast]
    print(f"\nHeld-out cast {cast}: {test['site'].iloc[0]} site, {test['time'].iloc[0][:16]} GMT, "
          f"water depth {test['water_depth_m'].iloc[0]:.1f} m, {len(test)} bursts")
    print(f"Training on the other {train['profile'].nunique()} casts ({len(train)} bursts) ...\n")

    out = test[["depth_m"]].round(2).copy()
    for target in cfg.TARGETS:
        model = clone(bundle["models"][target]).fit(train[features].values, train[target].values)
        out[f"{target}_measured"] = test[target].round(1)
        out[f"{target}_predicted"] = model.predict(test[features].values).round(1)

    # one row per depth stop keeps the table readable
    table = out.groupby(out["depth_m"].round(0)).mean().round(1).drop(columns="depth_m")
    table.index.name = "depth (m)"
    print(table.to_string())
    print()
    for target in cfg.TARGETS:
        err = (out[f"{target}_predicted"] - out[f"{target}_measured"]).abs().mean()
        pct = 100 * err / out[f"{target}_measured"].mean()
        print(f"{cfg.TARGET_LABELS[target]:>9}: mean abs error {err:5.1f} µm  ({pct:.0f} % of the cast's mean)")

    plt = plots.plt
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    for ax, target in zip(axes, cfg.TARGETS):
        ax.scatter(out[f"{target}_measured"], out["depth_m"], s=14, color=plots.MUTED, alpha=0.6, label="measured")
        ax.scatter(out[f"{target}_predicted"], out["depth_m"], s=14, color=plots.TARGET_COLOR[target],
                   alpha=0.8, label="predicted")
        ax.set_xlabel(cfg.TARGET_LABELS[target])
        ax.legend()
    axes[0].invert_yaxis()
    axes[0].set_ylabel("depth below surface (m)")
    fig.suptitle(f"Cast {cast} (never seen in training)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    path = cfg.FIG_DIR / f"demo_cast_{cast}.png"
    fig.savefig(path)
    print(f"\nprofile plot saved to {path.relative_to(cfg.ROOT)}")


def predict_custom(args):
    bundle = joblib.load(cfg.MODEL_DIR / "psd_models.joblib")
    ranges = bundle["ranges"]
    row = {
        "depth_m": args.depth,
        "water_depth_m": args.water_depth,
        "height_above_bed_m": max(args.water_depth - args.depth, 0),
        "tide_rate_m_per_hr": args.tide_rate,
        "temp_C": args.temp,
        "attn_m1": args.attn,
        "site_channel": int(args.site == "channel"),
        "salinity_psu": args.salinity,
        "turbidity_ntu": args.turbidity,
    }
    x = []
    for f in bundle["features"]:
        v = row[f] if row[f] is not None else ranges[f]["median"]
        if not ranges[f]["min"] <= v <= ranges[f]["max"]:
            print(f"warning: {f} = {v} is outside the training range "
                  f"[{ranges[f]['min']:.2f}, {ranges[f]['max']:.2f}] - the prediction is an extrapolation")
        x.append(v)
    print("\ninputs: " + ", ".join(f"{f} = {v:g}" for f, v in zip(bundle["features"], x)))
    for target in cfg.TARGETS:
        print(f"predicted {cfg.TARGET_LABELS[target]}: {bundle['models'][target].predict([x])[0]:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cast", help="cast to hold out, e.g. C21 or S09 (default: random)")
    p.add_argument("--predict", action="store_true", help="predict for the conditions given below")
    p.add_argument("--depth", type=float, default=5.0, help="sensor depth below surface, m")
    p.add_argument("--water-depth", type=float, default=12.0, help="total water depth, m")
    p.add_argument("--tide-rate", type=float, default=0.0, help="m per hour, + rising / - falling")
    p.add_argument("--temp", type=float, default=21.5, help="water temperature, deg C")
    p.add_argument("--attn", type=float, default=15.0, help="beam attenuation, 1/m")
    p.add_argument("--site", choices=["channel", "shallows"], default="channel")
    p.add_argument("--salinity", type=float, help="psu (only used when the model was trained with CTD data)")
    p.add_argument("--turbidity", type=float, help="NTU (only used when the model was trained with CTD data)")
    args = p.parse_args()
    if not cfg.DATASET_CSV.exists() or not (cfg.MODEL_DIR / "psd_models.joblib").exists():
        raise SystemExit("Run `python src/build_dataset.py` and `python src/train_evaluate.py` first.")
    predict_custom(args) if args.predict else holdout_cast(args.cast)
