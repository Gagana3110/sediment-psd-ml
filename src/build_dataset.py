"""Turn the raw USGS NetCDF casts into one tidy table of (features, targets).

Usage:  python src/build_dataset.py

Reads every zip / .nc under data/raw, applies quality control, averages the
1 Hz LISST samples into short bursts and writes data/processed/psd_dataset.csv.
CTD casts (salinity, turbidity) are merged in automatically when present.
"""
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

import config as cfg
from psd import psd_stats

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

EPIC_EPOCH = pd.Timestamp("1968-05-23")   # EPIC "true Julian day" 2440000
FILL = 1e30


def extract_zips():
    for z in sorted(cfg.RAW_DIR.glob("*.zip")):
        out = cfg.RAW_DIR / z.stem
        if not out.exists():
            print(f"extracting {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)


def epic_time(ds):
    """EPIC files store time as integer Julian day + milliseconds since midnight (GMT)."""
    days = pd.to_timedelta(ds["time"].values.astype("int64") - 2440000, unit="D")
    ms = pd.to_timedelta(ds["time2"].values.astype("int64"), unit="ms")
    return EPIC_EPOCH + days + ms


def parse_name(path):
    """CSF20PRO07ls-c.nc -> (7, 'ls', 'channel')"""
    m = re.match(r"CSF20PRO(\d+)([a-z]+)-([cs])\.nc$", path.name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), {"c": "channel", "s": "shallows"}[m.group(3)]


def squeeze(ds, name):
    v = ds[name].values.astype(float)
    v = v.reshape(v.shape[0], -1)[:, 0]
    v[v > FILL] = np.nan
    return v


def read_lisst(path):
    idx, _, site = parse_name(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        vconc = ds["vconc"].values.astype(float).reshape(ds.sizes["time"], -1)
        vconc[vconc > FILL] = np.nan
        df = pd.DataFrame({
            "time": epic_time(ds),
            "depth_m": squeeze(ds, "D_3"),
            "temp_C": squeeze(ds, "T_20"),
            "attn_m1": squeeze(ds, "ATTN_55"),
            "ptrans": squeeze(ds, "ptrans"),
            "d50_file": squeeze(ds, "D50"),
        })
        df["site"] = site
        df["profile"] = f"{site[0].upper()}{idx:02d}"
        df["water_depth_m"] = float(ds.attrs["WATER_DEPTH"])
        centers = ds["size_class"].values.astype(float)
    return df, vconc, centers


def find_var(ds, prefixes):
    for name in ds.data_vars:
        if any(name.lower().startswith(p) for p in prefixes):
            return name
    return None


def read_ctd(path):
    """Salinity / turbidity from an RBR CTD cast; variable names follow EPIC codes."""
    _, _, site = parse_name(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        out = pd.DataFrame({"time": epic_time(ds)})
        for col, prefixes in {"salinity_psu": ("s_", "sal"),
                              "turbidity_ntu": ("turb", "ntu", "nep"),
                              "ctd_depth_m": ("d_", "depth")}.items():
            name = find_var(ds, prefixes)
            if name is not None:
                out[col] = squeeze(ds, name)
    out["site"] = site
    return out


def load_raw():
    lisst, spectra, ctd, centers = [], [], [], None
    seen = {}
    for path in sorted(cfg.RAW_DIR.rglob("*.nc")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        if parsed[1] == "ls":
            df, vconc, centers = read_lisst(path)
            # one shallows file in the release is a copy of a channel cast
            key = (df["time"].iloc[0], len(df), round(float(np.nanmax(df["depth_m"])), 2))
            if key in seen:
                print(f"skipping {path.name}: duplicate of {seen[key]}")
                continue
            seen[key] = path.name
            lisst.append(df)
            spectra.append(vconc)
        elif parsed[1] == "ctd":
            ctd.append(read_ctd(path))
    if not lisst:
        raise SystemExit(f"No LISST files found under {cfg.RAW_DIR}. See README for download steps.")
    return (pd.concat(lisst, ignore_index=True), np.vstack(spectra),
            pd.concat(ctd, ignore_index=True) if ctd else None, centers)


def merge_ctd(df, ctd):
    """Attach the CTD sample nearest in time (same site, within 2 s)."""
    cols = [c for c in cfg.CTD_FEATURES if c in ctd.columns]
    df = df.sort_values("time")
    ctd = ctd.sort_values("time")[["time", "site"] + cols].dropna(subset=cols)
    merged = pd.merge_asof(df, ctd, on="time", by="site", direction="nearest",
                           tolerance=pd.Timedelta("2s"))
    print(f"CTD merged: {merged[cols[0]].notna().mean():.0%} of LISST samples matched")
    return merged.sort_index()


def quality_mask(df):
    return (
        df["depth_m"].ge(cfg.MIN_DEPTH_M)
        & df["depth_m"].le(df["water_depth_m"] + 1.0)        # deeper than the bed = bad pressure
        & df["ptrans"].between(cfg.PTRANS_MIN, cfg.PTRANS_MAX)
        & df["attn_m1"].gt(0)
    )


def add_tide_rate(bursts):
    """Rate of change of water depth between successive casts at a site, per day."""
    casts = (bursts.groupby(["site", "date", "profile"])
             .agg(t=("time", "min"), wd=("water_depth_m", "first")).reset_index())
    rates = []
    for _, g in casts.groupby(["site", "date"]):
        g = g.sort_values("t")
        hours = (g["t"] - g["t"].iloc[0]).dt.total_seconds().values / 3600
        rate = np.gradient(g["wd"].values, hours) if len(g) > 1 else np.zeros(len(g))
        rates.append(pd.DataFrame({"profile": g["profile"].values, "tide_rate_m_per_hr": rate}))
    return bursts.merge(pd.concat(rates), on="profile", how="left")


def main():
    extract_zips()
    df, vconc, ctd, centers = load_raw()
    n_raw = len(df)
    if ctd is not None:
        df = merge_ctd(df, ctd)

    keep = quality_mask(df).values
    df, vconc = df[keep].reset_index(drop=True), vconc[keep]
    print(f"1 Hz samples: {n_raw} raw -> {len(df)} after quality control")

    # average into bursts so that one row = one stable measurement
    t0 = df.groupby("profile")["time"].transform("min")
    df["burst"] = ((df["time"] - t0).dt.total_seconds() // cfg.BURST_SECONDS).astype(int)
    keys = [df["profile"], df["burst"]]
    feature_cols = ["depth_m", "temp_C", "attn_m1", "ptrans", "d50_file", "water_depth_m"]
    feature_cols += [c for c in cfg.CTD_FEATURES if c in df.columns]

    g = df.groupby(["profile", "burst"], sort=True)
    bursts = g[feature_cols].median()
    bursts["n"] = g.size()
    bursts["depth_range"] = g["depth_m"].max() - g["depth_m"].min()
    bursts["time"] = g["time"].min()
    bursts["site"] = g["site"].first()
    mean_psd = pd.DataFrame(vconc).groupby(keys, sort=True).mean()

    ok = ((bursts["n"] >= cfg.MIN_SAMPLES_PER_BURST)
          & (bursts["depth_range"] <= cfg.MAX_DEPTH_RANGE_IN_BURST)).values
    bursts, mean_psd = bursts[ok].reset_index(), mean_psd[ok].values

    bursts["d50_um"], bursts["sigma_um"] = psd_stats(mean_psd, centers)
    bursts["date"] = bursts["time"].dt.strftime("%Y-%m-%d")
    bursts["height_above_bed_m"] = (bursts["water_depth_m"] - bursts["depth_m"]).clip(lower=0)
    bursts["site_channel"] = (bursts["site"] == "channel").astype(int)
    bursts = add_tide_rate(bursts)

    good = bursts[cfg.TARGETS].notna().all(axis=1).values
    bursts, mean_psd = bursts[good].reset_index(drop=True), mean_psd[good]

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bursts.drop(columns=["burst", "n", "depth_range"]).to_csv(cfg.DATASET_CSV, index=False)
    np.savez_compressed(cfg.PSD_NPZ, vconc=mean_psd, centers=centers)

    print(f"bursts: {len(bursts)} rows from {bursts['profile'].nunique()} casts "
          f"({bursts.groupby('site')['profile'].nunique().to_dict()}) on {bursts['date'].nunique()} days")
    print(bursts[cfg.TARGETS + ["depth_m", "temp_C", "attn_m1"]].describe().round(2).T)
    r = np.corrcoef(bursts["d50_um"], bursts["d50_file"])[0, 1]
    print(f"check: our d50 vs the d50 stored in the files, r = {r:.3f}")
    print(f"wrote {cfg.DATASET_CSV.relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
