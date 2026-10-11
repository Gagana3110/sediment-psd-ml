"""Paths and constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_CSV = PROCESSED_DIR / "psd_dataset.csv"
PSD_NPZ = PROCESSED_DIR / "psd_spectra.npz"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

SEED = 42

# --- quality control -------------------------------------------------------
# Sequoia recommends trusting LISST inversions only for ~30-98 % optical
# transmission (below: multiple scattering, above: too few particles).
PTRANS_MIN, PTRANS_MAX = 30.0, 98.0
MIN_DEPTH_M = 0.3            # shallower samples are in air / bubbles
MIN_SALINITY = 20.0          # psu; the CTD reads near zero until it is submerged
BURST_SECONDS = 10           # 1 Hz samples are averaged into bursts this long
MIN_SAMPLES_PER_BURST = 6
MAX_DEPTH_RANGE_IN_BURST = 0.3   # m; rejects bursts taken while the package moves

# --- modelling -------------------------------------------------------------
TARGETS = ["d50_um", "sigma_um"]
TARGET_LABELS = {"d50_um": "d50 (µm)", "sigma_um": "σ (µm)"}

HYDRO_FEATURES = [
    "depth_m",              # sensor depth below surface
    "height_above_bed_m",   # water depth - sensor depth
    "water_depth_m",        # total depth at cast time = tidal stage
    "tide_rate_m_per_hr",   # d(water depth)/dt: flood (+) or ebb (-)
    "temp_C",
    "site_channel",         # 1 = channel site, 0 = shallows site
]
OPTICAL_FEATURES = ["attn_m1"]                       # beam attenuation, a turbidity/SSC proxy
CTD_FEATURES = ["salinity_psu", "turbidity_ntu"]     # only present if CTD zips are supplied

FEATURE_LABELS = {
    "depth_m": "depth",
    "height_above_bed_m": "height above bed",
    "water_depth_m": "water depth (tide stage)",
    "tide_rate_m_per_hr": "tide rate",
    "temp_C": "temperature",
    "site_channel": "site",
    "attn_m1": "beam attenuation",
    "salinity_psu": "salinity",
    "turbidity_ntu": "turbidity",
}
