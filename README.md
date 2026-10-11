# Machine learning for predicting sediment particle size distributions

UE24CS352A Machine Learning mini-project 62, team B-16:
Bandugula Sai Rithesh Reddy (PES1UG24AM064) and Gagana C (PES1UG24AM099).

We predict two summaries of the suspended-sediment particle size distribution, the
median diameter **d50** and the spread **σ**, from routinely measured water properties.
The project follows G. Egan, *Machine learning for predicting sediment particle size
distributions* (Stanford CS229, 2019) and tests the question that paper leaves open:
do the models still work on data from a different time or place?

## Results in brief

R² on test data (higher is better; below 0 is worse than predicting the mean).

| Test protocol | Random Forest d50 / σ | SVR d50 / σ |
|---|---|---|
| Random 70/30 split (the paper's protocol) | 0.86 / 0.82 | 0.69 / 0.72 |
| Unseen casts (5-fold, whole casts held out) | 0.56 / 0.58 | 0.60 / 0.62 |
| Unseen day (leave one day out) | 0.35 / 0.46 | 0.45 / 0.53 |
| Unseen site (train on one, test on the other) | -1.67 / -0.95 | -1.90 / -3.83 |

- With the paper's random split we reproduce its accuracy (the paper reports 0.81 / 0.75).
- Consecutive samples in a cast are nearly identical, so a random split lets the model
  look up the answer from a neighbour. On casts it has never seen, accuracy drops by about a third.
- No model transfers between the two sites.
- Salinity, the paper's dominant predictor, carries only about 8 % of the importance here,
  because it varied by only about 2 psu over the four days sampled.

All numbers are in `results/metrics.csv`; the write-up is `report/writeup.pdf` and the
slides are `report/slides.pptx`.

## Dataset

U.S. Geological Survey, *Profiles of salinity, temperature, depth, turbidity, and
distributions of particle size in suspension collected during four 0.25-day periods in
south San Francisco Bay, California, summer 2020*, <https://doi.org/10.5066/P99Q4CHM>.

- Vertical casts at a channel site and a shallows site on 21, 22, 24 and 28 July 2020.
- LISST-200X laser particle sizer: 94 NetCDF files, about 46,000 one-second samples of the
  36-class size spectrum, temperature and beam attenuation.
- RBR CTD with turbidity sensor: 99 NetCDF files of salinity, turbidity and depth at 6 Hz.
- After joining, quality control and 10-second burst averaging: 3,998 samples from 92 casts.
- The cleaned table is committed as `data/processed/psd_dataset.csv`, so the models and
  the demo run without downloading anything.

To rebuild from the raw files, download the four zips from the
[ScienceBase item page](https://www.sciencebase.gov/catalog/item/602a9a1ad34eb1203113b18c)
("Attached Files") and put them, still zipped, in `data/raw/`:

```
data/raw/CSF20PRO01ls_channel.zip
data/raw/CSF20PRO01ls_shallows.zip
data/raw/CSF20PRO02ctd_channel.zip
data/raw/CSF20PRO02ctd_shallows.zip
```

Download from the item page itself: the direct CTD links in the USGS metadata use an old
file name and return "page not found". Without the CTD zips the pipeline still runs, on
seven features instead of nine.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/build_dataset.py     # raw NetCDF -> data/processed/psd_dataset.csv   (needs data/raw)
python src/train_evaluate.py    # models, tuning, evaluation -> results/, figures/, models/   (~2 min)
python demo.py                  # live demo: hold out a random cast, retrain, compare
python demo.py --cast C21       # same for a chosen cast (C = channel, S = shallows)
python demo.py --predict --depth 6 --water-depth 12.5 --temp 21.5 --attn 20 --site channel
python report/make_report.py    # rebuild report/writeup.pdf
python report/make_slides.py    # rebuild report/slides.pptx
```

## Method

**Targets.** d50 and σ (volume-weighted standard deviation of diameter, the square root of
the paper's variance) are recomputed from each burst-averaged 36-class spectrum.

**Features (9).** Sensor depth, height above bed, total water depth (tide stage), rate of change
of water depth between casts (flood or ebb), temperature, salinity, turbidity, beam
attenuation, site. Total
volume concentration is excluded because it is derived from the size spectrum itself.

**Models.** Random Forest and SVR as in the paper, plus ridge regression, gradient boosting
and a 10-unit neural network. Forest size is chosen from the out-of-bag curve and SVR's
C and ε from a sweep on validation casts.

**Joining the instruments.** The LISST and CTD keep separate clocks. `build_dataset.py`
finds the offset for each day (3-7 s) by lining up their depth traces, then joins them
second by second. Depth is taken from the CTD, whose pressure sensor is the more reliable.

**Quality control.** Samples are dropped when the sensor is in air (depth < 0.3 m, or
salinity below 20 psu), optical transmission is outside 30-98 %, or there is no matching
CTD reading. One file in the release (`CSF20PRO11ls-s.nc`) is a copy of a channel cast and
is skipped.

## Repository layout

```
src/config.py           paths, thresholds, feature lists
src/build_dataset.py    NetCDF casts -> clean table
src/psd.py              d50 and σ from a size spectrum
src/train_evaluate.py   feature importance, tuning, five models x four protocols, ablation
src/plots.py            figures
demo.py                 live demonstration
data/processed/         cleaned dataset (committed)
results/                metrics, feature importance, ablation, summary
figures/                all figures
models/                 Random Forest trained on all data, used by demo.py --predict
report/                 write-up (PDF), slides (PPTX) and the scripts that build them
```

## Limitations

- Four days in one summer and two sites.
- Salinity varied by only about 2 psu, so its effect is barely tested.
- No chlorophyll, refractive index, or wave and tidal velocity, all of which the paper used.
- Beam attenuation is measured by the same instrument that measures the size distribution.
- SVR hyperparameters were chosen on one validation split of the same casts later used in
  cross-validation, so its cross-validated scores are slightly optimistic.

## References

- G. Egan (2019). Machine learning for predicting sediment particle size distributions. Stanford CS229 project report.
- R. M. Allen and S. McGill (2021). USGS data release, doi:10.5066/P99Q4CHM.
- C. Sahin et al. (2017). Floc size variability under strong turbulence: observations and artificial neural network modeling. Applied Ocean Research 68.
