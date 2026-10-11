"""Build the two-page write-up (report/writeup.pdf) from the results on disk.

Usage:  python report/make_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg

# DejaVu ships with matplotlib, so the PDF builds the same on every machine
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold")

INK, MUTED, BLUE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#1c5cab")
body = ParagraphStyle("body", fontName="Body", fontSize=8.6, leading=11.6, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=1.5)
h1 = ParagraphStyle("h1", fontName="Body-Bold", fontSize=10.5, leading=13, textColor=BLUE,
                    spaceBefore=5, spaceAfter=2)
title = ParagraphStyle("title", fontName="Body-Bold", fontSize=14.5, leading=18, textColor=INK)
sub = ParagraphStyle("sub", fontName="Body", fontSize=8.6, leading=11.5, textColor=MUTED, spaceAfter=2)
caption = ParagraphStyle("caption", parent=sub, fontSize=7.8, leading=10)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def results_table(metrics, protocols, models):
    short = {"Random 70/30 (paper)": "Random 70/30 (paper)", "Unseen casts (5-fold)": "Unseen casts",
             "Unseen day (leave-one-out)": "Unseen day", "Unseen site": "Unseen site"}
    top, second = ["Model"], [""]
    for p in protocols:
        top += [short.get(p, p), ""]
        second += ["d50", "σ"]
    rows = [top, second]
    for m in models:
        row = [m]
        for p in protocols:
            for t in cfg.TARGETS:
                v = metrics[(metrics.protocol == p) & (metrics.model == m) & (metrics.target == t)].r2.iloc[0]
                row.append(f"{v:.2f}" if v > -10 else "< −10")
        rows.append(row)
    t = Table(rows, colWidths=[3.6 * cm] + [1.75 * cm] * 8, repeatRows=2)
    style = [
        ("FONT", (0, 0), (-1, -1), "Body", 7.8), ("FONT", (0, 0), (-1, 1), "Body-Bold", 7.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (0, 0), (0, 1)),
        ("LINEBELOW", (0, 1), (-1, 1), 0.8, MUTED), ("LINEBELOW", (0, -1), (-1, -1), 0.8, MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for k in range(len(protocols)):   # each protocol heading spans its two target columns
        style.append(("SPAN", (1 + 2 * k, 0), (2 + 2 * k, 0)))
        style.append(("LINEBEFORE", (1 + 2 * k, 0), (1 + 2 * k, -1), 0.4, colors.HexColor("#c3c2b7")))
    t.setStyle(TableStyle(style))
    return t


def main():
    s = json.loads((cfg.RESULTS_DIR / "summary.json").read_text())
    metrics = pd.read_csv(cfg.RESULTS_DIR / "metrics.csv")
    ablation = pd.read_csv(cfg.RESULTS_DIR / "ablation.csv")
    protocols, models = list(s["r2"]), list(s["r2"][next(iter(s["r2"]))])
    r = lambda p, m, t: s["r2"][protocols[p]][m][t]
    rf, svr = "Random Forest", "SVR (RBF)"
    hydro = ablation[ablation.feature_set.str.startswith("Position")].set_index("target").r2

    story = [
        Paragraph("Machine learning for predicting sediment particle size distributions", title),
        Paragraph("UE24CS352A Machine Learning · Mini-project 62 · Team B-16 · "
                  "Bandugula Sai Rithesh Reddy (PES1UG24AM064), Gagana C (PES1UG24AM099)", sub),

        Paragraph("1. Problem statement", h1),
        Paragraph(
            "Mud particles suspended in estuaries stick together into flocs and break apart again within "
            "seconds to minutes. Their size sets how fast they settle, which is the most uncertain input "
            "to sediment-transport, erosion and contaminant models. No general physical model predicts floc "
            "size well. Following Egan (Stanford CS229, 2019), we ask whether a regression model can predict "
            "two summaries of the particle size distribution (PSD) from routinely measured water properties: "
            "the <b>median diameter d50</b> and the <b>spread σ</b> (the volume-weighted standard deviation "
            "of diameter, i.e. the square root of the paper's variance σ²). Success is measured, as in the "
            "paper, by the coefficient of determination R² on held-out data.", body),

        Paragraph("2. Dataset", h1),
        Paragraph(
            "The paper's own data are not public, so we used the closest public equivalent: the U.S. "
            "Geological Survey release <i>Profiles of salinity, temperature, depth, turbidity, and "
            "distributions of particle size in suspension … south San Francisco Bay, summer 2020</i> "
            "(doi:10.5066/P99Q4CHM). It covers the same bay with the same instrument family (a Sequoia "
            "LISST-200X laser particle sizer).", body),
        *bullets([
            f"<b>Coverage:</b> vertical casts at a deep channel site (11–14 m) and a shallows site (2–5 m) on "
            f"four days (21, 22, 24, 28 July 2020). The package was held about a minute at each depth.",
            "<b>Raw size:</b> 94 NetCDF files, 45,639 one-second samples after removing one file that is a "
            "duplicate of another. Each sample has the volume concentration in 36 size classes "
            "(1.2–459 µm), depth, temperature and beam attenuation.",
            f"<b>After cleaning:</b> {s['n_samples']:,} ten-second bursts from {s['n_casts']} casts.",
            "<b>What is missing:</b> the CTD files holding salinity and turbidity could not be downloaded "
            "(the published links return “page not found”), and the release has no chlorophyll or refractive "
            "index. Salinity was the paper's strongest predictor, so our feature set is weaker than theirs.",
        ]),

        Paragraph("3. Approach", h1),
        Paragraph(
            "<b>Targets.</b> We recompute d50 and σ from each burst-averaged PSD; our d50 agrees with the "
            "value stored in the files (r = 0.97). <b>Features (7).</b> Sensor depth, height above the bed, "
            "total water depth (the tidal stage), its rate of change between casts (flood or ebb), "
            "temperature, beam attenuation (a proxy for sediment concentration) and a site flag. Total "
            "volume concentration was deliberately left out because it is computed from the PSD itself.", body),
        Paragraph(
            "<b>Models.</b> We reproduce the paper's two models, Random Forest and support vector "
            "regression, and add a ridge baseline, gradient boosting and a 10-unit neural network (the "
            "architecture of Sahin et al., 2017). As in the paper, feature importance is averaged over "
            f"repeated 10-tree forests, the forest size ({s['n_trees']} trees) is chosen where the out-of-bag "
            f"score levels off, and SVR's C and ε come from a validation sweep (C = {s['svr_params']['C']:g}, "
            f"ε = {s['svr_params']['epsilon']:g} on standardised targets).", body),
        Paragraph(
            "<b>Evaluation: the main extension.</b> The paper closes by asking whether its models would work "
            "at another time or place. Consecutive samples in a cast are nearly identical, so a random split "
            "puts near-copies of every test point in the training set. We therefore report four protocols of "
            "increasing difficulty: (1) the paper's random 70/30 split, (2) five-fold cross-validation with "
            "whole casts held out, (3) leave-one-day-out, and (4) train on one site, test on the other.", body),

        Paragraph("4. Implementation", h1),
        *bullets([
            "<b>src/build_dataset.py</b> reads the NetCDF casts, decodes the EPIC time format, drops "
            "in-air samples, samples outside 30–98 % optical transmission (the instrument's valid range) "
            "and readings below the sea bed, then averages into 10-second bursts taken at a steady depth.",
            "<b>src/psd.py</b> computes d50 (log-interpolated on the cumulative volume curve) and σ.",
            "<b>src/train_evaluate.py</b> runs feature importance, tuning, all five models under all four "
            "protocols and a feature-set ablation; <b>src/plots.py</b> draws the figures.",
            "<b>demo.py</b> holds out one cast, retrains and prints predicted against measured values, or "
            "predicts for conditions typed on the command line. Python 3, scikit-learn, xarray; fixed seed.",
        ]),

        PageBreak(),
        Paragraph("5. Results", h1),
        Paragraph("R² on test data for each model and protocol (mean over splits; higher is better, "
                  "below 0 is worse than predicting the mean).", caption),
        results_table(metrics, protocols, models),
        Spacer(1, 5),
        Image(str(cfg.FIG_DIR / "fig8_protocol_comparison.png"), width=17.6 * cm, height=17.6 * cm * 0.652),
        Paragraph("Figure 1. R² of every model under the four test protocols. Bars are cut off at −0.3; "
                  "labels give the true value.", caption),
        *bullets([
            f"<b>The paper's result reproduces.</b> With a random split, Random Forest reaches R² = "
            f"{r(0, rf, 'd50_um'):.2f} for d50 and {r(0, rf, 'sigma_um'):.2f} for σ; the paper reports 0.81 "
            f"and 0.75 on different data and features.",
            f"<b>Most of that score is memorised neighbours.</b> On casts the model never saw, Random Forest "
            f"falls to {r(1, rf, 'd50_um'):.2f} and {r(1, rf, 'sigma_um'):.2f}. SVR, the smoother model, is "
            f"now best ({r(1, svr, 'd50_um'):.2f} and {r(1, svr, 'sigma_um'):.2f}); on an unseen day it scores "
            f"{r(2, svr, 'd50_um'):.2f} and {r(2, svr, 'sigma_um'):.2f}.",
            "<b>Nothing transfers between sites.</b> Every model has negative R² when trained on one site "
            "and tested on the other, because water depth and particle populations barely overlap.",
            f"<b>What drives the predictions.</b> Water depth (tide stage), sensor depth and beam attenuation "
            f"matter most; site and tide rate add almost nothing. Position, tide and temperature alone give "
            f"R² = {hydro['d50_um']:.2f} (d50) and {hydro['sigma_um']:.2f} (σ) on unseen casts with Random "
            f"Forest; adding attenuation helps σ but not d50.",
        ]),

        Paragraph("6. Conclusions", h1),
        Paragraph(
            "Machine learning can explain roughly half of the variation in floc size and spread on new casts "
            "from depth, tide, temperature and optical attenuation alone, which is useful but well short of "
            "the 0.75–0.84 a random split suggests. The headline lesson is methodological: for "
            "autocorrelated field data the split decides the score, and the paper's worry about overfitting "
            "was justified. Limits of our study: four days in one summer, two sites, and no salinity or "
            "turbidity. Adding the CTD data and the hydrodynamic measurements from the parent USGS release "
            "is the obvious next step; the pipeline merges CTD casts automatically when they "
            "are present, though that path is untested.", body),
    ]

    out = Path(__file__).with_name("writeup.pdf")
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            title="Machine learning for predicting sediment particle size distributions",
                            author="Team B-16")
    doc.build(story)
    print(f"wrote {out.relative_to(cfg.ROOT)} ({doc.page} pages)")


if __name__ == "__main__":
    main()
