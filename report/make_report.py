"""Build the two-page write-up (report/writeup.pdf) from the results on disk.

Usage:  python report/make_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg

# Calibri where available (Windows); otherwise DejaVu, which ships with matplotlib
WIN_FONTS = Path("C:/Windows/Fonts")
FALLBACK = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
if (WIN_FONTS / "calibri.ttf").exists():
    regular, bold = WIN_FONTS / "calibri.ttf", WIN_FONTS / "calibrib.ttf"
else:
    regular, bold = FALLBACK / "DejaVuSans.ttf", FALLBACK / "DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("Body", str(regular)))
pdfmetrics.registerFont(TTFont("Body-Bold", str(bold)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold")

INK, MUTED, BLUE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#1c5cab")
RULE, TINT = colors.HexColor("#d9d8d2"), colors.HexColor("#f1f5fb")
WIDTH = A4[0] - 4.0 * cm - 12      # frame width minus its 6 pt padding each side

body = ParagraphStyle("body", fontName="Body", fontSize=10.8, leading=15.6, textColor=INK, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=1, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=10.4, leading=14.4, spaceAfter=0)
h1 = ParagraphStyle("h1", fontName="Body-Bold", fontSize=13.5, leading=16, textColor=BLUE,
                    spaceBefore=15, spaceAfter=5)
title = ParagraphStyle("title", fontName="Body-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=4)
sub = ParagraphStyle("sub", fontName="Body", fontSize=10.2, leading=14, textColor=MUTED)
caption = ParagraphStyle("caption", parent=sub, fontSize=9.4, leading=12.5, spaceAfter=5)
tile_num = ParagraphStyle("tile_num", fontName="Body-Bold", fontSize=18, leading=21, textColor=BLUE)
tile_lab = ParagraphStyle("tile_lab", fontName="Body", fontSize=9.2, leading=11.5, textColor=MUTED)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def stat_tiles(tiles):
    """A row of headline numbers, each with a short label underneath."""
    t = Table([[[Paragraph(n, tile_num), Paragraph(lab, tile_lab)] for n, lab in tiles]],
              colWidths=[WIDTH / len(tiles)] * len(tiles))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white),
    ]))
    return t


def key_value_table(rows, key_width=3.3 * cm):
    t = Table([[Paragraph(f"<b>{k}</b>", cell), Paragraph(v, cell)] for k, v in rows],
              colWidths=[key_width, WIDTH - key_width])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    return t


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
    num = (WIDTH - 4.0 * cm) / 8
    t = Table(rows, colWidths=[4.0 * cm] + [num] * 8, repeatRows=2)
    style = [
        ("FONT", (0, 0), (-1, -1), "Body", 10.2), ("FONT", (0, 0), (-1, 1), "Body-Bold", 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (0, 0), (0, 1)), ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 1), (-1, 1), 0.8, MUTED), ("LINEBELOW", (0, -1), (-1, -1), 0.8, MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (1, 2), (2, -1), TINT),
    ]
    for k in range(len(protocols)):   # each protocol heading spans its two target columns
        style.append(("SPAN", (1 + 2 * k, 0), (2 + 2 * k, 0)))
    t.setStyle(TableStyle(style))
    return t


def main():
    s = json.loads((cfg.RESULTS_DIR / "summary.json").read_text())
    metrics = pd.read_csv(cfg.RESULTS_DIR / "metrics.csv")
    ablation = pd.read_csv(cfg.RESULTS_DIR / "ablation.csv")
    imp = pd.read_csv(cfg.RESULTS_DIR / "feature_importance.csv", index_col=0)
    data = pd.read_csv(cfg.DATASET_CSV)
    protocols, models = list(s["r2"]), list(s["r2"][next(iter(s["r2"]))])
    r = lambda p, m, t: s["r2"][protocols[p]][m][t]
    rf, svr = "Random Forest", "SVR (RBF)"
    abl = lambda prefix: ablation[ablation.feature_set.str.startswith(prefix)].set_index("target").r2
    hydro, attn, ctd = abl("Position"), abl("+ beam"), abl("+ salinity")
    sal_imp = 100 * imp.loc["salinity_psu", ["impurity_d50_um", "impurity_sigma_um"]]
    sal_lo, sal_hi = data.salinity_psu.min(), data.salinity_psu.max()
    n_casts = data.groupby("site").profile.nunique()
    best_rand = max(r(0, m, t) for m in models for t in cfg.TARGETS)
    low_rand = min(r(0, rf, t) for t in cfg.TARGETS)

    story = [
        Paragraph("Machine learning for predicting sediment particle size distributions", title),
        Paragraph("UE24CS352A Machine Learning · Mini-project 62 · Team B-16", sub),
        Paragraph("Bandugula Sai Rithesh Reddy (PES1UG24AM064) · Gagana C (PES1UG24AM099)", sub),
        Spacer(1, 7),
        HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=9),
        stat_tiles([
            (f"{s['n_samples']:,}", f"samples from {s['n_casts']} casts"),
            (f"{len(s['features'])}", "input features"),
            (f"{r(0, rf, 'd50_um'):.2f} / {r(0, rf, 'sigma_um'):.2f}", "R² (d50 / σ), random split"),
            (f"{r(1, svr, 'd50_um'):.2f} / {r(1, svr, 'sigma_um'):.2f}", "R² (d50 / σ), unseen casts"),
        ]),

        Paragraph("1. Problem statement", h1),
        Paragraph(
            "Suspended mud particles clump into flocs and break apart within seconds. Their size sets how "
            "fast they settle, and no physical model predicts it well. Following Egan (Stanford CS229, "
            "2019), we use regression to predict the <b>median particle diameter d50</b> and the "
            "<b>spread σ</b> of the size distribution from routine water measurements, scored by R² on "
            "held-out data.", body),

        Paragraph("2. Dataset", h1),
        Paragraph(
            "The paper's data are not public. We used a U.S. Geological Survey release from the same bay "
            "(doi:10.5066/P99Q4CHM).", body),
        key_value_table([
            ("Coverage", "Vertical casts at a channel site (11–14 m deep) and a shallows site (2–5 m) "
                         "in south San Francisco Bay, on four days in July 2020."),
            ("Instruments", "Laser particle sizer (36 size classes, 1.2–459 µm) and a CTD "
                            "(salinity, turbidity, depth)."),
            ("Size", f"193 raw files; after cleaning, {s['n_samples']:,} ten-second samples from "
                     f"{s['n_casts']} casts."),
            ("Not available", "Chlorophyll, refractive index, wave and tidal velocity (used in the paper)."),
        ]),

        Paragraph("3. Approach", h1),
        *bullets([
            f"<b>Features ({len(s['features'])}).</b> Depth, height above bed, water depth (tide stage), "
            f"tide rate, temperature, salinity, turbidity, beam attenuation, site.",
            f"<b>Models.</b> Random Forest and SVR as in the paper, plus ridge regression, gradient "
            f"boosting and a small neural network. Tuned as in the paper.",
            "<b>Evaluation, our main extension.</b> Neighbouring samples are nearly identical, so a random "
            "split flatters the model. We test four ways: random 70/30 (the paper's), unseen casts, "
            "an unseen day, and an unseen site.",
        ]),

        Paragraph("4. Implementation", h1),
        key_value_table([
            ("build_dataset.py", "Reads the raw files, aligns the two instruments' clocks, removes bad "
                                 "samples, averages into 10-second bursts, computes d50 and σ."),
            ("train_evaluate.py", "Feature importance, tuning, five models, four test protocols."),
            ("demo.py", "Holds out one cast, retrains, compares predicted with measured."),
        ], key_width=3.6 * cm),

        PageBreak(),
        Paragraph("5. Results", h1),
        Paragraph("R² on test data. Higher is better; below 0 is worse than predicting the mean. "
                  "Shaded: the paper's protocol.", caption),
        results_table(metrics, protocols, models),
        Spacer(1, 10),
        Image(str(cfg.FIG_DIR / "fig8_protocol_comparison.png"), width=14.2 * cm, height=14.2 * cm * 0.652),
        Paragraph("Figure 1. R² of every model under the four test protocols (bars cut off at −0.3).", caption),
        Spacer(1, 2),
        *bullets([
            f"<b>The paper's result reproduces.</b> Random split, Random Forest: R² = "
            f"{r(0, rf, 'd50_um'):.2f} (d50) and {r(0, rf, 'sigma_um'):.2f} (σ). The paper reports 0.81 and 0.75.",
            f"<b>Part of that is memorised neighbours.</b> On unseen casts the best model (SVR) scores "
            f"{r(1, svr, 'd50_um'):.2f} and {r(1, svr, 'sigma_um'):.2f}.",
            "<b>Nothing transfers between sites.</b> Every model scores below zero.",
            f"<b>Salinity matters little here.</b> It carries {sal_imp.iloc[0]:.0f} % of the importance, "
            f"against 72 % in the paper, because it only ranged from {sal_lo:.1f} to {sal_hi:.1f} psu. "
            f"Depth, tide stage and beam attenuation matter most.",
        ]),

        Paragraph("6. Conclusions", h1),
        Paragraph(
            f"Machine learning explains about {100 * r(1, svr, 'd50_um'):.0f} % of the variation in floc "
            f"size on new casts: useful, but well short of what a random split suggests. For field data "
            f"like this the split decides the score, so the paper's worry about overfitting was justified. "
            f"Limits: four days, two sites, a narrow salinity range. Next: add current and wave data and "
            f"more sites.", body),
    ]

    out = Path(__file__).with_name("writeup.pdf")
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.5 * cm,
                            title="Machine learning for predicting sediment particle size distributions",
                            author="Team B-16")
    doc.build(story)
    print(f"wrote {out.relative_to(cfg.ROOT)} ({doc.page} pages)")


if __name__ == "__main__":
    main()
