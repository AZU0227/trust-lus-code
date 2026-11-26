# -*- coding: utf-8 -*-
"""Draw Figure 1 v2 (design + cohort derivation) and the licence-free Figure 2
alternative (prediction schematic). PIL only — matplotlib is not installed."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = r"C:\Users\99243\Documents\deepseek-harness\default-workspace\stage3\report\v010_2026-11-26\figures"
os.makedirs(OUT, exist_ok=True)

F = r"C:\Windows\Fonts\arial.ttf"
FB = r"C:\Windows\Fonts\arialbd.ttf"
def f(sz, bold=False):
    try:
        return ImageFont.truetype(FB if bold else F, sz)
    except Exception:
        return ImageFont.load_default()

INK = (30, 30, 30)
GREY = (110, 110, 110)
BOX = (247, 249, 252)
EDGE = (60, 90, 140)
BLUE = (74, 126, 187)
ORANGE = (240, 138, 36)
GREEN = (76, 175, 80)
RED = (192, 57, 43)


def box(d, x, y, w, h, lines, fs=15, fill=BOX, edge=EDGE, bold_first=False, align="center"):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=edge, width=2)
    fnt = f(fs)
    lh = fs + 6
    total = lh * len(lines)
    ty = y + (h - total) / 2
    for i, ln in enumerate(lines):
        ff = f(fs, True) if (bold_first and i == 0) else fnt
        tw = d.textlength(ln, font=ff)
        tx = x + (w - tw) / 2 if align == "center" else x + 10
        d.text((tx, ty + i * lh), ln, fill=INK, font=ff)


def arrow(d, x1, y1, x2, y2, color=EDGE, w=2, head=8):
    d.line([(x1, y1), (x2, y2)], fill=color, width=w)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    for s in (+1, -1):
        d.line([(x2, y2),
                (x2 - head * math.cos(ang - s * 0.45), y2 - head * math.sin(ang - s * 0.45))],
               fill=color, width=w)


# ============================ Figure 1 ============================
W1, H1 = 1500, 1200
img = Image.new("RGB", (W1, H1), "white")
d = ImageDraw.Draw(img)

d.text((40, 24), "Figure 1. Origin of predictions and gate parameters, and derivation of both cohorts",
       fill=INK, font=f(26, True))
d.text((40, 58), "A  Primary bidirectional transfer", fill=GREY, font=f(18, True))

# ---- Panel A ----
yA = 90
box(d, 70, yA, 300, 118, ["OpenPOCUS", "1,385 clips · 185 clusters", "406 / 979 B-line ±", "Butterfly 1,011 + Vave 374"])
box(d, 470, yA, 300, 118, ["COVID-BLUeS", "360 videos · 63 clusters", "234 / 126 B-line ±", "362 recorded, 2 excluded"], fill=(252, 249, 245), edge=ORANGE)
box(d, 870, yA, 250, 118, ["M3 / M4 experts", "trained on the source", "of each direction"], fill=(247, 252, 247), edge=GREEN)
box(d, 1180, yA, 270, 118, ["Gate", "median/IQR scaling", "L2 logistic, C = 0.01", "τ = 0.5"], fill=(247, 252, 247), edge=GREEN)

arrow(d, 370, yA + 40, 470, yA + 40, BLUE)
d.text((372, yA + 16), "→ target", fill=BLUE, font=f(13))
arrow(d, 470, yA + 80, 370, yA + 80, ORANGE)
d.text((372, yA + 84), "← target", fill=ORANGE, font=f(13))
arrow(d, 770, yA + 59, 870, yA + 59)
arrow(d, 1120, yA + 59, 1180, yA + 59)

d.text((40, yA + 132), "Each direction: experts trained on the source domain, gate fitted on source out-of-fold predictions, evaluated on the target domain.",
       fill=GREY, font=f(14))

# ---- Panel B ----
d.text((40, 268), "B  Device extension", fill=GREY, font=f(18, True))
yB = 300
box(d, 70, yB, 300, 112, ["COVID-BLUeS", "360 videos", "pool selection + gate fit", "(source out-of-fold)"], fill=(252, 249, 245), edge=ORANGE)
box(d, 470, yB, 300, 112, ["Butterfly", "1,011 clips · 134 clusters", "five expert checkpoints", "(M1–M5, five folds)"], fill=(247, 252, 247), edge=GREEN)
box(d, 870, yB, 300, 112, ["Vave", "374 clips · 54 clusters", "22 / 352 B-line ±", "scored by the checkpoint means"], fill=(247, 252, 247), edge=GREEN)
box(d, 1220, yB, 230, 112, ["Pool M4 + M1", "soft / hard", "routing"], fill=BOX, edge=EDGE)

arrow(d, 370, yB + 56, 470, yB + 56, ORANGE)
arrow(d, 770, yB + 56, 870, yB + 56, GREEN)
arrow(d, 1170, yB + 56, 1220, yB + 56)
d.text((40, yB + 124), "The expert checkpoints and the gate come from different sources; Vave is itself a subset of OpenPOCUS.",
       fill=GREY, font=f(14))

# ---- Panel C: derivation ----
d.text((40, 462), "C  Derivation of both cohorts", fill=GREY, font=f(18, True))

def ladder(d, x0, y0, title, steps, note):
    d.text((x0, y0), title, fill=INK, font=f(16, True))
    x = x0
    y = y0 + 32
    for i, (label, delta, sub) in enumerate(steps):
        w = 150
        is_final = (i == len(steps) - 1)
        box(d, x, y, w, 74,
            [label] + ([delta] if delta else []) + ([sub] if sub else []),
            fs=14, fill=(236, 243, 252) if is_final else BOX,
            edge=BLUE if is_final else EDGE, bold_first=True)
        if not is_final:
            d.text((x + w + 6, y + 26), "→", fill=RED, font=f(20, True))
        x += w + 34
    d.text((x0, y + 84), note, fill=GREY, font=f(13))

ladder(d, 60, 492, "OpenPOCUS",
       [("1,871", "inventory", ""), ("1,787", "− 84", "consolidation only"),
        ("1,690", "− 97", "indeterminate"), ("1,681", "− 9", "missing label"),
        ("1,385", "− 296", "other devices"), ("1,385", "analysed", "190 IDs / 185 clusters")],
       "Zone codes 1 and 3 were positive. Code 2 (consolidation alone) never entered either cohort.")

ladder(d, 60, 690, "COVID-BLUeS",
       [("362", "recorded", ""), ("360", "− 2", "< 16 frames"), ("360", "analysed", "63 IDs / 63 clusters")],
       "Eight- and seven-frame videos were below the frozen 16-frame input length.")

# split of 1,385
yS = 890
d.text((60, yS), "OpenPOCUS analysed cohort by device", fill=INK, font=f(16, True))
box(d, 60, yS + 30, 330, 86, ["Butterfly", "1,011 clips · 134 clusters", "384 / 627 B-line ±"], fill=(247, 252, 247), edge=GREEN)
box(d, 430, yS + 30, 330, 86, ["Vave", "374 clips · 54 clusters", "22 / 352 B-line ±"], fill=(247, 252, 247), edge=GREEN)
box(d, 800, yS + 30, 300, 86, ["Shared with Butterfly", "3 clusters · 24 Vave clips", "7 positive · 17 negative"], fill=(253, 240, 240), edge=RED)
box(d, 1140, yS + 30, 320, 86, ["Post hoc subset", "350 clips · 51 clusters", "15 positive · 335 negative"], fill=(253, 240, 240), edge=RED)
arrow(d, 390, yS + 73, 430, yS + 73)
arrow(d, 760, yS + 73, 800, yS + 73, RED)
arrow(d, 1100, yS + 73, 1140, yS + 73, RED)
d.text((60, yS + 128), "Panel C adds the derivation of both analysis cohorts and the shared-cluster subset used in the post hoc sensitivity analysis (Table S6).",
       fill=GREY, font=f(14))

p1 = os.path.join(OUT, "Figure1_v10.png")
img.save(p1, "PNG")
print("saved", p1, img.size)

# ============================ Figure 2 alternative ============================
W2, H2 = 1280, 720
img2 = Image.new("RGB", (W2, H2), "white")
d2 = ImageDraw.Draw(img2)
d2.text((40, 22), "Figure 2 (alternative). Expert and routed probabilities for two COVID-BLUeS target clips",
        fill=INK, font=f(24, True))
d2.text((40, 56), "No video frames are reproduced. Values are the frozen clip-level predictions from the M3/M4 gate trained on OpenPOCUS.",
        fill=GREY, font=f(15))

CASES = [
    ("A  patient_37_R2 — reference label: B-line present", 1,
     [("M3 probability", 0.63764648437500004, BLUE),
      ("M4 probability", 0.14870605468750001, ORANGE),
      ("M4 weight α", 0.8416444883481502, GREY),
      ("Routed probability", 0.22613246659793937, RED)]),
    ("B  patient_1_L1 — reference label: B-line absent", 0,
     [("M3 probability", 0.32186279296874998, BLUE),
      ("M4 probability", 0.059378051757812501, ORANGE),
      ("M4 weight α", 0.7814782451671527, GREY),
      ("Routed probability", 0.11673667802407235, RED)]),
]

for pi, (title, ylab, rows) in enumerate(CASES):
    ox = 70 if pi == 0 else 690
    d2.text((ox, 110), title, fill=INK, font=f(17, True))
    x0, x1 = ox + 200, ox + 540
    for r, (name, val, col) in enumerate(rows):
        y = 150 + r * 66
        d2.text((ox, y + 6), name, fill=INK, font=f(14))
        d2.rectangle([x0, y, x1, y + 26], fill=(240, 240, 240), outline=None)
        wpx = int((x1 - x0) * max(0.0, min(1.0, val)))
        d2.rectangle([x0, y, x0 + wpx, y + 26], fill=col, outline=None)
        d2.text((x1 + 10, y + 5), "%.3f" % val, fill=INK, font=f(14, True))
    # threshold marker
    for r in range(len(rows)):
        y = 150 + r * 66
        xm = x0 + (x1 - x0) * 0.5
        d2.line([(xm, y - 6), (xm, y + 32)], fill=GREY, width=1)
    d2.text((x0, 150 + len(rows) * 66 + 4), "Dashed line: 0.5 threshold, used for illustration only.",
            fill=GREY, font=f(12))
    # reinterpretation line
    if ylab == 1:
        d2.text((ox, 150 + len(rows) * 66 + 26),
                "M3 alone places this clip above 0.5; the 84.2% weight on M4 moves\nthe routed score below 0.5.",
                fill=INK, font=f(13))
    else:
        d2.text((ox, 150 + len(rows) * 66 + 26),
                "Both experts and the routed score stay below 0.5.",
                fill=INK, font=f(13))

d2.text((40, H2 - 58),
        "Source: COVID-BLUeS (Wiedemann et al., IEEE JBHI 2025; doi:10.1109/JBHI.2025.3543686) and its public repository. "
        "Clip identifiers and labels are from the frozen analysis cohort.",
        fill=GREY, font=f(12))
d2.text((40, H2 - 34),
        "This panel set can replace the video-frame figure without third-party image permissions: it contains no image data.",
        fill=GREY, font=f(12))

p2 = os.path.join(OUT, "Figure2_alternative_no_frames.png")
img2.save(p2, "PNG")
print("saved", p2, img2.size)
