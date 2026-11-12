"""
figurer_praediktion.py – Figurer til prædiktionsafsnittet, lavet ud fra de gemte tabeller

Kører ingen modeller igen – læser kun tabel_17 (praediktion_v2.py) og tabel_20
(praediktion_tidstro.py) fra figurer_og_tabeller. Tager få sekunder.

Output i figurer_og_tabeller:
  fig_13_mdape_modeller.png        (rettet udgave: korrekte akseværdier, forklaring uden overlap)
  fig_17_praediktionstrappe.png    (ny, til hovedteksten: modellerne trin for trin med samtidige
                                    salg og tidstro, mod den niveaukorrigerede vurdering)

Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from faelles import BLAA, INK, MUTED, dk, overskrift

MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data
UD = MAPPE.parent / "figurer_og_tabeller"
ORANGE = "#eb6834"
BLAA_LYS = "#9cc3f0"


def laes(navn):
    return pd.read_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")


t17 = laes("tabel_17_praediktion_v2.csv").set_index(["Testdata", "Model"])
t20 = laes("tabel_20_tidstro.csv").set_index(["Variant", "Model"])

# =============================================================================
# FIGUR 13 (RETTET): MdAPE PR. MODEL, OPDELT PÅ TIDLIGERE SALG
# =============================================================================
overskrift("Figur 13 (rettet)")
modeller = ["M4", "M4 + vej", "Gradient boosting (basis)", "Gradient boosting (fuld)",
            "Offentlig vurdering", "Vurdering, niveaukorrigeret"]
grupper = [("Alle testhandler", INK), ("Huse solgt før (siden 2015)", BLAA),
           ("Huse ikke solgt før", ORANGE)]
fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
yy = np.arange(len(modeller))
for j, (g, farve) in enumerate(grupper):
    v = [t17.loc[(g, m), "MdAPE (%)"] for m in modeller]
    ax.barh(yy + (j - 1) * 0.27, v, height=0.26, color=farve, label=g)
    for yi, val in zip(yy, v):
        ax.text(val + 0.15, yi + (j - 1) * 0.27, dk(val, 1), va="center", fontsize=6.5, color=farve)
ax.set_yticks(yy, modeller, fontsize=8)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 21)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_xlabel("Median absolut %-fejl (MdAPE) på testdata – lavere er bedre")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=3)
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Præcision: modeller med vejniveau og tidligere salg mod den offentlige vurdering\n"
# "(modellerne trænet på salg fra hele 2019–2026)", loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_13_mdape_modeller.png")
plt.close(fig)
print("  gemt: fig_13_mdape_modeller.png")

# =============================================================================
# FIGUR 17 (NY): PRÆDIKTIONSTRAPPEN – MED SAMTIDIGE SALG OG TIDSTRO
# =============================================================================
# Begge versioner sammenlignes med den niveaukorrigerede vurdering pr. år (samme tal i begge,
# da vurderingen ikke afhænger af modellens træningsdata). Tidstro = variant b i tabel 20.
overskrift("Figur 17 (ny): prædiktionstrappen")
trin = [("M4", "M4", "M4", "1) Hedonisk model (M4)"),
        ("M4 + vej", "M4 + vej", "M4 + vej", "2) + prisniveau på vejen"),
        ("Gradient boosting (fuld)", "Gradient boosting (fuld)", "Gradient boosting (fuld)",
         "3) Gradient boosting\n    + vej og tidligere salg"),
        ("Vurdering, niveaukorrigeret", "Offentlig vurdering", None, "Offentlig vurdering\n(niveaukorrigeret)")]
ekspost, tidstro, labels = [], [], []
for m17, m20, _, lab in trin:
    ekspost.append(t17.loc[("Alle testhandler", m17), "MdAPE (%)"])
    tidstro.append(t20.loc[("b) Niveau pr. år", m20), "MdAPE (%)"])
    labels.append(lab)
for lab, e, t in zip(labels, ekspost, tidstro):
    print(f"{lab.replace(chr(10), ' '):45s} samtidige salg: {e:5.2f} %   tidstro: {t:5.2f} %")

fig, ax = plt.subplots(figsize=(16 / 2.54, 8.5 / 2.54))
yy = np.arange(len(trin))
er_vurd = np.array([False, False, False, True])
for j, (vals, farve, navn) in enumerate([(ekspost, BLAA_LYS, "Trænet på salg fra hele perioden 2019–2026"),
                                         (tidstro, BLAA, "Tidstro: trænet kun på salg før vurderingsdatoen (niveau pr. år)")]):
    pos = yy + (j - 0.5) * 0.36
    # vurderingen er den samme i begge versioner – den tegnes som én søjle nedenfor
    ax.barh(pos[~er_vurd], np.array(vals)[~er_vurd], height=0.34, color=farve, label=navn)
    for p, v, vurd in zip(pos, vals, er_vurd):
        if not vurd:
            ax.text(v + 0.15, p, dk(v, 1), va="center", fontsize=7, color=INK)
ax.barh(yy[er_vurd], np.array(tidstro)[er_vurd], height=0.5, color=ORANGE,
        label="Offentlig vurdering (niveaukorrigeret pr. år)")
ax.text(tidstro[-1] + 0.15, yy[-1], dk(tidstro[-1], 1), va="center", fontsize=7, color=INK)
ax.axvline(tidstro[-1], color=ORANGE, lw=0.8, ls="--")
ax.set_yticks(yy, labels, fontsize=8)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 20.5)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_xlabel("Median absolut %-fejl (MdAPE) på testdata – lavere er bedre")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=1)
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Fra hedonisk model til prædiktionsmodel: præcision trin for trin\n"
# "36.636 testhandler (salg i 2020 og 2022)", loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_17_praediktionstrappe.png")
plt.close(fig)
print("  gemt: fig_17_praediktionstrappe.png")
print(f"\nAlle filer ligger i: {UD}")
