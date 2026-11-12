"""
hovedmodel.py – Hedonisk prismodel (analyse 5.1: prisdrivere)

Estimerer fire OLS-modeller af ln(salgspris) på hovedperioden 2019+, så man kan se,
hvordan koefficienterne ændrer sig, når der kontrolleres for mere:

  M1  ln(pris) ~ ln(boligareal)                                 (simpel sammenhæng)
  M2  + øvrige boligkarakteristika                              (ingen tid/beliggenhed)
  M3  + kvartals-dummies (faste effekter for salgskvartal)
  M4  + postnummer-dummies (faste effekter for beliggenhed)     = HOVEDMODEL

Output i mappen figurer_og_tabeller:
  tabel_5_regression.csv        koefficienter (robuste std.fejl) for M1–M4
  tabel_6_effekter_M4.csv       hovedmodellen fortolket: %-effekt, 95 %-KI, kr. ved medianpris,
                                robuste og klyngerobuste (postnummer) std.fejl
  tabel_7_vif.csv               multikollinearitet (VIF) for boligkarakteristikaene
  fig_5_koefficientplot.png     %-effekter med 95 %-KI fra hovedmodellen

Kræver: pip3 install pyfixest statsmodels. faelles.py skal ligge i samme mappe.
Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Send outputtet i terminalen tilbage.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.outliers_influence import variance_inflation_factor

from faelles import (BLAA, GRID, INK, INK2, MUTED, REF, dk, forbered_data, overskrift)

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


# =============================================================================
# DATA OG VARIABLE (se faelles.py)
# =============================================================================
df, bolig = forbered_data(MAPPE)

# =============================================================================
# MODELLERNE
# =============================================================================
overskrift("Estimation")
formler = {
    "M1": "ln_pris ~ ln_boligareal",
    "M2": f"ln_pris ~ {bolig}",
    "M3": f"ln_pris ~ {bolig} | kvartal",
    "M4": f"ln_pris ~ {bolig} | kvartal + postnr",
}
modeller = {}
for navn, f in formler.items():
    modeller[navn] = pf.feols(f, data=df, vcov="hetero")
    mdl = modeller[navn]
    print(f"{navn}: N = {dk(mdl._N)}, R² = {mdl._r2:.3f}")

# Klyngerobuste std.fejl (postnummer) for hovedmodellen som robusthedstjek
m4_klynge = pf.feols(formler["M4"], data=df, vcov={"CRV1": "postnr"})

# =============================================================================
# TABEL 5: REGRESSIONSTABEL M1–M4
# =============================================================================
overskrift("Tabel 5 – koefficienter (robuste std.fejl i parentes)")
kolonner = {}
for navn, mdl in modeller.items():
    td = mdl.tidy()
    stjerner = np.select([td["Pr(>|t|)"] < 0.01, td["Pr(>|t|)"] < 0.05, td["Pr(>|t|)"] < 0.1],
                         ["***", "**", "*"], "")
    kolonner[navn] = [f"{b:.4f}{s} ({se:.4f})" for b, se, s in
                      zip(td["Estimate"], td["Std. Error"], stjerner)]
    kolonner[navn] = pd.Series(kolonner[navn], index=td.index)
t5 = pd.DataFrame(kolonner)
nederst = pd.DataFrame({
    navn: [dk(m._N), f"{m._r2:.3f}", f"{m._adj_r2:.3f}",
           "Ja" if navn in ("M3", "M4") else "Nej", "Ja" if navn == "M4" else "Nej"]
    for navn, m in modeller.items()},
    index=["N", "R²", "Justeret R²", "Kvartals-FE", "Postnummer-FE"])
t5 = pd.concat([t5.fillna(""), nederst])
print(t5.to_string())
gem_tabel(t5, "tabel_5_regression.csv")

# =============================================================================
# TABEL 6: HOVEDMODELLEN FORTOLKET
# =============================================================================
overskrift("Tabel 6 – hovedmodellen (M4) fortolket")
median_pris = df["pris"].median()
td = modeller["M4"].tidy()
tk = m4_klynge.tidy()
t6 = pd.DataFrame({
    "beta": td["Estimate"],
    "SE_robust": td["Std. Error"],
    "SE_klynge_postnr": tk["Std. Error"],
    "KI_nedre": td["2.5%"],
    "KI_oevre": td["97.5%"],
})
log_vars = ["ln_boligareal", "ln_grundareal", "ln_grund_x_raekkehus"]
er_log = t6.index.isin(log_vars)
# Dummies og antal: %-ændring = 100·(e^β − 1)
t6["pct_effekt"] = np.where(er_log, np.nan, 100 * (np.exp(t6["beta"]) - 1))
t6["pct_KI_nedre"] = np.where(er_log, np.nan, 100 * (np.exp(t6["KI_nedre"]) - 1))
t6["pct_KI_oevre"] = np.where(er_log, np.nan, 100 * (np.exp(t6["KI_oevre"]) - 1))
t6["kr_ved_medianpris"] = np.where(er_log, np.nan, median_pris * (np.exp(t6["beta"]) - 1))
# Log-variable: elasticitet. Effekt af 10 % mere areal = 100·(1,1^β − 1)
t6["pct_ved_10pct_mere"] = np.where(er_log, 100 * (1.1 ** t6["beta"] - 1), np.nan)
print(f"Medianpris i stikprøven: {dk(median_pris)} kr.")
print(t6.round(4).to_string())
gem_tabel(t6.round(5), "tabel_6_effekter_M4.csv")

# =============================================================================
# HVAD SKER DER MED KOEFFICIENTERNE? (omitted variable bias i praksis)
# =============================================================================
overskrift("Sammenligning på tværs af modeller")
sml = ["ln_boligareal", "energimaerke::A", "energimaerke::G", "energimaerke::Intet",
       "opfoerelsesperiode::før 1900", "opfoerelsesperiode::2010+", "opvarmning::Olie"]
rows = {}
for navn, mdl in modeller.items():
    b = mdl.coef()
    rows[navn] = [b.get(v, np.nan) for v in sml]
print(pd.DataFrame(rows, index=sml).round(4).to_string())
print("\nRå forskel i median kr./m² mellem A og D:",
      f"{100 * (df.loc[df.energimaerke == 'A', 'kr_pr_m2'].median() / df.loc[df.energimaerke == 'D', 'kr_pr_m2'].median() - 1):.1f} %")

# =============================================================================
# TABEL 7: MULTIKOLLINEARITET (VIF)
# =============================================================================
overskrift("Tabel 7 – VIF for boligkarakteristika (tilfældig stikprøve på 50.000)")
stik = df.sample(min(50_000, len(df)), random_state=1)
X = pd.concat([
    stik[["ln_boligareal", "ln_grundareal", "ln_grund_x_raekkehus", "raekkehus", "vaerelser",
          "to_etager_plus", "ombygget", "har_garage", "har_carport", "har_udhus",
          "har_udestue"]],
    pd.get_dummies(stik[["badevaerelser_kat", "opfoerelsesperiode", "energimaerke",
                         "opvarmning", "ydervaeg", "tag"]], drop_first=False)
    .drop(columns=[f"{k}_{v}" for k, v in REF.items()]),
], axis=1).astype(float)
X.insert(0, "konstant", 1.0)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns, name="VIF").drop("konstant").sort_values(ascending=False)
print(vif.round(2).head(15).to_string())
print("\nTommelfingerregel: VIF over 10 tyder på problematisk multikollinearitet."
      "\nDummies for samme kategoriske variabel (fx opførelsesperioder) har naturligt lidt"
      " forhøjet VIF, især når referencegruppen er lille.")
gem_tabel(vif.round(3).to_frame(), "tabel_7_vif.csv")

# =============================================================================
# FIGUR 5: KOEFFICIENTPLOT (HOVEDMODEL)
# =============================================================================
overskrift("Figur 5")
grupper = [
    ("Energimærke (ref. D)", "energimaerke::", ["A", "B", "C", "E", "F", "G", "Intet"]),
    ("Opførelsesperiode (ref. 1960–69)", "opfoerelsesperiode::",
     ["før 1900", "1900–29", "1930–49", "1950–59", "1970–79", "1980–89", "1990–99",
      "2000–09", "2010+"]),
    ("Opvarmning (ref. fjernvarme)", "opvarmning::",
     ["Varmepumpe", "Gas", "Olie", "Elvarme", "Andet"]),
    ("Øvrige", "", ["ombygget", "har_garage", "har_carport", "har_udhus", "har_udestue",
                    "to_etager_plus", "badevaerelser_kat::2", "badevaerelser_kat::3+"]),
]
paen = {"ombygget": "Ombygget", "har_garage": "Garage", "har_carport": "Carport",
        "har_udhus": "Udhus", "har_udestue": "Udestue", "to_etager_plus": "2+ etager",
        "badevaerelser_kat::2": "2 badeværelser (ref. 1)",
        "badevaerelser_kat::3+": "3+ badeværelser (ref. 1)"}

etiketter, eff, lo, hi, y, gruppe_y = [], [], [], [], [], []
pos = 0
for titel, praefiks, niveauer in grupper:
    gruppe_y.append((pos, titel))
    pos += 1
    for niv in niveauer:
        navn = praefiks + niv
        if navn not in t6.index:
            continue
        etiketter.append(paen.get(navn, niv))
        eff.append(t6.loc[navn, "pct_effekt"])
        lo.append(t6.loc[navn, "pct_KI_nedre"])
        hi.append(t6.loc[navn, "pct_KI_oevre"])
        y.append(pos)
        pos += 1
    pos += 0.6

fig, ax = plt.subplots(figsize=(16 / 2.54, 0.17 * pos + 0.8))
ax.axvline(0, color=MUTED, lw=1)
eff, lo, hi = np.array(eff), np.array(lo), np.array(hi)
ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt="o", color=BLAA, ecolor=BLAA,
            markersize=4, elinewidth=1.2, capsize=2)
ax.set_yticks(y, etiketter)
for gy, titel in gruppe_y:
    ax.text(0.0, gy, titel, transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=INK)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_xlabel("Forskel i salgspris i forhold til referencekategorien (%), 95 %-konfidensinterval")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Hedonisk prismodel (M4): udvalgte karakteristika\n"
# "Kontrolleret for boligareal, grundareal, værelser, salgskvartal og postnummer",
# loc="left")
fig.savefig(UD / "fig_5_koefficientplot.png")
plt.close(fig)
print("  gemt: fig_5_koefficientplot.png")

print(f"\nAlle filer ligger i: {UD}")
