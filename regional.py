"""
regional.py – Regionale forskelle i prisdriverne (analyse 5.2)

Tager hovedmodellen (M4: alle boligkarakteristika + kvartals- og postnummer-dummies) og lader
tre effekter variere med region via interaktionsled:
  1) arealelasticiteten        ln(boligareal) × region
  2) energimærket              hver energimærke-dummy × region
  3) oliefyr                   olie-dummy × region
Alle øvrige karakteristika har samme effekt i hele landet. Regionernes hovedeffekt tages med,
men er stort set overflødig, fordi postnummer-dummies allerede fanger alt, der er fælles for et
område.

Output i figurer_og_tabeller:
  tabel_8_regionale_effekter.csv   effekter pr. region i % og kr. (ved regionens medianpris)
  tabel_9_ftest.csv                F-test: er effekterne ens i alle regioner?
  fig_6_regionale_forskelle.png    arealelasticitet, energimærke-profil og oliefyr pr. region

Kræver: pip3 install pyfixest scipy. faelles.py skal ligge i samme mappe.
Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Send outputtet i terminalen tilbage.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from faelles import GRID, INK2, MUTED, REF, REGION_RAEKKEFOELGE, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
REF_REGION = "Midtjylland"   # referenceregion i interaktionerne (påvirker ikke resultaterne pr. region)

# Samme farver og linjetyper pr. region som i figur 1
REGION_FARVE = {"Hovedstaden": "#2a78d6", "Sjælland": "#eb6834", "Syddanmark": "#1baf7a",
                "Midtjylland": "#eda100", "Nordjylland": "#e87ba4"}
REGION_STIL = {"Hovedstaden": ("-", "o"), "Sjælland": ("--", "s"), "Syddanmark": ("-.", "^"),
               "Midtjylland": (":", "D"), "Nordjylland": ((0, (5, 1, 1, 1)), "v")}
MAERKER = ["A", "B", "C", "E", "F", "G", "Intet"]   # D er reference


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


# =============================================================================
# DATA
# =============================================================================
df, bolig = forbered_data(MAPPE)

# Dummies, der skal interageres med region
for m in MAERKER:
    df[f"em_{m}"] = (df["energimaerke"] == m).astype(int)
df["olie"] = (df["opvarmning"] == "Olie").astype(int)

print("\nHandler pr. region og antal med oliefyr / energimærke G:")
print(df.groupby("region").agg(handler=("pris", "size"), olie=("olie", "sum"),
                               G=("em_G", "sum")).reindex(REGION_RAEKKEFOELGE).to_string())

# =============================================================================
# MODEL M5: HOVEDMODEL + INTERAKTIONER MED REGION
# =============================================================================
overskrift("Estimation")
# Hovedeffekten af region tages med, som man altid bør ved interaktioner. Når postnumrene
# ligger inden for én region, er den overflødig ved siden af postnummer-dummies og bliver
# automatisk udeladt (pyfixest skriver så en advarsel om kollinearitet – det er forventet).
interaktioner = (f" + i(region, ref='{REF_REGION}')"
                 f" + i(region, ln_boligareal, ref='{REF_REGION}')"
                 + "".join(f" + i(region, em_{m}, ref='{REF_REGION}')" for m in MAERKER)
                 + f" + i(region, olie, ref='{REF_REGION}')")
m4 = pf.feols(f"ln_pris ~ {bolig} | kvartal + postnr", data=df, vcov="hetero")
m5 = pf.feols(f"ln_pris ~ {bolig}{interaktioner} | kvartal + postnr", data=df, vcov="hetero")
print(f"M4: N = {dk(m4._N)}, R² = {m4._r2:.4f}")
print(f"M5: N = {dk(m5._N)}, R² = {m5._r2:.4f}")

navne = list(m5._coefnames)
beta = pd.Series(m5._beta_hat, index=navne)
V = pd.DataFrame(m5._vcov, index=navne, columns=navne)


def regional_effekt(hovednavn, variabel):
    """Effekten i hver region = hovedeffekt + interaktion (0 for referenceregionen).
    Std.fejl fra varians af en sum: Var(a + b) = Var(a) + Var(b) + 2·Cov(a, b)."""
    ud = {}
    for r in REGION_RAEKKEFOELGE:
        led = [hovednavn] if r == REF_REGION else [hovednavn, f"region::{r}:{variabel}"]
        b = beta[led].sum()
        se = np.sqrt(V.loc[led, led].values.sum())
        ud[r] = (b, se)
    return ud


# =============================================================================
# TABEL 8: EFFEKTER PR. REGION
# =============================================================================
overskrift("Tabel 8 – effekter pr. region (robuste std.fejl)")
median_pris = df.groupby("region")["pris"].median()
raekker = []

for r, (b, se) in regional_effekt("ln_boligareal", "ln_boligareal").items():
    raekker.append({"region": r, "variabel": "Arealelasticitet", "beta": b, "SE": se,
                    "KI_nedre": b - 1.96 * se, "KI_oevre": b + 1.96 * se,
                    "pct_effekt": 100 * (1.1 ** b - 1),   # 10 % mere boligareal
                    "kr_ved_regionens_medianpris": median_pris[r] * (1.1 ** b - 1)})

for m in MAERKER:
    for r, (b, se) in regional_effekt(f"energimaerke::{m}", f"em_{m}").items():
        raekker.append({"region": r, "variabel": f"Energimærke {m} (ref. D)", "beta": b, "SE": se,
                        "KI_nedre": b - 1.96 * se, "KI_oevre": b + 1.96 * se,
                        "pct_effekt": 100 * (np.exp(b) - 1),
                        "kr_ved_regionens_medianpris": median_pris[r] * (np.exp(b) - 1)})

for r, (b, se) in regional_effekt("opvarmning::Olie", "olie").items():
    raekker.append({"region": r, "variabel": "Oliefyr (ref. fjernvarme)", "beta": b, "SE": se,
                    "KI_nedre": b - 1.96 * se, "KI_oevre": b + 1.96 * se,
                    "pct_effekt": 100 * (np.exp(b) - 1),
                    "kr_ved_regionens_medianpris": median_pris[r] * (np.exp(b) - 1)})

t8 = pd.DataFrame(raekker)
t8["pct_KI_nedre"] = np.where(t8["variabel"] == "Arealelasticitet",
                              100 * (1.1 ** t8["KI_nedre"] - 1), 100 * (np.exp(t8["KI_nedre"]) - 1))
t8["pct_KI_oevre"] = np.where(t8["variabel"] == "Arealelasticitet",
                              100 * (1.1 ** t8["KI_oevre"] - 1), 100 * (np.exp(t8["KI_oevre"]) - 1))
t8["regionens_medianpris"] = t8["region"].map(median_pris)
t8 = t8.set_index(["variabel", "region"])
print("Arealelasticitet: pct_effekt = % højere pris ved 10 % mere boligareal")
vis = t8[["beta", "SE", "pct_effekt", "pct_KI_nedre", "pct_KI_oevre", "kr_ved_regionens_medianpris"]]
for var in ["Arealelasticitet", "Energimærke A (ref. D)", "Energimærke G (ref. D)",
            "Energimærke Intet (ref. D)", "Oliefyr (ref. fjernvarme)"]:
    print(f"\n{var}:")
    print(vis.loc[var].round(4).to_string())
gem_tabel(t8.round(5), "tabel_8_regionale_effekter.csv")

# =============================================================================
# TABEL 9: F-TEST AF ENS EFFEKTER (Wald-test med robuste std.fejl)
# =============================================================================
overskrift("Tabel 9 – F-test: er effekterne ens i alle regioner?")


def wald(led):
    """H0: alle interaktionsled = 0, dvs. effekten er den samme i alle regioner.
    W = b' V⁻¹ b,  F = W / q  med q restriktioner."""
    b = beta[led].values
    v = V.loc[led, led].values
    W = float(b @ np.linalg.solve(v, b))
    q = len(led)
    F = W / q
    p = stats.f.sf(F, q, m5._N - len(navne))
    return q, F, p


andre = [r for r in REGION_RAEKKEFOELGE if r != REF_REGION]
test = {
    "Arealelasticitet ens i alle regioner": [f"region::{r}:ln_boligareal" for r in andre],
    "Energimærke-effekter ens i alle regioner": [f"region::{r}:em_{m}" for m in MAERKER for r in andre],
    "Olie-effekt ens i alle regioner": [f"region::{r}:olie" for r in andre],
}
t9 = pd.DataFrame([(navn, *wald(led)) for navn, led in test.items()],
                  columns=["Hypotese", "Antal restriktioner (q)", "F", "p-værdi"]).set_index("Hypotese")
print(t9.to_string(float_format=lambda x: f"{x:.4g}"))
print("\nLav p-værdi (< 0,05) = effekten er statistisk forskellig mellem regionerne."
      "\nMed så mange observationer bliver selv små forskelle signifikante – se også størrelsen.")
gem_tabel(t9, "tabel_9_ftest.csv")

# =============================================================================
# FIGUR 6: REGIONALE FORSKELLE
# =============================================================================
overskrift("Figur 6")
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16 / 2.54, 17 / 2.54),
                                 gridspec_kw={"height_ratios": [1, 1.6, 1]})
y = np.arange(len(REGION_RAEKKEFOELGE))


def prik_panel(ax, var, titel, xlabel):
    d = t8.loc[var].reindex(REGION_RAEKKEFOELGE)
    for i, r in enumerate(REGION_RAEKKEFOELGE):
        ax.errorbar(d.loc[r, "pct_effekt"], i,
                    xerr=[[d.loc[r, "pct_effekt"] - d.loc[r, "pct_KI_nedre"]],
                          [d.loc[r, "pct_KI_oevre"] - d.loc[r, "pct_effekt"]]],
                    fmt=REGION_STIL[r][1], color=REGION_FARVE[r], markersize=5, capsize=2,
                    elinewidth=1.2)
    ax.set_yticks(y, REGION_RAEKKEFOELGE)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    # Trin på 1, 2 eller 5 (× 10^k), så akseværdierne kan vises præcist med én decimal
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x, 1)} %"))
    ax.set_title(titel, loc="left")
    ax.set_xlabel(xlabel)


prik_panel(a1, "Arealelasticitet", "a) Prisstigning ved 10 % mere boligareal",
           "Højere salgspris (%), 95 %-konfidensinterval")

x = np.arange(len(["A", "B", "C", "D", "E", "F", "G"]))
for r in REGION_RAEKKEFOELGE:
    vaerdier = [t8.loc[(f"Energimærke {m} (ref. D)", r), "pct_effekt"] if m != "D" else 0.0
                for m in ["A", "B", "C", "D", "E", "F", "G"]]
    ls, mk = REGION_STIL[r]
    a2.plot(x, vaerdier, ls=ls, marker=mk, color=REGION_FARVE[r], lw=1.8, markersize=4,
            label=r)
a2.axhline(0, color=MUTED, lw=1)
a2.set_xticks(x, ["A", "B", "C", "D\n(ref.)", "E", "F", "G"])
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{dk(v)} %"))
a2.set_ylabel("Prisforskel i forhold til D")
a2.set_title("b) Energimærke: prisforskel i forhold til mærke D", loc="left")
a2.legend(ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
           handlelength=2.5, columnspacing=1.0)

prik_panel(a3, "Oliefyr (ref. fjernvarme)", "c) Oliefyr i forhold til fjernvarme",
           "Prisforskel (%), 95 %-konfidensinterval")

# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# fig.suptitle("Regionale forskelle i udvalgte prisdrivere (hovedmodel med interaktioner)",
# x=0.0, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(UD / "fig_6_regionale_forskelle.png")
plt.close(fig)
print("  gemt: fig_6_regionale_forskelle.png")
print(f"\nAlle filer ligger i: {UD}")
