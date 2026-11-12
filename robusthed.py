"""
robusthed.py – Robusthedstjek af hovedmodellen (M4)

Estimerer hovedmodellen igen med ændrede valg og viser, om de vigtigste resultater holder:

  R0  Hovedmodel (M4), 2019+                         = udgangspunktet
  R1a Hovedmodel, kun 2020+                          (samme stikprøve som R1b)
  R1b Kontant købesum i stedet for samlet, 2020+     (prisvariablen)
  R2  Hele perioden 2015+                            (hovedperioden)
  R3a Kun fritliggende huse                          (separate modeller pr. boligtype)
  R3b Kun række-/kæde-/dobbelthuse
  R4  Uden grænser for ekstreme huse                 (rensningstrin 10a–10c)
  R5  Kvartals-dummies pr. region                    (prisudviklingen må være forskellig
                                                      i hver region)
  R6  Uden opvarmning, ydervæg og tag                (hvor meget af energimærkets effekt
                                                      fanges af husets øvrige tekniske stand?)
Desuden: standardfejl for hovedmodellen beregnet på tre måder (robuste, klynget på
postnummer, klynget på ejendom – sidstnævnte fordi samme hus kan være solgt flere gange).

Output i figurer_og_tabeller:
  tabel_12_robusthed.csv        udvalgte effekter i alle specifikationer
  tabel_13_standardfejl.csv     hovedmodellens std.fejl på tre måder
  fig_9_robusthed.png           arealelasticitet, energimærke G og oliefyr på tværs af specifikationer

Kræver: data_renset.csv OG data_renset_inkl_ekstreme.csv (kør den nyeste rensning.py først).
faelles.py skal ligge i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter, MaxNLocator

from faelles import BLAA, INK, MUTED, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)

UDVALGTE = {
    "ln_boligareal": "Boligareal (elasticitet)",
    "ln_grundareal": "Grundareal (elasticitet)",
    "energimaerke::A": "Energimærke A (ref. D)",
    "energimaerke::C": "Energimærke C",
    "energimaerke::E": "Energimærke E",
    "energimaerke::G": "Energimærke G",
    "energimaerke::Intet": "Intet energimærke",
    "opvarmning::Olie": "Oliefyr (ref. fjernvarme)",
    "opvarmning::Gas": "Gas",
    "opvarmning::Varmepumpe": "Varmepumpe",
    "opfoerelsesperiode::før 1900": "Opført før 1900 (ref. 1960–69)",
    "opfoerelsesperiode::2010+": "Opført 2010+",
    "badevaerelser_kat::2": "2 badeværelser (ref. 1)",
    "har_garage": "Garage",
    "tag::Fibercement": "Fibercementtag (ref. tegl)",
}
LOG_VARS = ["ln_boligareal", "ln_grundareal"]


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def kun_boligtype(formel):
    """Fjerner række-dummyen og interaktionen, når der kun er én boligtype i data."""
    return formel.replace(" + ln_grund_x_raekkehus + raekkehus", "")


def uden_teknik(formel):
    """Fjerner opvarmning, ydervæg og tag fra modellen."""
    for col in ["opvarmning", "ydervaeg", "tag"]:
        start = formel.index(f" + i({col},")
        slut = formel.index(")", start) + 1
        formel = formel[:start] + formel[slut:]
    return formel


# =============================================================================
# DATA
# =============================================================================
df, bolig = forbered_data(MAPPE)                                      # 2019+
df15, _ = forbered_data(MAPPE, fra_aar=2015)                          # hele perioden
dfx, _ = forbered_data(MAPPE, fil="data_renset_inkl_ekstreme.csv")    # uden grænser

df["kvartal_region"] = df["kvartal"] + "_" + df["region"]

df20 = df[df["salgsaar"] >= 2020].copy()
df20k = df20[df20["kontantkoebesum"] > 0].copy()
df20k["ln_kontant"] = np.log(df20k["kontantkoebesum"])
df20 = df20.loc[df20k.index]   # præcis samme handler i R1a og R1b

FE = "kvartal + postnr"
specs = {
    "R0 Hovedmodel 2019+": (f"ln_pris ~ {bolig} | {FE}", df),
    "R1a Hovedmodel 2020+": (f"ln_pris ~ {bolig} | {FE}", df20),
    "R1b Kontant købesum 2020+": (f"ln_kontant ~ {bolig} | {FE}", df20k),
    "R2 Hele perioden 2015+": (f"ln_pris ~ {bolig} | {FE}", df15),
    "R3a Kun fritliggende": (f"ln_pris ~ {kun_boligtype(bolig)} | {FE}",
                             df[df["raekkehus"] == 0]),
    "R3b Kun række/kæde/dobbelt": (f"ln_pris ~ {kun_boligtype(bolig)} | {FE}",
                                   df[df["raekkehus"] == 1]),
    "R4 Inkl. ekstreme huse": (f"ln_pris ~ {bolig} | {FE}", dfx),
    # Kvartal × region som almindelige dummies (155 stk.) og postnummer som fast effekt.
    # Matematisk det samme som "kvartal^region + postnr", men konvergerer hurtigt.
    "R5 Kvartal pr. region": (f"ln_pris ~ {bolig} + i(kvartal_region) | postnr", df),
    "R6 Uden opvarmning/ydervæg/tag": (f"ln_pris ~ {uden_teknik(bolig)} | {FE}", df),
}

# =============================================================================
# ESTIMATION
# =============================================================================
overskrift("Estimation")
fits = {}
for navn, (formel, data) in specs.items():
    fits[navn] = pf.feols(formel, data=data, vcov="hetero")
    print(f"{navn:35s} N = {dk(fits[navn]._N):>8}   R² = {fits[navn]._r2:.3f}")

# =============================================================================
# TABEL 12: UDVALGTE EFFEKTER PÅ TVÆRS AF SPECIFIKATIONER
# =============================================================================
overskrift("Tabel 12 – udvalgte effekter (elasticiteter for areal, ellers % i forhold til ref.)")
raekker = {}
for navn, f in fits.items():
    td = f.tidy()
    kol = {}
    for var, label in UDVALGTE.items():
        if var not in td.index:
            kol[label] = np.nan
            continue
        b = td.loc[var, "Estimate"]
        kol[label] = b if var in LOG_VARS else 100 * (np.exp(b) - 1)
    kol["N"] = f._N
    kol["R²"] = f._r2
    raekker[navn] = kol
t12 = pd.DataFrame(raekker)
vis = t12.copy().astype(object)
for label in t12.index:
    if label == "N":
        vis.loc[label] = [dk(x) for x in t12.loc[label]]
    elif label == "R²" or "elasticitet" in label:
        vis.loc[label] = [f"{x:.3f}" if pd.notna(x) else "–" for x in t12.loc[label]]
    else:
        vis.loc[label] = [f"{dk(x, 1)} %" if pd.notna(x) else "–" for x in t12.loc[label]]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(vis.to_string())
gem_tabel(t12.round(4), "tabel_12_robusthed.csv")

# =============================================================================
# TABEL 13: STANDARDFEJL PÅ TRE MÅDER
# =============================================================================
overskrift("Tabel 13 – hovedmodellens standardfejl på tre måder")
formel0 = specs["R0 Hovedmodel 2019+"][0]
se = {"Robust (heteroskedasticitet)": fits["R0 Hovedmodel 2019+"].tidy()["Std. Error"]}
for navn, klynge in [("Klynget på postnummer", "postnr"),
                     ("Klynget på ejendom (BFE)", "bestemtfastejendombfenr")]:
    se[navn] = pf.feols(formel0, data=df, vcov={"CRV1": klynge}).tidy()["Std. Error"]
t13 = pd.DataFrame(se).loc[list(UDVALGTE)].rename(index=UDVALGTE)
t13.insert(0, "beta", fits["R0 Hovedmodel 2019+"].tidy()["Estimate"].loc[list(UDVALGTE)].values)
t13["Laveste |t| (største std.fejl)"] = (t13["beta"].abs() / t13[list(se)].max(axis=1))
print(t13.round(4).to_string())
print(f"\nAntal ejendomme solgt mere end én gang i 2019+: "
      f"{dk((df['bestemtfastejendombfenr'].value_counts() > 1).sum())}")
gem_tabel(t13.round(5), "tabel_13_standardfejl.csv")

# =============================================================================
# FIGUR 9: ROBUSTHED AF TRE CENTRALE RESULTATER
# =============================================================================
overskrift("Figur 9")
paneler = [("ln_boligareal", "a) Boligareal\n(elasticitet)", False),
           ("energimaerke::G", "b) Energimærke G\nvs. D (%)", True),
           ("opvarmning::Olie", "c) Oliefyr vs.\nfjernvarme (%)", True)]
navne = list(fits)
fig, akser = plt.subplots(1, 3, figsize=(16 / 2.54, 9 / 2.54), sharey=True)
for ax, (var, titel, pct) in zip(akser, paneler):
    for i, navn in enumerate(navne):
        td = fits[navn].tidy()
        if var not in td.index:
            continue
        b, lo, hi = td.loc[var, ["Estimate", "2.5%", "97.5%"]]
        if pct:
            b, lo, hi = (100 * (np.exp(x) - 1) for x in (b, lo, hi))
        farve = INK if i == 0 else BLAA
        ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="o", color=farve, markersize=4,
                    capsize=2, elinewidth=1.2)
    ref = fits[navne[0]].tidy().loc[var, "Estimate"]
    ax.axvline(100 * (np.exp(ref) - 1) if pct else ref, color=MUTED, lw=0.8, ls="--")
    ax.set_title(titel, loc="left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    if pct:
        # Panelets titel angiver "(%)", så %-tegnet udelades i akseværdierne (ellers for tæt)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x)))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x, 2)))
akser[0].set_yticks(range(len(navne)), navne, fontsize=7.5)
akser[0].invert_yaxis()
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# fig.suptitle("Robusthed: centrale resultater på tværs af specifikationer (95 %-KI)\n"
# "Stiplet linje = hovedmodellen", x=0.0, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(UD / "fig_9_robusthed.png")
plt.close(fig)
print("  gemt: fig_9_robusthed.png")
print(f"\nAlle filer ligger i: {UD}")
