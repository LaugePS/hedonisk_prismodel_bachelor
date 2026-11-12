"""
vurderingsgab.py – Hvilke huse sælges over/under den offentlige vurdering? (analyse 5.9)

Prædiktionen (5.6) viste, at vurderingen i median ligger ca. 4 % under salgsprisen, men at
spredningen er stor. Her undersøges, om afvigelserne er SYSTEMATISKE, dvs. om bestemte typer
huse (energimærke, opvarmning, alder, størrelse, region) sælges over eller under vurderingen.

Stikprøve
  Salg i 2020 med 2020-vurdering og salg i 2022 med 2022-vurdering (vurderingsår = salgsår),
  ejendomsværdi > 0. Samme sammenlignelige handler som i praediktion.py (her alle, ikke kun testdata).

Afhængige variable
  ln_gab = ln(salgspris) − ln(vurdering) = ln(salgspris / vurdering)     (kontinuert)
  over   = 1, hvis salgspris > vurdering, ellers 0                       (binær)

Del A – Dekomposition (tabel 14, figur 11)
  Den samme hedoniske model estimeres to gange på de samme handler: med ln(salgspris) og med
  ln(vurdering) som afhængig variabel. Koefficienterne er markedets og vurderingens "implicitte
  priser" på hver egenskab. Forskellen er præcis koefficienten i en regression af ln_gab
  (OLS er lineær i y). En forskel ≠ 0 betyder, at vurderingen prissætter egenskaben anderledes
  end markedet.
Del B – Sandsynlighed for salg over vurderingen (tabel 15, figur 12) – Békés & Kézdi kap. 11
  LPM 1:  over ~ boligkarakteristika + region + kvartal              (OLS på 0/1-variabel)
  Logit:  samme variable, gennemsnitlige marginale effekter (AME)
  LPM 2:  over ~ boligkarakteristika | kvartal + postnummer          (inden for postnummer)
  Standardfejl klynget på postnummer (vurderingsfejl ligner hinanden i samme område).
Del C – Deskriptiv (tabel 16, figur 10): andel over vurderingen og median gab pr. gruppe.

Output i figurer_og_tabeller:
  tabel_14_dekomposition.csv, tabel_15_lpm_logit.csv, tabel_16_vurderingsgab_grupper.csv
  fig_10_vurderingsgab_fordeling.png, fig_11_marked_vs_vurdering.png, fig_12_lpm_vs_logit.png

Kræver: pip3 install pyfixest statsmodels. faelles.py skal ligge i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from faelles import BLAA, INK, MUTED, REF, REGION_RAEKKEFOELGE, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
AAR = [2020, 2022]           # år hvor vurderingsår = salgsår
REF_REGION = "Midtjylland"   # samme reference som i regional.py
ORANGE = "#eb6834"
KLYNGE = {"CRV1": "postnr"}

UDVALGTE = {
    "ln_boligareal": "Boligareal (log)",
    "ln_grundareal": "Grundareal (log)",
    "raekkehus": "Række/kæde/dobbelthus",
    "energimaerke::A": "Energimærke A (ref. D)",
    "energimaerke::B": "Energimærke B",
    "energimaerke::C": "Energimærke C",
    "energimaerke::E": "Energimærke E",
    "energimaerke::F": "Energimærke F",
    "energimaerke::G": "Energimærke G",
    "energimaerke::Intet": "Intet energimærke",
    "opvarmning::Olie": "Oliefyr (ref. fjernvarme)",
    "opvarmning::Gas": "Gas",
    "opvarmning::Varmepumpe": "Varmepumpe",
    "opvarmning::Elvarme": "Elvarme",
    "opfoerelsesperiode::før 1900": "Opført før 1900 (ref. 1960–69)",
    "opfoerelsesperiode::2010+": "Opført 2010+",
    "ombygget": "Ombygget",
    "badevaerelser_kat::2": "2 badeværelser (ref. 1)",
    "har_garage": "Garage",
    "tag::Stråtag": "Stråtag (ref. tegl)",
    "tag::Fibercement": "Fibercementtag",
}
LOG_VARS = ["ln_boligareal", "ln_grundareal"]


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def pct(b):
    """Log-koefficient -> %-effekt: 100·(e^β − 1)."""
    return 100 * (np.exp(b) - 1)


def til_statsmodels(formel):
    """Oversætter pyfixest's i(x, ref='r') til statsmodels' C(x, Treatment(reference='r'))."""
    return re.sub(r"i\((\w+), ref='([^']*)'\)", r"C(\1, Treatment(reference='\2'))", formel)


def sm_navn(var):
    """pyfixest-navn (energimaerke::G) -> statsmodels-navn (C(energimaerke, ...)[T.G])."""
    if "::" not in var:
        return var
    col, niv = var.split("::")
    ref = REF_REGION if col == "region" else REF[col]
    return f"C({col}, Treatment(reference='{ref}'))[T.{niv}]"


# =============================================================================
# DATA
# =============================================================================
df, bolig = forbered_data(MAPPE)
d = df[df["salgsaar"].isin(AAR) & (df["vurd_matcher_salgsaar"] == 1)
       & (df["ejendomvaerdibeloeb"] > 0)].copy()
d["ln_vurdering"] = np.log(d["ejendomvaerdibeloeb"])
d["ln_gab"] = d["ln_pris"] - d["ln_vurdering"]
d["over"] = (d["pris"] > d["ejendomvaerdibeloeb"]).astype(int)

# Meget små kategorier fjernes (fx opvarmning "Ukendt" med 2 handler). I logit giver en kategori,
# hvor alle har samme udfald, "perfekt separation": koefficienten går mod uendelig, og
# estimationen bryder sammen. Fjernes i alle modeller, så stikprøven er den samme overalt.
MIN_KATEGORI = 30
foer = len(d)
for col in REF:
    antal = d[col].map(d[col].value_counts())
    lille = (antal < MIN_KATEGORI) & (d[col] != REF[col])   # referencen fjernes aldrig
    for niv in d.loc[lille, col].unique():
        print(f"Fjerner {col} = '{niv}' ({(d[col] == niv).sum()} handler – under {MIN_KATEGORI})")
    d = d[~lille]
print(f"Handler fjernet pga. meget små kategorier: {foer - len(d)}")

overskrift("Stikprøve")
print(f"Sammenlignelige handler: {dk(len(d))}")
print(d.groupby("salgsaar").agg(Handler=("over", "size"), Andel_over=("over", "mean"),
                                Median_gab=("vurderingsgab", "median")).round(3).to_string())
print(f"\nFordeling af salgspris/vurdering (percentiler):")
print(d["vurderingsgab"].quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
      .round(3).to_string())

# =============================================================================
# DEL C (vises først): ANDEL OVER VURDERINGEN PR. GRUPPE
# =============================================================================
overskrift("Tabel 16 – andel solgt over vurderingen og median salgspris/vurdering pr. gruppe")
grupper = {"Salgsår": "salgsaar", "Region": "region", "Boligtype": "boligtype",
           "Energimærke": "energimaerke", "Opvarmning": "opvarmning",
           "Opførelsesperiode": "opfoerelsesperiode"}
raekker = []
for titel, col in grupper.items():
    g = d.groupby(col).agg(Handler=("over", "size"), Andel_over=("over", "mean"),
                           Median_gab=("vurderingsgab", "median"))
    g.index = pd.MultiIndex.from_product([[titel], g.index.astype(str)])
    raekker.append(g)
t16 = pd.concat(raekker)
vis = t16.copy().astype(object)
vis["Handler"] = [dk(x) for x in t16["Handler"]]
vis["Andel_over"] = [f"{dk(100 * x, 1)} %" for x in t16["Andel_over"]]
vis["Median_gab"] = [dk(x, 3) for x in t16["Median_gab"]]
print(vis.to_string())
gem_tabel(t16.round(4), "tabel_16_vurderingsgab_grupper.csv")

# =============================================================================
# DEL A: DEKOMPOSITION – MARKEDETS VS. VURDERINGENS IMPLICITTE PRISER
# =============================================================================
overskrift("Del A – hedonisk model for salgspris og for vurdering (samme handler)")
FE = "kvartal + postnr"
m_pris = pf.feols(f"ln_pris ~ {bolig} | {FE}", data=d, vcov=KLYNGE)
m_vurd = pf.feols(f"ln_vurdering ~ {bolig} | {FE}", data=d, vcov=KLYNGE)
m_gab = pf.feols(f"ln_gab ~ {bolig} | {FE}", data=d, vcov=KLYNGE)
for navn, m in [("ln(salgspris)", m_pris), ("ln(vurdering)", m_vurd), ("ln(pris/vurdering)", m_gab)]:
    print(f"{navn:20s} N = {dk(m._N):>7}   R² = {m._r2:.3f}   R² inden for postnr = {m._r2_within:.3f}")

tp, tv, tg = m_pris.tidy(), m_vurd.tidy(), m_gab.tidy()
# Kontrol: forskellen mellem de to modeller er præcis gab-modellens koefficienter
afvig = (tp["Estimate"] - tv["Estimate"] - tg["Estimate"]).abs().max()
print(f"Kontrol: største afvigelse mellem β(pris) − β(vurdering) og β(gab): {afvig:.2e}")

overskrift("Tabel 14 – dekomposition (elasticiteter for log-variable, ellers %)")
rk = {}
for var, label in UDVALGTE.items():
    if var not in tp.index:
        continue
    bp, bv, bg = tp.loc[var, "Estimate"], tv.loc[var, "Estimate"], tg.loc[var, "Estimate"]
    f = (lambda x: x) if var in LOG_VARS else pct
    rk[label] = {"Marked (salgspris)": f(bp), "Vurdering": f(bv),
                 "Gab: pris/vurdering": f(bg), "Std.fejl gab (log)": tg.loc[var, "Std. Error"],
                 "p-værdi gab": tg.loc[var, "Pr(>|t|)"]}
t14 = pd.DataFrame(rk).T
vis = t14.copy().astype(object)
for label in t14.index:
    log = "(log)" in label
    for c in ["Marked (salgspris)", "Vurdering", "Gab: pris/vurdering"]:
        vis.loc[label, c] = dk(t14.loc[label, c], 3) if log else f"{dk(t14.loc[label, c], 1)} %"
    vis.loc[label, "Std.fejl gab (log)"] = dk(t14.loc[label, "Std.fejl gab (log)"], 4)
    vis.loc[label, "p-værdi gab"] = dk(t14.loc[label, "p-værdi gab"], 3)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(vis.to_string())
print("\nLæsning: 'Gab' > 0 betyder, at huse med egenskaben sælges relativt højere i forhold til "
      "vurderingen end referencehuse i samme postnummer og kvartal.")
gem_tabel(t14.round(5), "tabel_14_dekomposition.csv")

# =============================================================================
# DEL B: LPM OG LOGIT FOR SALG OVER VURDERINGEN
# =============================================================================
overskrift("Del B – sandsynlighed for salg over vurderingen")
region_led = f"i(region, ref='{REF_REGION}')"
lpm1 = pf.feols(f"over ~ {bolig} + {region_led} | kvartal", data=d, vcov=KLYNGE)
lpm2 = pf.feols(f"over ~ {bolig} | {FE}", data=d, vcov=KLYNGE)

# Logit med statsmodels (samme variable som LPM 1; kvartal som dummies)
d_l = d.copy()
d_l["klynge"] = pd.factorize(d_l["postnr"])[0]
logit_formel = til_statsmodels(f"over ~ {bolig} + {region_led}") + " + C(kvartal)"
logit = smf.logit(logit_formel, data=d_l).fit(
    disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": d_l["klynge"]})
ame = logit.get_margeff(at="overall", method="dydx", dummy=True).summary_frame()

# Modelmål (Békés & Kézdi kap. 11)
p_lpm = lpm1.predict()
p_logit = logit.predict()
y = d["over"].to_numpy()
print(f"Andel over vurderingen (y = 1): {y.mean():.3f}")
print(f"LPM 1  N = {dk(lpm1._N)}   R² = {lpm1._r2:.3f}   "
      f"Brier = {np.mean((y - p_lpm) ** 2):.4f}   "
      f"forudsagt sandsynlighed < 0 eller > 1: {100 * np.mean((p_lpm < 0) | (p_lpm > 1)):.2f} %")
print(f"Logit  N = {dk(int(logit.nobs))}   McFadden pseudo-R² = {logit.prsquared:.3f}   "
      f"Brier = {np.mean((y - p_logit) ** 2):.4f}")
print(f"LPM 2  N = {dk(lpm2._N)}   R² = {lpm2._r2:.3f}   (postnummer som fast effekt)")
print(f"Forudsagte sandsynligheder: LPM 1 fra {p_lpm.min():.2f} til {p_lpm.max():.2f}, "
      f"logit fra {p_logit.min():.2f} til {p_logit.max():.2f}")
print(f"Korrelation mellem LPM- og logit-sandsynligheder: {np.corrcoef(p_lpm, p_logit)[0, 1]:.3f}")

overskrift("Tabel 15 – effekt på sandsynligheden for salg over vurderingen (procentpoint)")
alle_vars = dict(UDVALGTE)
for r in REGION_RAEKKEFOELGE:
    if r != REF_REGION:
        alle_vars[f"region::{r}"] = f"Region {r} (ref. {REF_REGION})"
t1, t2 = lpm1.tidy(), lpm2.tidy()
rk = {}
for var, label in alle_vars.items():
    smv = sm_navn(var)
    rk[label] = {
        "LPM 1": 100 * t1.loc[var, "Estimate"] if var in t1.index else np.nan,
        "LPM 1 std.fejl": 100 * t1.loc[var, "Std. Error"] if var in t1.index else np.nan,
        "Logit AME": 100 * ame.loc[smv, "dy/dx"] if smv in ame.index else np.nan,
        "Logit std.fejl": 100 * ame.loc[smv, "Std. Err."] if smv in ame.index else np.nan,
        "LPM 2 (postnr-FE)": 100 * t2.loc[var, "Estimate"] if var in t2.index else np.nan,
        "LPM 2 std.fejl": 100 * t2.loc[var, "Std. Error"] if var in t2.index else np.nan,
    }
t15 = pd.DataFrame(rk).T
print(t15.round(1).to_string(na_rep="–"))
print("\nLog-variable: effekt af at gå 1 log-enhed op (ca. +172 % areal). "
      "Divider med 10 for effekten af ca. +10 % areal (0,1 log-enhed).")
gem_tabel(t15.round(3), "tabel_15_lpm_logit.csv")

# =============================================================================
# FIGUR 10: FORDELING AF SALGSPRIS/VURDERING OG ANDEL OVER PR. ENERGIMÆRKE
# =============================================================================
overskrift("Figurer")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 / 2.54, 7.5 / 2.54),
                             gridspec_kw={"width_ratios": [1.3, 1]})
bins = np.linspace(0.4, 2.0, 65)
for aar, farve in zip(AAR, [BLAA, ORANGE]):
    x = d.loc[d["salgsaar"] == aar, "vurderingsgab"]
    a1.hist(x, bins=bins, density=True, histtype="step", lw=1.4, color=farve)
a1.axvline(1, color=INK, lw=0.8, ls="--")
a1.set_xlabel("Salgspris / offentlig vurdering")
a1.set_yticks([])
a1.set_ylim(0, a1.get_ylim()[1] * 1.3)   # plads til forklaringen over fordelingen
handles = [Line2D([], [], color=f, lw=1.4) for f in [BLAA, ORANGE]]
a1.legend(handles, [f"Salg i {aar} (median "
                    f"{dk(d.loc[d['salgsaar'] == aar, 'vurderingsgab'].median(), 2)})"
                    for aar in AAR], frameon=False, fontsize=7, loc="upper right")
a1.set_title("a) Fordeling af salgspris / vurdering\n(vist mellem 0,4 og 2,0)", loc="left",
             fontsize=8.5)
a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x, 1)))

orden = ["A", "B", "C", "D", "E", "F", "G", "Intet"]
andel = d.groupby("energimaerke")["over"].mean().reindex(orden) * 100
a2.bar(range(len(orden)), andel, color=[INK if k == "D" else BLAA for k in orden], width=0.7)
a2.axhline(100 * d["over"].mean(), color=MUTED, lw=0.8, ls="--")
a2.set_xticks(range(len(orden)), orden)
a2.set_ylim(0, 100)
a2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f} %"))
a2.set_title("b) Andel solgt over vurderingen\npr. energimærke (stiplet = alle)", loc="left",
             fontsize=8.5)
a2.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(UD / "fig_10_vurderingsgab_fordeling.png")
plt.close(fig)
print("  gemt: fig_10_vurderingsgab_fordeling.png")

# =============================================================================
# FIGUR 11: MARKEDETS VS. VURDERINGENS PRISSÆTNING (DUMMIES, %)
# =============================================================================
dummies = [v for v in UDVALGTE if v not in LOG_VARS and v in tp.index]
labels = [UDVALGTE[v] for v in dummies]
fig, ax = plt.subplots(figsize=(16 / 2.54, 11 / 2.54))
yy = np.arange(len(dummies))
for t, farve, navn, dy in [(tp, INK, "Markedet (salgspris)", -0.17),
                            (tv, ORANGE, "Offentlig vurdering", 0.17)]:
    b = t.loc[dummies, "Estimate"]
    lo, hi = t.loc[dummies, "2.5%"], t.loc[dummies, "97.5%"]
    ax.errorbar(pct(b), yy + dy, xerr=[pct(b) - pct(lo), pct(hi) - pct(b)], fmt="o",
                color=farve, markersize=3.5, capsize=1.5, elinewidth=1, label=navn)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(yy, labels, fontsize=7.5)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Markedets og vurderingens prissætning af boligens egenskaber\n"
# "Samme handler og model, postnummer og kvartal holdt fast (95 %-KI)",
# loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_11_marked_vs_vurdering.png")
plt.close(fig)
print("  gemt: fig_11_marked_vs_vurdering.png")

# =============================================================================
# FIGUR 12: LPM VS. LOGIT (MARGINALE EFFEKTER, PROCENTPOINT)
# =============================================================================
vis_vars = [l for l in t15.index if "(log)" not in l and pd.notna(t15.loc[l, "LPM 1"])]
fig, ax = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))
yy = np.arange(len(vis_vars))
for kol, se_kol, farve, navn, dy in [("LPM 1", "LPM 1 std.fejl", INK, "LPM", -0.17),
                                     ("Logit AME", "Logit std.fejl", BLAA, "Logit (AME)", 0.17)]:
    b, s = t15.loc[vis_vars, kol], t15.loc[vis_vars, se_kol]
    ax.errorbar(b, yy + dy, xerr=1.96 * s, fmt="o", color=farve, markersize=3.5, capsize=1.5,
                elinewidth=1, label=navn)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(yy, vis_vars, fontsize=7.5)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} pp"))
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Effekt på sandsynligheden for salg over den offentlige vurdering\n"
# "Procentpoint, 95 %-KI (std.fejl klynget på postnummer)", loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_12_lpm_vs_logit.png")
plt.close(fig)
print("  gemt: fig_12_lpm_vs_logit.png")
print(f"\nAlle filer ligger i: {UD}")
