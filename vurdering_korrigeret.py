"""
vurdering_korrigeret.py – Kan den offentlige vurdering forbedres med registerdata? (analyse 5.15)

Vurderingsgabet (5.10) viste, at vurderingen næsten ikke prissætter energimærket, mens markedet
gør det kraftigt. Her testes, om vurderingen bliver mere præcis, når den korrigeres for de
systematiske afvigelser – og om en kombination af vurderingen og prædiktionsmodellen er bedre end
hver for sig. Det er et ANDET spørgsmål end i 5.12–5.14 (dér måtte vurderingen aldrig bruges
som input). Her er vurderingen udgangspunktet, og registerdata bruges til at forbedre den.

Data
  Samme sammenlignelige handler i 2020 og 2022 som i praediktion_tidstro.py:
    Kalibreringshandler (ikke test): bruges til at ESTIMERE korrektionerne
    Testhandler:                     bruges kun til at EVALUERE dem
  Forudsigelserne fra den tidstro trin 5-model (gradient boosting + vej, tidligere salg,
  koordinater, kystafstand og naboernes prisniveau) læses fra forudsigelser_geo.csv.
  (Opdateret 2/11-2026: før brugtes trin 3-modellen fra forudsigelser_tidstro.csv. De gamle
  resultater ligger i figurer_og_tabeller/gammel_trin3.)

Korrektioner (alle estimeret på kalibreringshandlerne, evalueret på testhandlerne)
  V0  Rå vurdering
  V1  Niveaukorrigeret pr. kvartal                       (= variant c i 5.14)
  V2  + energimærke (OLS):  ln(P/V) ~ energimærke | kvartal
  V2m + energimærke (median): medianforholdet salgspris/V1 pr. energimærke
  V3  + alle boligkarakteristika og postnummer:  ln(P/V) ~ boligkarakteristika | kvartal + postnr
  K1  Kombination:  ln P ~ ln V + ln P̂_model | kvartal   (vægtene estimeres på kalibrering)
  K2  Kombination + boligkarakteristika og postnummer
  Til sammenligning: trin 5-modellen (tidstro) alene, niveaukorrigeret pr. kvartal.

Tidstro robusthed (kun salg i 2022): energimærke-korrektionen estimeres på salg i 2020
(tidligere vurderingsår) og anvendes på 2022-vurderingen.

Usikkerhed: bootstrap (1.000) af forskellen i MdAPE i forhold til V1.

Output i figurer_og_tabeller:
  tabel_23_korrigeret_vurdering.csv, tabel_24_korrigeret_energimaerke.csv,
  tabel_25_korrigeret_bootstrap.csv, fig_16_korrigeret_vurdering.png

Kræver: forudsigelser_geo.csv (kør praediktion_geo.py først) og
data_renset.csv. faelles.py i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Tager 1–2 minutter.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter

from faelles import BLAA, INK, MUTED, REF, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
SEED = 42
B = 1000
ORANGE = "#eb6834"
GROEN = "#3a9a5b"
ORDEN_EM = ["A", "B", "C", "D", "E", "F", "G", "Intet"]


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def abs_pct(pred, pris):
    return np.abs(100 * (pred - pris) / pris)


def praecision(d, kol):
    pct = 100 * (d[kol] - d["pris"]) / d["pris"]
    return pd.Series({"N": len(d), "MAE (kr.)": np.mean(np.abs(d[kol] - d["pris"])),
                      "MdAPE (%)": np.median(np.abs(pct)), "Median %-fejl": np.median(pct),
                      "Inden for ±10 % (andel)": np.mean(np.abs(pct) <= 10),
                      "Inden for ±20 % (andel)": np.mean(np.abs(pct) <= 20)})


def bootstrap_mdape(fejl_dict, B=B, seed=SEED, blok=100):
    rng = np.random.default_rng(seed)
    n = len(next(iter(fejl_dict.values())))
    ud = {k: [] for k in fejl_dict}
    for start in range(0, B, blok):
        idx = rng.integers(0, n, size=(min(blok, B - start), n), dtype=np.int32)
        for k, f in fejl_dict.items():
            ud[k].append(np.median(f[idx], axis=1))
    return pd.DataFrame({k: np.concatenate(v) for k, v in ud.items()})


# =============================================================================
# DATA
# =============================================================================
MODEL_KOL = "p_nabo"   # trin 5-modellen fra praediktion_geo.py (før 2/11: "p_gb" fra forudsigelser_tidstro.csv)
fil = MAPPE / "forudsigelser_geo.csv"
if not fil.exists():
    raise SystemExit("forudsigelser_geo.csv mangler – kør praediktion_geo.py først.")
pred = pd.read_csv(fil, sep=";", decimal=",", encoding="utf-8-sig")
df, bolig = forbered_data(MAPPE)
d = df.merge(pred[["behandlingsid", "test", MODEL_KOL, "p_tidl"]], on="behandlingsid", how="inner")
d = d.dropna(subset=[MODEL_KOL])
d["p_model"] = d[MODEL_KOL]
d["ln_vurd"] = np.log(d["ejendomvaerdibeloeb"])
d["ln_gab"] = d["ln_pris"] - d["ln_vurd"]
d["ln_model"] = np.log(d["p_model"])
d["har_tidl_salg"] = d["p_tidl"].notna()

# Eksplicitte dummies (så pyfixest kan forudsige på testhandlerne)
dummynavne = {}
for col, ref in REF.items():
    niveauer = [x for x in sorted(d[col].unique()) if x != ref]
    for k, niv in enumerate(niveauer):
        d[f"d_{col}_{k}"] = (d[col] == niv).astype(int)
    dummynavne[col] = [f"d_{col}_{k}" for k in range(len(niveauer))]
bolig_eksplicit = bolig
for col, navne in dummynavne.items():
    bolig_eksplicit = re.sub(rf"i\({col}, ref='[^']*'\)", " + ".join(navne), bolig_eksplicit)
em = " + ".join(dummynavne["energimaerke"])
em_navne = {f"d_energimaerke_{k}": niv for k, niv in
            enumerate([x for x in sorted(d["energimaerke"].unique()) if x != REF["energimaerke"]])}

kal = d[~d["test"]].copy()
test = d[d["test"]].copy()
overskrift("Data")
print(f"Kalibreringshandler: {dk(len(kal))}   Testhandler: {dk(len(test))}")

# =============================================================================
# KORREKTIONER – ESTIMERET PÅ KALIBRERING, ANVENDT PÅ TEST
# =============================================================================
overskrift("Korrektioner (estimeret på kalibreringshandlerne)")
test["V0"] = test["ejendomvaerdibeloeb"]

faktor = (kal["pris"] / kal["ejendomvaerdibeloeb"]).groupby(kal["kvartal"]).median()
test["V1"] = test["V0"] * test["kvartal"].map(faktor)

m_v2 = pf.feols(f"ln_gab ~ {em} | kvartal", data=kal, vcov={"CRV1": "postnr"})
test["V2"] = np.exp(test["ln_vurd"] + m_v2.predict(newdata=test))
print("V2 – energimærke-korrektion af vurderingen (% i forhold til D, klynget std.fejl):")
t2 = m_v2.tidy()
for var, niv in em_navne.items():
    b, se = t2.loc[var, "Estimate"], t2.loc[var, "Std. Error"]
    print(f"  {niv:6s} {dk(100 * (np.exp(b) - 1), 1):>6} %   (std.fejl {se:.4f})")

# V2m: median-baseret energimærke-korrektion. OLS rammer GENNEMSNITTET af ln(P/V), som trækkes
# ned af enkelte handler langt under vurderingen (tunge haler). Da præcisionen måles med en
# median (MdAPE), bruges her i stedet medianforholdet pr. energimærke efter niveaukorrektionen.
kal["V1"] = kal["ejendomvaerdibeloeb"] * kal["kvartal"].map(faktor)
r_em = (kal["pris"] / kal["V1"]).groupby(kal["energimaerke"]).median()
test["V2m"] = test["V1"] * test["energimaerke"].map(r_em)
print("\nV2m – median-baseret energimærke-korrektion (medianforhold salgspris / V1 pr. energimærke):")
print("  " + "   ".join(f"{k}: {dk(100 * (v - 1), 1)} %" for k, v in r_em.reindex(ORDEN_EM).items()))
lav_gab = kal.groupby("energimaerke").apply(lambda g: (g["ln_gab"].mean(), g["ln_gab"].median()))
print("Gennemsnit vs. median af ln(P/V) pr. energimærke (viser de tunge haler):")
for k in ORDEN_EM:
    if k in lav_gab.index:
        mean_, med_ = lav_gab[k]
        print(f"  {k:6s} gennemsnit {mean_:+.3f}   median {med_:+.3f}   forskel {mean_ - med_:+.3f}")

m_v3 = pf.feols(f"ln_gab ~ {bolig_eksplicit} | kvartal + postnr", data=kal, vcov={"CRV1": "postnr"})
test["V3"] = np.exp(test["ln_vurd"] + m_v3.predict(newdata=test))
print(f"V3 – fuld korrektion: R² på kalibrering = {m_v3._r2:.3f}")

m_k1 = pf.feols("ln_pris ~ ln_vurd + ln_model | kvartal", data=kal, vcov={"CRV1": "postnr"})
test["K1"] = np.exp(m_k1.predict(newdata=test))
tk = m_k1.tidy()
print(f"K1 – kombination: vægt på ln(vurdering) = {tk.loc['ln_vurd', 'Estimate']:.3f} "
      f"(std.fejl {tk.loc['ln_vurd', 'Std. Error']:.3f}), vægt på ln(model) = "
      f"{tk.loc['ln_model', 'Estimate']:.3f} (std.fejl {tk.loc['ln_model', 'Std. Error']:.3f}), "
      f"sum = {tk['Estimate'].sum():.3f}")

m_k2 = pf.feols(f"ln_pris ~ ln_vurd + ln_model + {bolig_eksplicit} | kvartal + postnr", data=kal,
                vcov={"CRV1": "postnr"})
test["K2"] = np.exp(m_k2.predict(newdata=test))
tk2 = m_k2.tidy()
print(f"K2 – kombination + boligkarakteristika: vægt på ln(vurdering) = "
      f"{tk2.loc['ln_vurd', 'Estimate']:.3f}, på ln(model) = {tk2.loc['ln_model', 'Estimate']:.3f}")

faktor_m5 = (kal["pris"] / kal["p_model"]).groupby(kal["kvartal"]).median()
test["M5"] = test["p_model"] * test["kvartal"].map(faktor_m5)

MODELLER = {"V0": "V0 Rå vurdering", "V1": "V1 Vurdering, niveau pr. kvartal",
            "V2": "V2 + energimærke (OLS)", "V2m": "V2m + energimærke (median)", "V3": "V3 + boligkarakteristika og postnr.",
            "K1": "K1 Kombination vurdering + trin 5", "K2": "K2 Kombination + boligkarakteristika",
            "M5": "Trin 5-model alene (tidstro)"}
foer = len(test)
test = test.dropna(subset=list(MODELLER))
print(f"\nTesthandler med alle forudsigelser: {dk(len(test))} (udeladt {foer - len(test)}, "
      f"typisk postnumre uden kalibreringshandler)")

# =============================================================================
# TABEL 23: PRÆCISION
# =============================================================================
overskrift("Tabel 23 – præcision på testdata")
grupper = {"Alle testhandler": test, "Huse solgt før": test[test["har_tidl_salg"]],
           "Huse ikke solgt før": test[~test["har_tidl_salg"]],
           "Salg i 2020": test[test["salgsaar"] == 2020], "Salg i 2022": test[test["salgsaar"] == 2022]}
raekker = [praecision(g, k).rename((gnavn, MODELLER[k]))
           for gnavn, g in grupper.items() for k in MODELLER]
t23 = pd.DataFrame(raekker)
t23.index = pd.MultiIndex.from_tuples(t23.index, names=["Testdata", "Model"])
vis = t23.copy().astype(object)
vis["N"] = t23["N"].map(lambda x: dk(x))
vis["MAE (kr.)"] = t23["MAE (kr.)"].map(lambda x: dk(x))
for k in ["MdAPE (%)", "Median %-fejl"]:
    vis[k] = t23[k].map(lambda x: dk(x, 1))
for k in ["Inden for ±10 % (andel)", "Inden for ±20 % (andel)"]:
    vis[k] = t23[k].map(lambda x: f"{dk(100 * x, 1)} %")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(vis.to_string())
gem_tabel(t23.round(4), "tabel_23_korrigeret_vurdering.csv")

# =============================================================================
# TABEL 24: MdAPE OG MEDIAN %-FEJL PR. ENERGIMÆRKE
# =============================================================================
overskrift("Tabel 24 – pr. energimærke (MdAPE og median %-fejl)")
rk = {}
for niv in ORDEN_EM:
    g = test[test["energimaerke"] == niv]
    if len(g) == 0:
        continue
    r = {"N": len(g)}
    for k in ["V1", "V2", "V2m", "K1", "M5"]:
        pct = 100 * (g[k] - g["pris"]) / g["pris"]
        r[f"MdAPE {k}"] = np.median(np.abs(pct))
        r[f"Median %-fejl {k}"] = np.median(pct)
    rk[niv] = r
t24 = pd.DataFrame(rk).T
vis = t24.round(1).astype(object)
vis["N"] = t24["N"].map(lambda x: dk(x))
print(vis.to_string())
print("\nMedian %-fejl < 0: forudsigelsen ligger under salgsprisen (huset undervurderes).")
gem_tabel(t24.round(3), "tabel_24_korrigeret_energimaerke.csv")

# =============================================================================
# TABEL 25: BOOTSTRAP – FORSKEL I MdAPE I FORHOLD TIL V1
# =============================================================================
overskrift(f"Tabel 25 – bootstrap ({B} gentrækninger): forskel i MdAPE i forhold til V1")
bs_alle = {}
rk = []
for gnavn in ["Alle testhandler", "Huse solgt før", "Huse ikke solgt før"]:
    g = grupper[gnavn]
    res = bootstrap_mdape({k: abs_pct(g[k], g["pris"]).to_numpy() for k in MODELLER})
    bs_alle[gnavn] = res
    for k in MODELLER:
        if k == "V1":
            continue
        diff = res[k] - res["V1"]
        rk.append(pd.Series({
            "MdAPE": np.median(abs_pct(g[k], g["pris"])),
            "MdAPE V1": np.median(abs_pct(g["V1"], g["pris"])),
            "Forskel, pp": np.median(abs_pct(g[k], g["pris"])) - np.median(abs_pct(g["V1"], g["pris"])),
            "95 %-KI nedre": diff.quantile(0.025), "95 %-KI øvre": diff.quantile(0.975),
            "Andel bedre end V1": (diff < 0).mean()}, name=(gnavn, MODELLER[k])))
t25 = pd.DataFrame(rk)
t25.index = pd.MultiIndex.from_tuples(t25.index, names=["Testdata", "Model"])
print(t25.round(2).to_string())
print("\nNegativ forskel = mere præcis end den niveaukorrigerede vurdering (V1).")
gem_tabel(t25.round(4), "tabel_25_korrigeret_bootstrap.csv")

# =============================================================================
# TIDSTRO ROBUSTHED: ENERGIMÆRKE-KORREKTION FRA 2020 ANVENDT PÅ 2022
# =============================================================================
overskrift("Tidstro robusthed: energimærke-korrektion estimeret på 2020-salg, anvendt på 2022")
salg20 = d[d["salgsaar"] == 2020]          # alle 2020-salg er historiske set fra 2022
t22 = test[test["salgsaar"] == 2022].copy()
m_t = pf.feols(f"ln_gab ~ {em} | kvartal", data=salg20, vcov={"CRV1": "postnr"})
gamma = m_t.coef()[[c for c in m_t.coef().index if c.startswith("d_energimaerke")]]
korr = sum(gamma[c] * t22[c] for c in gamma.index)       # kun relative forskelle (D = 0)
t22["V0_em"] = t22["V0"] * np.exp(korr)
t22["V1_em"] = t22["V1"] * np.exp(korr)
# Median-version: medianforhold pr. energimærke i 2020 relativt til alle 2020-salg
f20 = salg20["pris"] / salg20["ejendomvaerdibeloeb"]
r20 = f20.groupby(salg20["energimaerke"]).median() / f20.median()
t22["V0_emm"] = t22["V0"] * t22["energimaerke"].map(r20)
t22["V1_emm"] = t22["V1"] * t22["energimaerke"].map(r20)
rk = {}
for k, navn in [("V0", "Rå vurdering"), ("V0_em", "Rå vurdering + energimærke (2020-koef.)"),
                ("V0_emm", "Rå vurdering + energimærke (2020-median)"),
                ("V1", "Niveau pr. kvartal"), ("V1_em", "Niveau pr. kvartal + energimærke (2020-koef.)"),
                ("V1_emm", "Niveau pr. kvartal + energimærke (2020-median)")]:
    rk[navn] = praecision(t22, k)
tt = pd.DataFrame(rk).T
vis = tt[["MdAPE (%)", "Median %-fejl", "Inden for ±20 % (andel)"]].round(2).astype(object)
vis.insert(0, "N", tt["N"].map(lambda x: dk(x)))
print(vis.to_string())
res = bootstrap_mdape({k: abs_pct(t22[k], t22["pris"]).to_numpy()
                       for k in ["V0", "V0_em", "V0_emm", "V1", "V1_em", "V1_emm"]})
for a, b in [("V0_em", "V0"), ("V0_emm", "V0"), ("V1_em", "V1"), ("V1_emm", "V1")]:
    diff = res[a] - res[b]
    print(f"{a} − {b}: {np.median(abs_pct(t22[a], t22['pris'])) - np.median(abs_pct(t22[b], t22['pris'])):+.2f} pp "
          f"[{diff.quantile(0.025):+.2f}; {diff.quantile(0.975):+.2f}]")

# =============================================================================
# FIGUR 16: a) MdAPE MED 95 %-KI  b) MEDIAN %-FEJL PR. ENERGIMÆRKE
# =============================================================================
overskrift("Figur")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54),
                             gridspec_kw={"width_ratios": [1.2, 1.1]})
koder = list(MODELLER)
farver = {"V0": MUTED, "V1": ORANGE, "V2": ORANGE, "V2m": ORANGE, "V3": ORANGE, "K1": GROEN, "K2": GROEN, "M5": BLAA}
res = bs_alle["Alle testhandler"]
for i, k in enumerate(koder):
    m = np.median(abs_pct(test[k], test["pris"]))
    lo, hi = res[k].quantile(0.025), res[k].quantile(0.975)
    a1.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=farver[k], markersize=4,
                capsize=2, elinewidth=1.2)
    a1.text(hi + 0.08, i, dk(m, 1), va="center", fontsize=6.5, color=farver[k])
a1.set_yticks(range(len(koder)), [MODELLER[k] for k in koder], fontsize=7)
a1.invert_yaxis()
a1.grid(axis="y", visible=False)
a1.axvline(np.median(abs_pct(test["V1"], test["pris"])), color=ORANGE, lw=0.7, ls="--")
a1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x, 1)))
a1.set_xlim(right=a1.get_xlim()[1] + 0.2)   # plads til tallet ved V0 inden for akserne (rettet 19/11)
a1.set_xlabel("MdAPE (%), 95 %-bootstrap-KI")
a1.set_title("a) Præcision på testdata", loc="left", fontsize=8.5)

x = np.arange(len(t24))
for j, (k, farve, navn) in enumerate([("V1", ORANGE, "Vurdering (V1)"),
                                      ("V2m", INK, "+ energimærke, median (V2m)"),
                                      ("M5", BLAA, "Model alene (trin 5)")]):
    a2.bar(x + (j - 1) * 0.27, t24[f"Median %-fejl {k}"], width=0.26, color=farve, label=navn)
a2.axhline(0, color=MUTED, lw=0.8)
a2.set_xticks(x, t24.index, fontsize=6.5)
a2.set_xlim(-0.6, len(t24) - 0.4)
a2.grid(axis="x", visible=False)
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{dk(v)} %"))
a2.set_title("b) Median %-fejl pr. energimærke\n(> 0: forudsagt over salgsprisen)",
             loc="left", fontsize=8.5)
fig.legend(*a2.get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower right",
           ncol=3, bbox_to_anchor=(1.0, -0.01), title="Panel b:", title_fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(UD / "fig_16_korrigeret_vurdering.png")
plt.close(fig)
print("  gemt: fig_16_korrigeret_vurdering.png")
print(f"\nAlle filer ligger i: {UD}")
