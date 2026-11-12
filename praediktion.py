"""
praediktion.py – Hedonisk model vs. offentlig ejendomsvurdering (analyse 5.3)

Hvor præcist forudsiger hovedmodellen salgspriser på huse, den ikke har set, sammenlignet med
den offentlige ejendomsvurdering? Random forest bruges som benchmark.

Design
  Testdata:  salg i 2020 med 2020-vurdering og salg i 2022 med 2022-vurdering
             (vurderingsår = salgsår). Halvdelen trækkes tilfældigt ud som testdata.
  Træning:   alle øvrige handler 2019+ (inkl. den anden halvdel). Testhandlerne indgår ikke.
  Modeller:  1) Hedonisk model (M4): OLS med postnummer- og kvartals-dummies
             2) Random forest (benchmark): samme karakteristika + postnummer som tal og
                postnummerets gennemsnitlige log-pris i træningsdata
             3) Offentlig vurdering: ejendomsværdien som den er
             4) Offentlig vurdering, niveaukorrigeret: ganget med medianforholdet
                salgspris/vurdering i træningsdata for samme år. Viser, hvor godt vurderingen
                rangerer husene, når et generelt niveauskift (f.eks. forsigtighed) fjernes.
  Mål:       RMSE, MAE, median absolut %-fejl (MdAPE), gennemsnitlig %-fejl (skævhed) og
             andel forudsagt inden for ±10 % og ±20 % af salgsprisen.
  Robusthed: kun salg i 1. kvartal 2020 (tæt på vurderingsdatoen).

Output i figurer_og_tabeller:
  tabel_10_praediktion.csv           præcision pr. model og år
  tabel_11_praediktion_region.csv    MdAPE pr. region og model
  fig_7a_forudsagt_vs_faktisk.png    forudsagt mod faktisk pris (model og vurdering)
  fig_7b_fejl_pr_region.png          fordeling af %-fejl pr. region (model og vurdering)

Kræver: pip3 install pyfixest scikit-learn. faelles.py skal ligge i samme mappe.
Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Random forest kan tage nogle minutter.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from sklearn.ensemble import RandomForestRegressor

import re

from faelles import BLAA, INK, MUTED, REF, REGION_RAEKKEFOELGE, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
TEST_AAR = [2020, 2022]      # år hvor vurderingsår = salgsår giver en fair sammenligning
TEST_ANDEL = 0.5             # andel af de sammenlignelige handler, der bruges som testdata
SEED = 42                    # fast tilfældighed, så resultaterne kan genskabes
RF_TRAER = 150               # antal træer i random forest
RF_STIKPROEVE = 0.5          # hvert træ trænes på 50 % af træningsdata (bagging)
ORANGE = "#eb6834"


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


# =============================================================================
# DATA OG OPDELING I TRÆNING/TEST
# =============================================================================
df, bolig = forbered_data(MAPPE)

sammenlignelig = (df["salgsaar"].isin(TEST_AAR) & (df["vurd_matcher_salgsaar"] == 1)
                  & (df["ejendomvaerdibeloeb"] > 0))
rng = np.random.default_rng(SEED)
df["test"] = sammenlignelig & (rng.random(len(df)) < TEST_ANDEL)
train = df[~df["test"]].copy()
test = df[df["test"]].copy()

overskrift("Opdeling")
print("Sammenlignelige handler (vurderingsår = salgsår):")
print(df[sammenlignelig].groupby("salgsaar").size().to_string())
print(f"\nTræningsdata: {dk(len(train))} handler   Testdata: {dk(len(test))} handler")

# =============================================================================
# 1) HEDONISK MODEL (M4)
# =============================================================================
overskrift("1) Hedonisk model (M4)")
# Samme specifikation som i hovedmodel.py, men dummies laves eksplicit i stedet for med i(),
# fordi pyfixest kun kan forudsige på nye data, når modellen er skrevet med almindelige
# variable. Matematisk er det præcis den samme model.
dummynavne = {}
for col, ref in REF.items():
    niveauer = [n for n in sorted(df[col].unique()) if n != ref]
    navne = []
    for k, niv in enumerate(niveauer):
        navn = f"d_{col}_{k}"
        for d in (train, test):
            d[navn] = (d[col] == niv).astype(int)
        navne.append(navn)
    dummynavne[col] = navne
bolig_eksplicit = bolig
for col, navne in dummynavne.items():
    bolig_eksplicit = re.sub(rf"i\({col}, ref='[^']*'\)", " + ".join(navne), bolig_eksplicit)
m4 = pf.feols(f"ln_pris ~ {bolig_eksplicit} | kvartal + postnr", data=train, vcov="hetero")
sigma2 = float(np.var(m4._u_hat))   # variansen af residualerne i træningsdata
test["ln_hat_ols"] = m4.predict(newdata=test)
# Tilbagetransformation fra log til kroner: e^(ln P̂) · e^(σ²/2)
test["p_ols"] = np.exp(test["ln_hat_ols"]) * np.exp(sigma2 / 2)
print(f"R² i træningsdata: {m4._r2:.3f}   Residualvarians σ² = {sigma2:.4f}"
      f"   Korrektionsfaktor e^(σ²/2) = {np.exp(sigma2 / 2):.4f}")
print(f"Testhandler uden forudsigelse (postnummer/kvartal findes ikke i træningsdata): "
      f"{test['p_ols'].isna().sum()}")

# =============================================================================
# 2) RANDOM FOREST (BENCHMARK)
# =============================================================================
overskrift("2) Random forest")
# Postnummerets gennemsnitlige log-pris beregnes KUN på træningsdata (ellers data-leakage)
postnr_niveau = train.groupby("postnr")["ln_pris"].mean()
for d in (train, test):
    d["postnr_niveau"] = d["postnr"].map(postnr_niveau)
    d["postnr_tal"] = pd.to_numeric(d["postnr"], errors="coerce")
    d["tid"] = d["kvartal"].str[:4].astype(int) * 4 + d["kvartal"].str[-1].astype(int)  # kvartaler som tal

numeriske = ["ln_boligareal", "ln_grundareal", "raekkehus", "vaerelser", "to_etager_plus",
             "ombygget", "har_garage", "har_carport", "har_udhus", "har_udestue",
             "postnr_niveau", "postnr_tal", "tid"]
kategoriske = ["badevaerelser_kat", "opfoerelsesperiode", "energimaerke", "opvarmning",
               "ydervaeg", "tag", "region"]
alle = pd.get_dummies(pd.concat([train, test])[numeriske + kategoriske],
                      columns=kategoriske, dtype=float)
X_train, X_test = alle.loc[train.index], alle.loc[test.index]
ok_train = X_train.notna().all(axis=1)

rf = RandomForestRegressor(n_estimators=RF_TRAER, max_samples=RF_STIKPROEVE,
                           min_samples_leaf=5, max_features=0.33, n_jobs=-1,
                           oob_score=True, random_state=SEED)
rf.fit(X_train[ok_train], train.loc[ok_train, "ln_pris"])
sigma2_rf = float(np.var(train.loc[ok_train, "ln_pris"] - rf.oob_prediction_))
ok_test = X_test.notna().all(axis=1)
test["p_rf"] = np.nan
test.loc[ok_test, "p_rf"] = np.exp(rf.predict(X_test[ok_test])) * np.exp(sigma2_rf / 2)
print(f"Out-of-bag R² (på log-pris): {rf.oob_score_:.3f}   σ² = {sigma2_rf:.4f}")

vigtighed = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nVariabelbetydning (feature importance), top 12:")
print(vigtighed.head(12).round(3).to_string())

# =============================================================================
# 3–4) OFFENTLIG VURDERING
# =============================================================================
overskrift("3–4) Offentlig vurdering")
test["p_vurd"] = test["ejendomvaerdibeloeb"]
# Niveaukorrektion: medianforholdet salgspris/vurdering i TRÆNINGSdata for samme år
faktor = (train[train["salgsaar"].isin(TEST_AAR) & (train["vurd_matcher_salgsaar"] == 1)
                & (train["ejendomvaerdibeloeb"] > 0)]
          .assign(f=lambda d: d["pris"] / d["ejendomvaerdibeloeb"])
          .groupby("salgsaar")["f"].median())
print("Medianforhold salgspris/vurdering i træningsdata (bruges til niveaukorrektion):")
print(faktor.round(3).to_string())
test["p_vurd_korr"] = test["p_vurd"] * test["salgsaar"].map(faktor)

# =============================================================================
# TABEL 10: PRÆCISION
# =============================================================================
overskrift("Tabel 10 – præcision på testdata")
MODELLER = {"p_ols": "Hedonisk model (M4)", "p_rf": "Random forest",
            "p_vurd": "Offentlig vurdering", "p_vurd_korr": "Vurdering, niveaukorrigeret"}
test = test.dropna(subset=list(MODELLER))   # samme handler for alle modeller
print(f"Testhandler med forudsigelse fra alle modeller: {dk(len(test))}")


def praecision(d, kol):
    fejl = d[kol] - d["pris"]
    pct = 100 * fejl / d["pris"]
    return pd.Series({
        "N": len(d),
        "RMSE (kr.)": np.sqrt(np.mean(fejl ** 2)),
        "MAE (kr.)": np.mean(np.abs(fejl)),
        "MdAPE (%)": np.median(np.abs(pct)),
        "Gns. %-fejl (skævhed)": np.mean(pct),
        "Median %-fejl": np.median(pct),
        "Inden for ±10 % (andel)": np.mean(np.abs(pct) <= 10),
        "Inden for ±20 % (andel)": np.mean(np.abs(pct) <= 20),
    })


grupper = {"Alle testhandler": test}
for aar in TEST_AAR:
    grupper[f"Salg i {aar}"] = test[test["salgsaar"] == aar]
grupper["Robusthed: 1. kvartal 2020"] = test[test["kvartal"] == "2020Q1"]

raekker = []
for gnavn, d in grupper.items():
    for kol, mnavn in MODELLER.items():
        raekker.append(praecision(d, kol).rename((gnavn, mnavn)))
t10 = pd.DataFrame(raekker)
t10.index = pd.MultiIndex.from_tuples(t10.index, names=["Testdata", "Model"])
vis = t10.copy()
for k in ["RMSE (kr.)", "MAE (kr.)"]:
    vis[k] = vis[k].map(lambda x: dk(x))
for k in ["MdAPE (%)", "Gns. %-fejl (skævhed)", "Median %-fejl"]:
    vis[k] = vis[k].map(lambda x: dk(x, 1))
for k in ["Inden for ±10 % (andel)", "Inden for ±20 % (andel)"]:
    vis[k] = vis[k].map(lambda x: f"{dk(100 * x, 1)} %")
vis["N"] = vis["N"].map(lambda x: dk(x))
pd.set_option("display.width", 250)
print(vis.to_string())
gem_tabel(t10.round(4), "tabel_10_praediktion.csv")

# =============================================================================
# TABEL 11: PRÆCISION PR. REGION
# =============================================================================
overskrift("Tabel 11 – MdAPE (%) pr. region")
for kol in MODELLER:
    test[f"pct_{kol}"] = 100 * (test[kol] - test["pris"]) / test["pris"]
t11 = (test.groupby("region")[[f"pct_{k}" for k in MODELLER]]
       .agg(lambda s: np.median(np.abs(s)))
       .rename(columns={f"pct_{k}": v for k, v in MODELLER.items()})
       .reindex(REGION_RAEKKEFOELGE))
t11.insert(0, "N", test.groupby("region").size().reindex(REGION_RAEKKEFOELGE))
print(t11.round(1).to_string())
gem_tabel(t11.round(3), "tabel_11_praediktion_region.csv")

# =============================================================================
# FIGUR 7a: FORUDSAGT VS. FAKTISK
# =============================================================================
overskrift("Figurer")
FMT_MIO = FuncFormatter(lambda x, _: dk(x / 1e6, 1).rstrip("0").rstrip(","))
fig, akser = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54), sharex=True, sharey=True)
graense = (test["pris"].quantile(0.005), test["pris"].quantile(0.995))
for ax, (kol, titel) in zip(akser, [("p_ols", "Hedonisk model (M4)"),
                                     ("p_vurd", "Offentlig vurdering")]):
    ax.hexbin(test["pris"], test[kol], gridsize=60, xscale="log", yscale="log", mincnt=1,
              bins="log", cmap="Blues", linewidths=0,
              extent=(np.log10(graense[0]), np.log10(graense[1]),
                      np.log10(graense[0]), np.log10(graense[1])))
    ax.plot(graense, graense, color=INK, lw=1, label="Forudsagt = faktisk")
    ax.plot(graense, [g * 0.8 for g in graense], color=MUTED, lw=0.8, ls="--", label="±20 %")
    ax.plot(graense, [g * 1.2 for g in graense], color=MUTED, lw=0.8, ls="--")
    mdape = t10.loc[("Alle testhandler", MODELLER[kol]), "MdAPE (%)"]
    ax.set_title(f"{titel}\nMedian absolut fejl: {dk(mdape, 1)} %", loc="left", fontsize=9)
    ax.set_xlabel("Faktisk salgspris (mio. kr., log-skala)")
    for akse in (ax.xaxis, ax.yaxis):
        akse.set_major_locator(FixedLocator([0.25e6, 0.5e6, 1e6, 2e6, 4e6, 8e6, 16e6]))
        akse.set_major_formatter(FMT_MIO)
        akse.set_minor_formatter(NullFormatter())
    ax.set_xlim(graense)
    ax.set_ylim(graense)
akser[0].set_ylabel("Forudsagt pris / vurdering (mio. kr.)")
akser[0].legend(loc="upper left", fontsize=7)
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# fig.suptitle(f"Forudsagt mod faktisk salgspris på testdata (salg i {' og '.join(map(str, TEST_AAR))})",
# x=0.0, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(UD / "fig_7a_forudsagt_vs_faktisk.png")
plt.close(fig)
print("  gemt: fig_7a_forudsagt_vs_faktisk.png")

# =============================================================================
# FIGUR 7b: FEJL PR. REGION
# =============================================================================
fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
pos = np.arange(len(REGION_RAEKKEFOELGE))
for forskyd, kol, farve, navn in [(-0.2, "p_ols", BLAA, "Hedonisk model (M4)"),
                                  (0.2, "p_vurd", ORANGE, "Offentlig vurdering")]:
    data = [test.loc[test["region"] == r, f"pct_{kol}"].dropna() for r in REGION_RAEKKEFOELGE]
    bp = ax.boxplot(data, positions=pos + forskyd, widths=0.32, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6),
                    boxprops=dict(facecolor=farve, alpha=0.35, edgecolor=farve),
                    whiskerprops=dict(color=farve), capprops=dict(color=farve))
    bp["boxes"][0].set_label(navn)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(pos, REGION_RAEKKEFOELGE)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_ylabel("(Forudsagt − faktisk) / faktisk")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Fejl i forhold til salgsprisen pr. region (testdata)\n"
# "Boks: 25–75 %-fraktil, streg: median, knurhår: 5–95 %-fraktil", loc="left")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
fig.savefig(UD / "fig_7b_fejl_pr_region.png")
plt.close(fig)
print("  gemt: fig_7b_fejl_pr_region.png")
print(f"\nAlle filer ligger i: {UD}")
