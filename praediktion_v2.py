"""
praediktion_v2.py – En bedre prædiktionsmodel (analyse 5.11)

Kan en model baseret på registerdata komme tættere på – eller slå – den offentlige vurdering,
når den får bedre information om beliggenhed og om netop dette hus? Hovedmodellen (M4) ændres
IKKE; den bruges fortsat til at forklare prisen. Dette er en separat prædiktionsmodel.

Samme testdata som praediktion.py (samme SEED og samme udtrækning), så resultaterne kan
sammenlignes direkte med tabel 10.

Tre forbedringer
  1. Vejniveau: vejens gennemsnitlige afvigelse fra M4's forudsigelse (residual) i træningsdata,
     "trukket mod 0" (= postnummerets niveau), når der er få salg på vejen (shrinkage):
         vej_effekt = Σ residualer / (n + K),   K = σ²_inden for vej / σ²_mellem veje
     Husets egne salg indgår aldrig i dets egen vej_effekt (heller ikke i træningsdata).
  2. Tidligere salg af samme hus (fra 2015): seneste salgspris FØR den aktuelle handel,
     fremskrevet med et regionalt kvartalsprisindeks. Kun salg uden for testdata bruges.
  3. Gradient boosting (HistGradientBoostingRegressor) i stedet for random forest.

Modeller på testdata
  M4 (som tabel 10)         hedonisk model, tilbagetransformeret med e^(σ²/2)
  M4                        hedonisk model, median-forudsigelse e^(ln P̂)
  M4 + vej                  M4 + vej_effekt
  Gradient boosting (basis) samme variable som random forest i praediktion.py
  Gradient boosting (fuld)  + vej_effekt + tidligere salg
  Tidligere salg            kun den fremskrevne tidligere salgspris (kun huse solgt før)
  Offentlig vurdering (rå og niveaukorrigeret)

Output i figurer_og_tabeller:
  tabel_17_praediktion_v2.csv, tabel_18_praediktion_v2_region.csv,
  tabel_19_variabelbetydning.csv, fig_13_mdape_modeller.png, fig_14_fejlfordeling.png

Kræver: data_renset.csv med kolonnen vejnavn (kør den nyeste rensning.py).
faelles.py skal ligge i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Kan tage 5–10 minutter.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from faelles import BLAA, INK, MUTED, REF, REGION_RAEKKEFOELGE, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
TEST_AAR = [2020, 2022]      # samme som praediktion.py
TEST_ANDEL = 0.5
SEED = 42
VALIDERING = 0.1             # andel af træningsdata, der holdes ude til σ² og variabelbetydning
ORANGE = "#eb6834"
GROEN = "#3a9a5b"


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def praecision(d, kol):
    fejl = d[kol] - d["pris"]
    pct = 100 * fejl / d["pris"]
    return pd.Series({
        "N": len(d),
        "RMSE (kr.)": np.sqrt(np.mean(fejl ** 2)),
        "MAE (kr.)": np.mean(np.abs(fejl)),
        "MdAPE (%)": np.median(np.abs(pct)),
        "Median %-fejl": np.median(pct),
        "Inden for ±10 % (andel)": np.mean(np.abs(pct) <= 10),
        "Inden for ±20 % (andel)": np.mean(np.abs(pct) <= 20),
    })


# =============================================================================
# DATA – PRÆCIS SAMME OPDELING SOM praediktion.py
# =============================================================================
df, bolig = forbered_data(MAPPE)
if "vejnavn" not in df.columns:
    raise SystemExit("data_renset.csv mangler kolonnen 'vejnavn' – kør den nyeste rensning.py først.")

sammenlignelig = (df["salgsaar"].isin(TEST_AAR) & (df["vurd_matcher_salgsaar"] == 1)
                  & (df["ejendomvaerdibeloeb"] > 0))
rng = np.random.default_rng(SEED)
df["test"] = sammenlignelig & (rng.random(len(df)) < TEST_ANDEL)
df["dato"] = pd.to_datetime(df["koebsaftaledato"])
df["vej"] = (df["postnr"] + "_" + df["vejnavn"].astype(str).str.strip().str.lower()).where(
    df["vejnavn"].notna())
df["bfe"] = df["bestemtfastejendombfenr"]

# Træningsdata deles i en del til estimation og en valideringsdel (bruges til σ² og
# variabelbetydning – aldrig testdata)
rng_v = np.random.default_rng(SEED + 1)
df["valid"] = ~df["test"] & (rng_v.random(len(df)) < VALIDERING)

overskrift("Opdeling")
print(f"Træning: {dk((~df['test'] & ~df['valid']).sum())}   Validering: {dk(df['valid'].sum())}"
      f"   Test: {dk(df['test'].sum())}  (samme testhandler som praediktion.py)")
print(f"Handler med vejnavn: {dk(df['vej'].notna().sum())} ({100 * df['vej'].notna().mean():.1f} %)"
      f"   Antal veje: {dk(df['vej'].nunique())}")

# =============================================================================
# 1) HEDONISK MODEL (M4) – SOM I praediktion.py
# =============================================================================
overskrift("1) Hedonisk model (M4)")
dummynavne = {}
for col, ref in REF.items():
    niveauer = [n for n in sorted(df[col].unique()) if n != ref]
    for k, niv in enumerate(niveauer):
        df[f"d_{col}_{k}"] = (df[col] == niv).astype(int)
    dummynavne[col] = [f"d_{col}_{k}" for k in range(len(niveauer))]
train = df[~df["test"]].copy()   # træning + validering (som i praediktion.py)
bolig_eksplicit = bolig
for col, navne in dummynavne.items():
    bolig_eksplicit = re.sub(rf"i\({col}, ref='[^']*'\)", " + ".join(navne), bolig_eksplicit)
m4 = pf.feols(f"ln_pris ~ {bolig_eksplicit} | kvartal + postnr", data=train, vcov="iid")
df["ln_hat_ols"] = m4.predict(newdata=df)
sigma2_ols = float(np.var(m4._u_hat))
print(f"R² = {m4._r2:.3f}   σ² = {sigma2_ols:.4f}")

# =============================================================================
# 2) VEJNIVEAU MED SHRINKAGE
# =============================================================================
overskrift("2) Vejniveau")
df["resid"] = df["ln_pris"] - df["ln_hat_ols"]
tr = df[~df["test"] & df["vej"].notna() & df["resid"].notna()]
vej_S = tr.groupby("vej")["resid"].sum()
vej_n = tr.groupby("vej")["resid"].size()
hus_S = tr.groupby(["vej", "bfe"])["resid"].sum()
hus_n = tr.groupby(["vej", "bfe"])["resid"].size()

# K estimeres fra data: K = σ²_inden for vej / σ²_mellem veje (empirisk Bayes-shrinkage)
inden = tr["resid"] - tr.groupby("vej")["resid"].transform("mean")
s2_inden = float((inden ** 2).sum() / (len(tr) - len(vej_n)))
mange = vej_n[vej_n >= 2]
s2_mellem = max(float(((vej_S / vej_n)[mange.index]).var() - (s2_inden / mange).mean()), 1e-6)
K = s2_inden / s2_mellem
print(f"σ² inden for vej = {s2_inden:.4f}   σ² mellem veje = {s2_mellem:.4f}   K = {K:.1f}")
print(f"(En vej med {K:.0f} salg får vægten 50 % på sit eget niveau og 50 % på postnummerets.)")

idx = pd.MultiIndex.from_arrays([df["vej"], df["bfe"]])
S = df["vej"].map(vej_S).fillna(0).to_numpy() - hus_S.reindex(idx).fillna(0).to_numpy()
n = df["vej"].map(vej_n).fillna(0).to_numpy() - hus_n.reindex(idx).fillna(0).to_numpy()
df["vej_effekt"] = S / (n + K)            # husets egne salg er trukket fra
df["vej_antal"] = n                        # antal andre salg på vejen i træningsdata
print(f"Testhandler med mindst ét andet salg på vejen: "
      f"{100 * (df.loc[df['test'], 'vej_antal'] > 0).mean():.1f} %")
print(f"Spredning på vej_effekt (testdata): {df.loc[df['test'], 'vej_effekt'].std():.3f}")

# =============================================================================
# 3) TIDLIGERE SALG AF SAMME HUS
# =============================================================================
overskrift("3) Tidligere salg af samme hus")
d15, _ = forbered_data(MAPPE, fra_aar=2015)
d15["dato"] = pd.to_datetime(d15["koebsaftaledato"])
# Testhandler må ikke bruges som "tidligere salg" for andre handler
test_id = set(df.loc[df["test"], "behandlingsid"])
pulje = d15[~d15["behandlingsid"].isin(test_id)].copy()

# Regionalt kvartalsprisindeks: gennemsnitlig ln(kr./m²) pr. region og kvartal, efter at
# postnummerets gennemsnit er trukket fra (så indekset ikke påvirkes af, hvor i regionen der
# sælges mest i et givet kvartal). Kun data uden for testdata.
pulje["r"] = np.log(pulje["kr_pr_m2"]) - pulje.groupby("postnr")["kr_pr_m2"].transform(
    lambda s: np.log(s).mean())
indeks = pulje.groupby(["region", "kvartal"])["r"].mean()
print("Regionalt prisindeks (log-point i forhold til 2019K1), udvalgte kvartaler:")
vis_idx = indeks.unstack("region")
vis_idx = vis_idx - vis_idx.loc["2019Q1"] if "2019Q1" in vis_idx.index else vis_idx
print(vis_idx.loc[[k for k in ["2015Q1", "2019Q1", "2020Q1", "2022Q1", "2024Q1", "2026Q1"]
                   if k in vis_idx.index]].round(3).to_string())

hoejre = (pulje[["bestemtfastejendombfenr", "dato", "pris", "kvartal"]]
          .rename(columns={"bestemtfastejendombfenr": "bfe", "dato": "tidl_dato",
                           "pris": "tidl_pris", "kvartal": "tidl_kvartal"})
          .dropna(subset=["bfe", "tidl_dato"]).sort_values("tidl_dato"))
hoejre["tidl_dato_kopi"] = hoejre["tidl_dato"]
venstre = df[["bfe", "dato"]].reset_index().dropna(subset=["dato"]).sort_values("dato")
m = pd.merge_asof(venstre, hoejre, left_on="dato", right_on="tidl_dato_kopi", by="bfe",
                  direction="backward", allow_exact_matches=False).set_index("index")
df = df.join(m[["tidl_pris", "tidl_kvartal", "tidl_dato"]])
nu = pd.MultiIndex.from_arrays([df["region"], df["kvartal"]])
foer = pd.MultiIndex.from_arrays([df["region"], df["tidl_kvartal"]])
df["ln_tidl_fremskrevet"] = (np.log(df["tidl_pris"]) + indeks.reindex(nu).to_numpy()
                             - indeks.reindex(foer).to_numpy())
df["aar_siden_salg"] = (df["dato"] - df["tidl_dato"]).dt.days / 365.25
df["har_tidl_salg"] = df["ln_tidl_fremskrevet"].notna().astype(int)
print(f"\nAndel med tidligere salg: træning {100 * df.loc[~df['test'], 'har_tidl_salg'].mean():.1f} %"
      f"   test {100 * df.loc[df['test'], 'har_tidl_salg'].mean():.1f} %")
print("År siden tidligere salg (testdata):")
print(df.loc[df["test"], "aar_siden_salg"].describe().round(1).to_string())

# =============================================================================
# 4) GRADIENT BOOSTING
# =============================================================================
overskrift("4) Gradient boosting")
postnr_niveau = df[~df["test"]].groupby("postnr")["ln_pris"].mean()
df["postnr_niveau"] = df["postnr"].map(postnr_niveau)
df["postnr_tal"] = pd.to_numeric(df["postnr"], errors="coerce")
df["tid"] = df["kvartal"].str[:4].astype(int) * 4 + df["kvartal"].str[-1].astype(int)

basis = ["ln_boligareal", "ln_grundareal", "raekkehus", "vaerelser", "to_etager_plus",
         "ombygget", "har_garage", "har_carport", "har_udhus", "har_udestue",
         "postnr_niveau", "postnr_tal", "tid"]
kategoriske = ["badevaerelser_kat", "opfoerelsesperiode", "energimaerke", "opvarmning",
               "ydervaeg", "tag", "region"]
nye = ["vej_effekt", "vej_antal", "ln_tidl_fremskrevet", "aar_siden_salg"]
X = pd.get_dummies(df[basis + nye + kategoriske], columns=kategoriske, dtype=float)
kol_basis = [k for k in X.columns if k not in nye]
fit_rows = ~df["test"] & ~df["valid"]
y = df["ln_pris"]

modeller_gb = {}
for navn, kol in [("basis", kol_basis), ("fuld", list(X.columns))]:
    gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=2000, max_leaf_nodes=63,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=50, random_state=SEED)
    gb.fit(X.loc[fit_rows, kol], y[fit_rows])
    df[f"ln_hat_gb_{navn}"] = gb.predict(X[kol])
    res_v = y[df["valid"]] - df.loc[df["valid"], f"ln_hat_gb_{navn}"]
    r2_v = 1 - res_v.var() / y[df["valid"]].var()
    print(f"Gradient boosting ({navn}): {gb.n_iter_} træer   R² på valideringsdata = {r2_v:.3f}"
          f"   σ² = {res_v.var():.4f}")
    modeller_gb[navn] = (gb, kol)

# Variabelbetydning: hvor meget falder R² på valideringsdata, når én variabel blandes tilfældigt?
gb_fuld, kol_fuld = modeller_gb["fuld"]
v = df[df["valid"]].sample(min(20_000, int(df["valid"].sum())), random_state=SEED)
pi = permutation_importance(gb_fuld, X.loc[v.index, kol_fuld], y[v.index], n_repeats=3,
                            random_state=SEED, scoring="r2")
t19 = (pd.DataFrame({"Fald i R²": pi.importances_mean, "Std.": pi.importances_std},
                    index=kol_fuld).sort_values("Fald i R²", ascending=False))
print("\nVariabelbetydning (permutation importance på valideringsdata), top 12:")
print(t19.head(12).round(3).to_string())
gem_tabel(t19.round(4), "tabel_19_variabelbetydning.csv")

# =============================================================================
# FORUDSIGELSER I KRONER
# =============================================================================
# Log-modellerne forudsiger medianen af prisen: e^(ln P̂). Korrektionen e^(σ²/2) giver i stedet
# gennemsnittet, men skubber medianfejlen op. Da hovedmålet er MdAPE, bruges median-
# forudsigelsen her. "M4 (som tabel 10)" vises med korrektion for at kunne genfinde tabel 10.
df["p_ols_korr"] = np.exp(df["ln_hat_ols"] + sigma2_ols / 2)
df["p_ols"] = np.exp(df["ln_hat_ols"])
df["p_ols_vej"] = np.exp(df["ln_hat_ols"] + df["vej_effekt"])
df["p_gb_basis"] = np.exp(df["ln_hat_gb_basis"])
df["p_gb_fuld"] = np.exp(df["ln_hat_gb_fuld"])
df["p_tidl"] = np.exp(df["ln_tidl_fremskrevet"])
df["p_vurd"] = df["ejendomvaerdibeloeb"]
faktor = (df[~df["test"] & sammenlignelig].assign(f=lambda d: d["pris"] / d["ejendomvaerdibeloeb"])
          .groupby("salgsaar")["f"].median())
df["p_vurd_korr"] = df["p_vurd"] * df["salgsaar"].map(faktor)

MODELLER = {"p_ols_korr": "M4 (som tabel 10)", "p_ols": "M4", "p_ols_vej": "M4 + vej",
            "p_gb_basis": "Gradient boosting (basis)", "p_gb_fuld": "Gradient boosting (fuld)",
            "p_tidl": "Kun tidligere salg (fremskrevet)",
            "p_vurd": "Offentlig vurdering", "p_vurd_korr": "Vurdering, niveaukorrigeret"}
FAELLES = [k for k in MODELLER if k != "p_tidl"]

# =============================================================================
# TABEL 17: PRÆCISION PÅ TESTDATA
# =============================================================================
overskrift("Tabel 17 – præcision på testdata")
test = df[df["test"]].dropna(subset=FAELLES).copy()
print(f"Testhandler med forudsigelse fra alle modeller: {dk(len(test))}")
grupper = {"Alle testhandler": test,
           "Huse solgt før (siden 2015)": test[test["har_tidl_salg"] == 1],
           "Huse ikke solgt før": test[test["har_tidl_salg"] == 0]}
for aar in TEST_AAR:
    grupper[f"Salg i {aar}"] = test[test["salgsaar"] == aar]
raekker = []
for gnavn, d in grupper.items():
    for kol, mnavn in MODELLER.items():
        if d[kol].notna().all() and len(d) > 0:
            raekker.append(praecision(d, kol).rename((gnavn, mnavn)))
t17 = pd.DataFrame(raekker)
t17.index = pd.MultiIndex.from_tuples(t17.index, names=["Testdata", "Model"])
vis = t17.copy().astype(object)
for k in ["RMSE (kr.)", "MAE (kr.)", "N"]:
    vis[k] = t17[k].map(lambda x: dk(x))
for k in ["MdAPE (%)", "Median %-fejl"]:
    vis[k] = t17[k].map(lambda x: dk(x, 1))
for k in ["Inden for ±10 % (andel)", "Inden for ±20 % (andel)"]:
    vis[k] = t17[k].map(lambda x: f"{dk(100 * x, 1)} %")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(vis.to_string())
gem_tabel(t17.round(4), "tabel_17_praediktion_v2.csv")

# =============================================================================
# TABEL 18: MdAPE PR. REGION
# =============================================================================
overskrift("Tabel 18 – MdAPE (%) pr. region")
vis_mod = ["p_ols", "p_ols_vej", "p_gb_fuld", "p_vurd", "p_vurd_korr"]
t18 = pd.DataFrame({MODELLER[k]: test.groupby("region").apply(
    lambda g, k=k: np.median(np.abs(100 * (g[k] - g["pris"]) / g["pris"])))
    for k in vis_mod}).reindex(REGION_RAEKKEFOELGE)
t18.insert(0, "N", test.groupby("region").size().reindex(REGION_RAEKKEFOELGE))
print(t18.round(1).to_string())
gem_tabel(t18.round(3), "tabel_18_praediktion_v2_region.csv")

# =============================================================================
# FIGUR 13: MdAPE PR. MODEL, OPDELT PÅ TIDLIGERE SALG
# =============================================================================
overskrift("Figurer")
fig_mod = ["p_ols", "p_ols_vej", "p_gb_basis", "p_gb_fuld", "p_vurd", "p_vurd_korr"]
fig_grp = [("Alle testhandler", INK), ("Huse solgt før (siden 2015)", BLAA),
           ("Huse ikke solgt før", ORANGE)]
fig, ax = plt.subplots(figsize=(16 / 2.54, 8.5 / 2.54))
yy = np.arange(len(fig_mod))
for j, (g, farve) in enumerate(fig_grp):
    vaerdier = [t17.loc[(g, MODELLER[k]), "MdAPE (%)"] if (g, MODELLER[k]) in t17.index
                else np.nan for k in fig_mod]
    ax.barh(yy + (j - 1) * 0.27, vaerdier, height=0.26, color=farve, label=g)
    for yi, val in zip(yy, vaerdier):
        if pd.notna(val):
            ax.text(val + 0.2, yi + (j - 1) * 0.27, dk(val, 1), va="center", fontsize=6.5,
                    color=farve)
ax.set_yticks(yy, [MODELLER[k] for k in fig_mod], fontsize=8)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Median absolut %-fejl (MdAPE) på testdata – lavere er bedre")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.legend(frameon=False, fontsize=7, loc="lower right")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Præcision: modeller med vejniveau og tidligere salg mod den offentlige vurdering",
# loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_13_mdape_modeller.png")
plt.close(fig)
print("  gemt: fig_13_mdape_modeller.png")

# =============================================================================
# FIGUR 14: FORDELING AF %-FEJL – BEDSTE MODEL VS. VURDERING
# =============================================================================
fig, ax = plt.subplots(figsize=(16 / 2.54, 7.5 / 2.54))
bins = np.linspace(-60, 60, 121)
for kol, farve in [("p_ols", MUTED), ("p_gb_fuld", BLAA), ("p_vurd_korr", ORANGE)]:
    pct = 100 * (test[kol] - test["pris"]) / test["pris"]
    ax.hist(pct, bins=bins, density=True, histtype="step", lw=1.4, color=farve)
ax.axvline(0, color=INK, lw=0.8, ls="--")
ax.set_yticks([])
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.set_xlabel("%-fejl: (forudsagt − faktisk) / faktisk  (vist mellem −60 % og +60 %)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
from matplotlib.lines import Line2D  # noqa: E402
ax.legend([Line2D([], [], color=f, lw=1.4) for f in [MUTED, BLAA, ORANGE]],
          [f"{MODELLER[k]} (MdAPE {dk(t17.loc[('Alle testhandler', MODELLER[k]), 'MdAPE (%)'], 1)} %)"
           for k in ["p_ols", "p_gb_fuld", "p_vurd_korr"]],
          frameon=False, fontsize=7, loc="upper right")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Fordeling af forudsigelsesfejl på testdata", loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(UD / "fig_14_fejlfordeling.png")
plt.close(fig)
print("  gemt: fig_14_fejlfordeling.png")
print(f"\nAlle filer ligger i: {UD}")
