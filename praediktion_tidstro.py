"""
praediktion_tidstro.py – Tidstro robusthedstjek af prædiktionsmodellen (analyse 5.13)

I praediktion_v2.py blev modellerne trænet på salg fra hele 2019–2026 – også salg i samme
kvartal som og efter testhandlen. Den offentlige vurdering kender kun forholdene på
vurderingsdatoen (1. januar). Her får modellerne præcis samme informationsgrundlag i tid:

  Testhandler i 2020: træning KUN på salg 2015–2019   (før vurderingsdatoen 1/1-2020)
  Testhandler i 2022: træning KUN på salg 2015–2021   (før vurderingsdatoen 1/1-2022)

Alle variable (vej-effekt, postnummerniveau, tidligere salg, prisindeks) beregnes kun på
træningsdata. Testhandlerne forudsiges på prisniveauet i 4. kvartal året før – ligesom
vurderingen er fastsat på prisniveauet ved årsskiftet.

Tre varianter (samme for modeller og vurdering, så sammenligningen er symmetrisk)
  a) Strengt tidstro:   ingen information fra salgsåret.
  b) Niveau pr. år:     både model og vurdering ganges med medianforholdet salgspris/forudsigelse
                        blandt de øvrige sammenlignelige handler i salgsåret (ikke testhandler).
  c) Niveau pr. kvartal: som b, men pr. salgskvartal – fjerner prisudviklingen i løbet af året,
                        så kun evnen til at rangere husene afgør præcisionen.

Usikkerhed: bootstrap (1.000 gentrækninger af testhandlerne) af MdAPE og af forskellen i
MdAPE mellem gradient boosting og vurderingen.

Output: forudsigelser_tidstro.csv (i MAPPE – bruges af vurdering_korrigeret.py) og i
figurer_og_tabeller:
  tabel_20_tidstro.csv, tabel_21_tidstro_grupper.csv, tabel_22_bootstrap.csv,
  fig_15_tidstro.png

Kræver: data_renset.csv med vejnavn. faelles.py i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. 5–15 minutter.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import HistGradientBoostingRegressor

from faelles import BLAA, INK, MUTED, REF, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
TEST_AAR = [2020, 2022]
TEST_ANDEL = 0.5
SEED = 42
B = 1000                      # antal bootstrap-gentrækninger
ORANGE = "#eb6834"
GB_PARAMS = dict(learning_rate=0.1, max_iter=3000, max_leaf_nodes=63, min_samples_leaf=40,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=50, random_state=SEED)


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def abs_pct(pred, pris):
    return np.abs(100 * (pred - pris) / pris)


def praecision(d, kol):
    fejl = d[kol] - d["pris"]
    pct = 100 * fejl / d["pris"]
    return pd.Series({"N": len(d), "MAE (kr.)": np.mean(np.abs(fejl)),
                      "MdAPE (%)": np.median(np.abs(pct)), "Median %-fejl": np.median(pct),
                      "Inden for ±10 % (andel)": np.mean(np.abs(pct) <= 10),
                      "Inden for ±20 % (andel)": np.mean(np.abs(pct) <= 20)})


def bootstrap_mdape(fejl_dict, B=B, seed=SEED, blok=100):
    """Bootstrap af MdAPE for hver model på de samme gentrukne testhandler.
    fejl_dict: {model: array med absolutte %-fejl}. Returnerer DataFrame (B × modeller)."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(fejl_dict.values())))
    ud = {k: [] for k in fejl_dict}
    for start in range(0, B, blok):
        idx = rng.integers(0, n, size=(min(blok, B - start), n), dtype=np.int32)
        for k, f in fejl_dict.items():
            ud[k].append(np.median(f[idx], axis=1))
    return pd.DataFrame({k: np.concatenate(v) for k, v in ud.items()})


# =============================================================================
# DATA – SAMME TESTHANDLER SOM praediktion.py OG praediktion_v2.py
# =============================================================================
df19, _ = forbered_data(MAPPE)
samml19 = (df19["salgsaar"].isin(TEST_AAR) & (df19["vurd_matcher_salgsaar"] == 1)
           & (df19["ejendomvaerdibeloeb"] > 0))
rng = np.random.default_rng(SEED)
test_id = set(df19.loc[samml19 & (rng.random(len(df19)) < TEST_ANDEL), "behandlingsid"])

d, bolig = forbered_data(MAPPE, fra_aar=2015)
if "vejnavn" not in d.columns:
    raise SystemExit("data_renset.csv mangler kolonnen 'vejnavn' – kør den nyeste rensning.py først.")
d["test"] = d["behandlingsid"].isin(test_id)
d["dato"] = pd.to_datetime(d["koebsaftaledato"])
d["bfe"] = d["bestemtfastejendombfenr"]
d["vej"] = (d["postnr"] + "_" + d["vejnavn"].astype(str).str.strip().str.lower()).where(
    d["vejnavn"].notna())
d["tid"] = d["kvartal"].str[:4].astype(int) * 4 + d["kvartal"].str[-1].astype(int)
d["sammenlignelig"] = ((d["vurd_matcher_salgsaar"] == 1) & (d["ejendomvaerdibeloeb"] > 0))

# Eksplicitte dummies (pyfixest kan kun forudsige på nye data på denne måde)
dummynavne = {}
for col, ref in REF.items():
    niveauer = [x for x in sorted(d[col].unique()) if x != ref]
    for k, niv in enumerate(niveauer):
        d[f"d_{col}_{k}"] = (d[col] == niv).astype(int)
    dummynavne[col] = [f"d_{col}_{k}" for k in range(len(niveauer))]
bolig_eksplicit = bolig
for col, navne in dummynavne.items():
    bolig_eksplicit = re.sub(rf"i\({col}, ref='[^']*'\)", " + ".join(navne), bolig_eksplicit)

basis = ["ln_boligareal", "ln_grundareal", "raekkehus", "vaerelser", "to_etager_plus",
         "ombygget", "har_garage", "har_carport", "har_udhus", "har_udestue",
         "postnr_niveau", "postnr_tal", "tid"]
kategoriske = ["badevaerelser_kat", "opfoerelsesperiode", "energimaerke", "opvarmning",
               "ydervaeg", "tag", "region"]
nye = ["vej_effekt", "vej_antal", "ln_tidl_fremskrevet", "aar_siden_salg"]

overskrift("Testhandler")
print(f"Testhandler fundet i 2015+-data: {dk(d['test'].sum())} af {dk(len(test_id))}")

# =============================================================================
# ÉN MODEL PR. TESTÅR, TRÆNET KUN PÅ SALG FØR 1. JANUAR I TESTÅRET
# =============================================================================
resultater = []
for aar in TEST_AAR:
    overskrift(f"Testår {aar}: træning på salg før 1/1-{aar}")
    skaering = pd.Timestamp(aar, 1, 1)
    sidste_kv = f"{aar - 1}Q4"
    tr = d[d["dato"] < skaering].copy()
    # NB: testhandler fra 2020 indgår i træningen for 2022 – de er på det tidspunkt historiske
    # salg, som alle kunne kende. De evalueres kun af 2020-modellen, der aldrig har set dem.
    # Mål: alle sammenlignelige handler i teståret. Testhandlerne evalueres; de øvrige
    # ("kalibrering") bruges kun til niveaukorrektionen i variant b og c.
    mal = d[(d["salgsaar"] == aar) & d["sammenlignelig"]].copy()
    print(f"Træning: {dk(len(tr))} handler ({tr['salgsaar'].min()}–{tr['salgsaar'].max()})   "
          f"Testhandler: {dk(mal['test'].sum())}   Kalibrering: {dk((~mal['test']).sum())}")

    # --- M4 ---
    m4 = pf.feols(f"ln_pris ~ {bolig_eksplicit} | kvartal + postnr", data=tr, vcov="iid")
    tr["ln_hat_ols"] = m4.predict(newdata=tr)
    mal["ln_hat_ols"] = m4.predict(newdata=mal.assign(kvartal=sidste_kv))
    print(f"M4: R² = {m4._r2:.3f}")

    # --- Vej-effekt (husets egne salg trukket fra) ---
    tr["resid"] = tr["ln_pris"] - tr["ln_hat_ols"]
    t = tr[tr["vej"].notna() & tr["resid"].notna()]
    vej_S, vej_n = t.groupby("vej")["resid"].sum(), t.groupby("vej")["resid"].size()
    hus_S = t.groupby(["vej", "bfe"])["resid"].sum()
    hus_n = t.groupby(["vej", "bfe"])["resid"].size()
    inden = t["resid"] - t.groupby("vej")["resid"].transform("mean")
    s2_inden = float((inden ** 2).sum() / (len(t) - len(vej_n)))
    mange = vej_n[vej_n >= 2]
    s2_mellem = max(float((vej_S / vej_n)[mange.index].var() - (s2_inden / mange).mean()), 1e-6)
    K = s2_inden / s2_mellem
    for x in (tr, mal):
        ix = pd.MultiIndex.from_arrays([x["vej"], x["bfe"]])
        S = x["vej"].map(vej_S).fillna(0).to_numpy() - hus_S.reindex(ix).fillna(0).to_numpy()
        n = x["vej"].map(vej_n).fillna(0).to_numpy() - hus_n.reindex(ix).fillna(0).to_numpy()
        x["vej_effekt"], x["vej_antal"] = S / (n + K), n
    print(f"Vej: K = {K:.1f}   testhandler med andet salg på vejen: "
          f"{100 * (mal.loc[mal['test'], 'vej_antal'] > 0).mean():.1f} %")

    # --- Regionalt prisindeks og tidligere salg (kun træningsdata) ---
    tr["r"] = np.log(tr["kr_pr_m2"]) - tr.groupby("postnr")["kr_pr_m2"].transform(
        lambda s: np.log(s).mean())
    indeks = tr.groupby(["region", "kvartal"])["r"].mean()
    hoejre = (tr[["bfe", "dato", "pris", "kvartal"]]
              .rename(columns={"dato": "tidl_dato", "pris": "tidl_pris", "kvartal": "tidl_kvartal"})
              .dropna(subset=["bfe"]).sort_values("tidl_dato"))
    hoejre["noegle"] = hoejre["tidl_dato"]
    for navn, x in [("tr", tr), ("mal", mal)]:
        v = x[["bfe", "dato"]].reset_index().sort_values("dato")
        m = pd.merge_asof(v, hoejre, left_on="dato", right_on="noegle", by="bfe",
                          direction="backward", allow_exact_matches=False).set_index("index")
        x[["tidl_pris", "tidl_kvartal", "tidl_dato"]] = m[["tidl_pris", "tidl_kvartal", "tidl_dato"]]
        # Træningshandler fremskrives til eget kvartal; teståret kun til 4. kvartal året før
        til_kv = x["kvartal"] if navn == "tr" else pd.Series(sidste_kv, index=x.index)
        nu = pd.MultiIndex.from_arrays([x["region"], til_kv])
        foer = pd.MultiIndex.from_arrays([x["region"], x["tidl_kvartal"]])
        x["ln_tidl_fremskrevet"] = (np.log(x["tidl_pris"]) + indeks.reindex(nu).to_numpy()
                                    - indeks.reindex(foer).to_numpy())
        slut = x["dato"] if navn == "tr" else pd.Series(skaering, index=x.index)
        x["aar_siden_salg"] = (slut - x["tidl_dato"]).dt.days / 365.25
    print(f"Testhandler med tidligere salg: {100 * mal.loc[mal['test'], 'ln_tidl_fremskrevet'].notna().mean():.1f} %")

    # --- Gradient boosting ---
    postnr_niveau = tr.groupby("postnr")["ln_pris"].mean()
    for x in (tr, mal):
        x["postnr_niveau"] = x["postnr"].map(postnr_niveau)
        x["postnr_tal"] = pd.to_numeric(x["postnr"], errors="coerce")
    mal["tid"] = (aar - 1) * 4 + 4    # prisniveau i 4. kvartal året før (tidstro)
    X = pd.get_dummies(pd.concat([tr, mal])[basis + nye + kategoriske], columns=kategoriske,
                       dtype=float)
    X_tr, X_mal = X.iloc[:len(tr)], X.iloc[len(tr):]
    gb = HistGradientBoostingRegressor(**GB_PARAMS)
    gb.fit(X_tr, tr["ln_pris"])
    mal["ln_hat_gb"] = gb.predict(X_mal)
    print(f"Gradient boosting: {gb.n_iter_} træer")

    # --- Forudsigelser i kroner (median-forudsigelse) ---
    mal["p_ols"] = np.exp(mal["ln_hat_ols"])
    mal["p_ols_vej"] = np.exp(mal["ln_hat_ols"] + mal["vej_effekt"])
    mal["p_gb"] = np.exp(mal["ln_hat_gb"])
    mal["p_tidl"] = np.exp(mal["ln_tidl_fremskrevet"])
    mal["p_vurd"] = mal["ejendomvaerdibeloeb"]
    resultater.append(mal)

alle = pd.concat(resultater)
# Forudsigelserne gemmes, så vurdering_korrigeret.py kan bruge dem (testhandler + kalibrering)
alle[["behandlingsid", "test", "salgsaar", "kvartal", "pris", "p_ols", "p_ols_vej", "p_gb",
      "p_tidl", "p_vurd"]].to_csv(MAPPE / "forudsigelser_tidstro.csv", index=False, sep=";",
                                  decimal=",", encoding="utf-8-sig")
print(f"  gemt: forudsigelser_tidstro.csv ({dk(len(alle))} handler)")
MODELLER = {"p_ols": "M4", "p_ols_vej": "M4 + vej", "p_gb": "Gradient boosting (fuld)",
            "p_tidl": "Kun tidligere salg", "p_vurd": "Offentlig vurdering"}
KERNE = ["p_ols", "p_ols_vej", "p_gb", "p_vurd"]
alle = alle.dropna(subset=KERNE)

# =============================================================================
# VARIANT b OG c: SYMMETRISK NIVEAUKORREKTION (KUN FRA KALIBRERINGSHANDLER)
# =============================================================================
VARIANTER = {"a": "a) Strengt tidstro", "b": "b) Niveau pr. år", "c": "c) Niveau pr. kvartal"}
kal = alle[~alle["test"]]
for kol in MODELLER:
    alle[f"{kol}_a"] = alle[kol]
    for v, gruppe in [("b", "salgsaar"), ("c", "kvartal")]:
        k = kal.dropna(subset=[kol])
        faktor = (k["pris"] / k[kol]).groupby(k[gruppe]).median()
        alle[f"{kol}_{v}"] = alle[kol] * alle[gruppe].map(faktor)
test = alle[alle["test"]].copy()
test["har_tidl_salg"] = test["p_tidl"].notna()

overskrift("Niveaukorrektion (medianforhold salgspris/forudsigelse i kalibreringshandler)")
for kol in ["p_gb", "p_vurd"]:
    k = kal.dropna(subset=[kol])
    print(f"{MODELLER[kol]}: pr. år " +
          ", ".join(f"{a}: {x:.3f}" for a, x in (k["pris"] / k[kol]).groupby(k["salgsaar"]).median().items()))

# =============================================================================
# TABEL 20: PRÆCISION PR. VARIANT
# =============================================================================
overskrift("Tabel 20 – tidstro præcision på testdata")
print(f"Testhandler med forudsigelse fra alle modeller: {dk(len(test))}")
raekker = []
for v, vnavn in VARIANTER.items():
    for kol in KERNE:
        raekker.append(praecision(test, f"{kol}_{v}").rename((vnavn, MODELLER[kol])))
t20 = pd.DataFrame(raekker)
t20.index = pd.MultiIndex.from_tuples(t20.index, names=["Variant", "Model"])
vis = t20.copy().astype(object)
vis["N"] = t20["N"].map(lambda x: dk(x))
vis["MAE (kr.)"] = t20["MAE (kr.)"].map(lambda x: dk(x))
for k in ["MdAPE (%)", "Median %-fejl"]:
    vis[k] = t20[k].map(lambda x: dk(x, 1))
for k in ["Inden for ±10 % (andel)", "Inden for ±20 % (andel)"]:
    vis[k] = t20[k].map(lambda x: f"{dk(100 * x, 1)} %")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(vis.to_string())
gem_tabel(t20.round(4), "tabel_20_tidstro.csv")

# =============================================================================
# TABEL 21: MdAPE PR. DELGRUPPE OG VARIANT
# =============================================================================
overskrift("Tabel 21 – MdAPE (%) pr. delgruppe")
grupper = {"Alle": test, "Huse solgt før": test[test["har_tidl_salg"]],
           "Huse ikke solgt før": test[~test["har_tidl_salg"]],
           "Salg i 2020": test[test["salgsaar"] == 2020], "Salg i 2022": test[test["salgsaar"] == 2022]}
rk = {}
for v, vnavn in VARIANTER.items():
    for gnavn, g in grupper.items():
        rk[(vnavn, gnavn)] = {"N": len(g), **{MODELLER[k]: np.median(abs_pct(g[f"{k}_{v}"], g["pris"]))
                                             for k in MODELLER if g[f"{k}_{v}"].notna().all()}}
t21 = pd.DataFrame(rk).T
vis = t21.round(1).astype(object)
vis["N"] = t21["N"].map(lambda x: dk(x))
print(vis.to_string(na_rep="–"))
gem_tabel(t21.round(3), "tabel_21_tidstro_grupper.csv")

# =============================================================================
# TABEL 22: BOOTSTRAP – ER FORSKELLEN MELLEM MODEL OG VURDERING SIKKER?
# =============================================================================
overskrift(f"Tabel 22 – bootstrap ({B} gentrækninger af testhandlerne)")
bs = {}
rk = []
for v, vnavn in VARIANTER.items():
    for gnavn in ["Alle", "Huse solgt før", "Huse ikke solgt før"]:
        g = grupper[gnavn]
        res = bootstrap_mdape({k: abs_pct(g[f"{k}_{v}"], g["pris"]).to_numpy()
                               for k in ["p_gb", "p_vurd"]})
        bs[(v, gnavn)] = res
        diff = res["p_gb"] - res["p_vurd"]
        rk.append(pd.Series({
            "MdAPE model": np.median(abs_pct(g[f"p_gb_{v}"], g["pris"])),
            "MdAPE vurdering": np.median(abs_pct(g[f"p_vurd_{v}"], g["pris"])),
            "Forskel (model − vurdering), pp": np.median(abs_pct(g[f"p_gb_{v}"], g["pris"]))
            - np.median(abs_pct(g[f"p_vurd_{v}"], g["pris"])),
            "95 %-KI nedre": diff.quantile(0.025), "95 %-KI øvre": diff.quantile(0.975),
            "Andel gentrækninger hvor model er bedst": (diff < 0).mean(),
        }, name=(vnavn, gnavn)))
t22 = pd.DataFrame(rk)
t22.index = pd.MultiIndex.from_tuples(t22.index, names=["Variant", "Testhandler"])
print(t22.round(2).to_string())
print("\nLæsning: negativ forskel = modellen har mindre fejl end vurderingen. Indeholder "
      "95 %-intervallet 0, er forskellen ikke statistisk sikker.")
gem_tabel(t22.round(4), "tabel_22_bootstrap.csv")

# =============================================================================
# FIGUR 15: MdAPE MED 95 %-KI – MODEL VS. VURDERING I DE TRE VARIANTER
# =============================================================================
overskrift("Figur")
fig, akser = plt.subplots(1, 3, figsize=(16 / 2.54, 7.5 / 2.54), sharex=True, sharey=True)
gnavne = ["Alle", "Huse solgt før", "Huse ikke solgt før"]
for ax, (v, vnavn) in zip(akser, VARIANTER.items()):
    for j, (kol, farve, navn) in enumerate([("p_gb", BLAA, "Gradient boosting (fuld)"),
                                            ("p_vurd", ORANGE, "Offentlig vurdering")]):
        for i, gnavn in enumerate(gnavne):
            res = bs[(v, gnavn)][kol]
            m = np.median(abs_pct(grupper[gnavn][f"{kol}_{v}"], grupper[gnavn]["pris"]))
            lo, hi = res.quantile(0.025), res.quantile(0.975)
            ax.errorbar(m, i + (j - 0.5) * 0.3, xerr=[[m - lo], [hi - m]], fmt="o",
                        color=farve, markersize=4, capsize=2, elinewidth=1.2,
                        label=navn if i == 0 else None)
    ax.set_title(vnavn, loc="left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
akser[0].set_yticks(range(len(gnavne)), gnavne, fontsize=8)
akser[0].invert_yaxis()
akser[1].set_xlabel("MdAPE på testdata (95 %-bootstrap-KI) – lavere er bedre")
fig.legend(*akser[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# fig.suptitle("Tidstro sammenligning: model trænet kun på salg før vurderingsdatoen",
# x=0.0, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(UD / "fig_15_tidstro.png")
plt.close(fig)
print("  gemt: fig_15_tidstro.png")
print(f"\nAlle filer ligger i: {UD}")
