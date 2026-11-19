"""
praediktion_geo.py – Prædiktion med koordinater og naboernes prisniveau (udvidelse af 5.13–5.16)

Bygger videre på det tidstro design i praediktion_tidstro.py og ændrer ingen eksisterende
resultater. Samme testhandler (salg i 2020 og 2022 med vurdering fra salgsåret), én model pr.
testår, og træning KUN på salg før vurderingsdatoen 1. januar i teståret:

  Testhandler i 2020: træning på salg 2015–2019
  Testhandler i 2022: træning på salg 2015–2021

Prædiktionstrappen (trin 1–3 som i praediktion_tidstro.py, trin 4–5 er nye)
  1. M4 (hedonisk model)
  2. M4 + prisniveau på vejen
  3. Gradient boosting (fuld): + vej, tidligere salg, postnummerniveau
  4. + koordinater: x, y, to drejede koordinater (45°) og ln(kystafstand)
     Træer deler kun langs akserne. De drejede koordinater lader dem også dele på skrå.
  5. + naboernes prisniveau: de nærmeste tidligere salg (målt i luftlinje)
       nabo_resid_10 / _30  gennemsnitligt M4-residual for de 10 / 30 nærmeste salg
                            (hvor meget dyrere end modellen forudsiger, nabohusene er solgt)
       nabo_krm2_10         gennemsnitlig kr./m² for de 10 nærmeste, fremskrevet med regionalt
                            prisindeks til samme tidspunkt
       nabo_afstand_10      afstanden til den 10. nærmeste (hvor tæt naboerne ligger)
     Husets egne salg (samme BFE) tælles aldrig med, og naboerne findes KUN blandt salg før
     vurderingsdatoen. Tæthedsindekset bruges ikke, fordi det tæller bygninger, der først
     handles senere (ville give et lille informationsforspring).

Kombination med vurderingen (som K1 og K2 i vurdering_korrigeret.py, men med trin 5-modellen)
  K1  ln P ~ ln V + ln P̂ | kvartal                         (vægtene estimeres på kalibrering)
  K2  ln P ~ ln V + ln P̂ + boligkarakteristika | kvartal + postnr  ("lært kombination")
  V1  vurderingen niveaukorrigeret pr. kvartal (sammenligningsgrundlag)

Varianter og usikkerhed som i praediktion_tidstro.py: a) strengt tidstro, b) niveau pr. år,
c) niveau pr. kvartal, og bootstrap (1.000) af forskellen i MdAPE.

Kontrol: trin 1–3 skal give de samme tal som tabel 20 (praediktion_tidstro.py). Det tjekkes og
udskrives.

Output i figurer_og_tabeller:
  tabel_33_praediktion_geo.csv        præcision pr. variant og model (trin 1–5 og vurdering)
  tabel_34_praediktion_geo_grupper.csv MdAPE pr. delgruppe
  tabel_35_praediktion_geo_bootstrap.csv bootstrap: trin 5 vs. vurdering og trin 5 vs. trin 3
  tabel_36_kombination_geo.csv        K1/K2 med trin 5-modellen mod V1
  fig_20_praediktionstrappe_geo.png   prædiktionstrappen trin 1–5 (tidstro, niveau pr. år)
og forudsigelser_geo.csv i data.

Kræver: data_renset.csv og geografi_kolonner.csv (laves af geografi.py) i bachelorprojekt/data.
faelles.py i samme mappe. Pakker: pyfixest, scikit-learn. Tager ca. 15–30 minutter.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KDTree

from faelles import BLAA, INK, MUTED, REF, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER (samme som praediktion_tidstro.py)
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"      # bachelorprojekt/data
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
TEST_AAR = [2020, 2022]
TEST_ANDEL = 0.5
SEED = 42
B = 1000
ORANGE, GROEN = "#eb6834", "#2e8b57"
GB_PARAMS = dict(learning_rate=0.1, max_iter=3000, max_leaf_nodes=63, min_samples_leaf=40,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=50, random_state=SEED)
K_NABO = [10, 30]               # antal nærmeste naboer
EKSTRA = 10                     # ekstra naboer i opslaget, så husets egne salg kan sorteres fra


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
    rng = np.random.default_rng(seed)
    n = len(next(iter(fejl_dict.values())))
    ud = {k: [] for k in fejl_dict}
    for start in range(0, B, blok):
        idx = rng.integers(0, n, size=(min(blok, B - start), n), dtype=np.int32)
        for k, f in fejl_dict.items():
            ud[k].append(np.median(f[idx], axis=1))
    return pd.DataFrame({k: np.concatenate(v) for k, v in ud.items()})


# =============================================================================
# DATA – SAMME TESTHANDLER SOM praediktion_tidstro.py
# =============================================================================
df19, _ = forbered_data(MAPPE)
samml19 = (df19["salgsaar"].isin(TEST_AAR) & (df19["vurd_matcher_salgsaar"] == 1)
           & (df19["ejendomvaerdibeloeb"] > 0))
rng = np.random.default_rng(SEED)
test_id = set(df19.loc[samml19 & (rng.random(len(df19)) < TEST_ANDEL), "behandlingsid"])

d, bolig = forbered_data(MAPPE, fra_aar=2015)
d["test"] = d["behandlingsid"].isin(test_id)
d["dato"] = pd.to_datetime(d["koebsaftaledato"])
d["bfe"] = d["bestemtfastejendombfenr"]
d["vej"] = (d["postnr"] + "_" + d["vejnavn"].astype(str).str.strip().str.lower()).where(
    d["vejnavn"].notna())
d["tid"] = d["kvartal"].str[:4].astype(int) * 4 + d["kvartal"].str[-1].astype(int)
d["sammenlignelig"] = ((d["vurd_matcher_salgsaar"] == 1) & (d["ejendomvaerdibeloeb"] > 0))

# Geografi (fra geografi.py)
GEO = MAPPE / "geografi_kolonner.csv"
if not GEO.exists():
    raise SystemExit("geografi_kolonner.csv mangler – kør geografi.py først.")
geo = pd.read_csv(GEO, sep=";", decimal=",", encoding="utf-8-sig")
foer = len(d)
d = d.merge(geo[["behandlingsid", "bygning_x", "bygning_y", "kystafstand_m"]],
            on="behandlingsid", how="left", validate="one_to_one")
assert len(d) == foer
d["rot1"] = (d["bygning_x"] + d["bygning_y"]) / np.sqrt(2)
d["rot2"] = (d["bygning_x"] - d["bygning_y"]) / np.sqrt(2)
d["ln_kyst"] = np.log(d["kystafstand_m"].clip(lower=1))
d["ln_krm2"] = np.log(d["kr_pr_m2"])
d["har_koord"] = d["bygning_x"].notna() & d["bygning_y"].notna()

# Eksplicitte dummies til M4 (så pyfixest kan forudsige på nye data)
dummynavne = {}
for col, ref in REF.items():
    niveauer = [x for x in sorted(d[col].unique()) if x != ref]
    for k, niv in enumerate(niveauer):
        d[f"d_{col}_{k}"] = (d[col] == niv).astype(int)
    dummynavne[col] = [f"d_{col}_{k}" for k in range(len(niveauer))]
bolig_eksplicit = bolig
for col, navne in dummynavne.items():
    bolig_eksplicit = re.sub(rf"i\({col}, ref='[^']*'\)", " + ".join(navne), bolig_eksplicit)

BASIS = ["ln_boligareal", "ln_grundareal", "raekkehus", "vaerelser", "to_etager_plus",
         "ombygget", "har_garage", "har_carport", "har_udhus", "har_udestue",
         "postnr_niveau", "postnr_tal", "tid"]
KATEGORISKE = ["badevaerelser_kat", "opfoerelsesperiode", "energimaerke", "opvarmning",
               "ydervaeg", "tag", "region"]
NYE = ["vej_effekt", "vej_antal", "ln_tidl_fremskrevet", "aar_siden_salg"]
GEO_VAR = ["bygning_x", "bygning_y", "rot1", "rot2", "ln_kyst"]
NABO_VAR = ["nabo_resid_10", "nabo_resid_30", "nabo_krm2_10", "nabo_afstand_10"]
TRIN = {"p_gb": BASIS + NYE, "p_geo": BASIS + NYE + GEO_VAR,
        "p_nabo": BASIS + NYE + GEO_VAR + NABO_VAR}

overskrift("Testhandler og koordinater")
print(f"Testhandler fundet: {dk(d['test'].sum())} af {dk(len(test_id))}")
print(f"Handler med koordinat: {100 * d['har_koord'].mean():.2f} %")


def naboer(tr, x, er_tr, indeks, til_kv):
    """Naboernes prisniveau for husene i x, beregnet ud fra træningssalgene i tr.
    Husets egne salg (samme BFE) og – for træningshandler – handlen selv sorteres fra."""
    kilde = tr[tr["har_koord"] & tr["resid"].notna()]
    tree = KDTree(kilde[["bygning_x", "bygning_y"]].to_numpy())
    resid = kilde["resid"].to_numpy()
    bfe_k = kilde["bfe"].to_numpy()
    # kr./m² deflateret med det regionale indeks i salgskvartalet (fælles basis)
    defl = (kilde["ln_krm2"].to_numpy()
            - indeks.reindex(pd.MultiIndex.from_arrays([kilde["region"], kilde["kvartal"]])).to_numpy())
    ud = pd.DataFrame(np.nan, index=x.index, columns=NABO_VAR)
    q = x[x["har_koord"]]
    afst, idx = tree.query(q[["bygning_x", "bygning_y"]].to_numpy(), k=max(K_NABO) + EKSTRA)
    gyldig = bfe_k[idx] != q["bfe"].to_numpy()[:, None]
    if er_tr:                                   # handlen selv (hvis BFE mangler)
        egen = kilde.index.get_indexer(q.index)
        gyldig &= idx != egen[:, None]
    gyldig &= ~np.isnan(defl[idx])
    rang = np.cumsum(gyldig, axis=1)
    for k in K_NABO:
        brug = gyldig & (rang <= k)
        n = brug.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ud.loc[q.index, f"nabo_resid_{k}"] = (np.where(brug, resid[idx], 0).sum(axis=1) / n)
            if k == 10:
                maal_idx = indeks.reindex(pd.MultiIndex.from_arrays(
                    [q["region"], til_kv.loc[q.index]])).to_numpy()
                ud.loc[q.index, "nabo_krm2_10"] = (np.where(brug, defl[idx], 0).sum(axis=1) / n
                                                   + maal_idx)
                ud.loc[q.index, "nabo_afstand_10"] = np.log1p(np.where(brug, afst, 0).max(axis=1))
    return ud


# =============================================================================
# ÉN MODEL PR. TESTÅR, TRÆNET KUN PÅ SALG FØR 1. JANUAR I TESTÅRET
# =============================================================================
resultater = []
for aar in TEST_AAR:
    overskrift(f"Testår {aar}: træning på salg før 1/1-{aar}")
    skaering = pd.Timestamp(aar, 1, 1)
    sidste_kv = f"{aar - 1}Q4"
    tr = d[d["dato"] < skaering].copy()
    mal = d[(d["salgsaar"] == aar) & d["sammenlignelig"]].copy()
    print(f"Træning: {dk(len(tr))} handler   Testhandler: {dk(mal['test'].sum())}   "
          f"Kalibrering: {dk((~mal['test']).sum())}")

    # --- Trin 1: M4 ---
    m4 = pf.feols(f"ln_pris ~ {bolig_eksplicit} | kvartal + postnr", data=tr, vcov="iid")
    tr["ln_hat_ols"] = m4.predict(newdata=tr)
    mal["ln_hat_ols"] = m4.predict(newdata=mal.assign(kvartal=sidste_kv))

    # --- Trin 2: vej-effekt (som praediktion_tidstro.py) ---
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

    # --- Regionalt prisindeks og tidligere salg (som praediktion_tidstro.py) ---
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
        til_kv = x["kvartal"] if navn == "tr" else pd.Series(sidste_kv, index=x.index)
        nu = pd.MultiIndex.from_arrays([x["region"], til_kv])
        foer_ix = pd.MultiIndex.from_arrays([x["region"], x["tidl_kvartal"]])
        x["ln_tidl_fremskrevet"] = (np.log(x["tidl_pris"]) + indeks.reindex(nu).to_numpy()
                                    - indeks.reindex(foer_ix).to_numpy())
        slut = x["dato"] if navn == "tr" else pd.Series(skaering, index=x.index)
        x["aar_siden_salg"] = (slut - x["tidl_dato"]).dt.days / 365.25

    # --- Trin 5: naboernes prisniveau (kun træningssalg, aldrig husets egne salg) ---
    tr[NABO_VAR] = naboer(tr, tr, True, indeks, tr["kvartal"])
    mal[NABO_VAR] = naboer(tr, mal, False, indeks, pd.Series(sidste_kv, index=mal.index))
    mt = mal[mal["test"]]
    print(f"Naboer: median afstand til 10. nærmeste for testhandler = "
          f"{dk(np.expm1(mt['nabo_afstand_10']).median())} m   "
          f"testhandler med naboniveau: {100 * mt['nabo_resid_10'].notna().mean():.1f} %")
    print(f"Korrelation mellem vej-effekt og nabo-residual (træning, vej_antal > 0): "
          f"{tr.loc[tr['vej_antal'] > 0, ['vej_effekt', 'nabo_resid_10']].corr().iat[0, 1]:.2f}")

    # --- Trin 3–5: gradient boosting ---
    postnr_niveau = tr.groupby("postnr")["ln_pris"].mean()
    for x in (tr, mal):
        x["postnr_niveau"] = x["postnr"].map(postnr_niveau)
        x["postnr_tal"] = pd.to_numeric(x["postnr"], errors="coerce")
    mal["tid"] = (aar - 1) * 4 + 4    # prisniveau i 4. kvartal året før (tidstro)
    for kol, var in TRIN.items():
        X = pd.get_dummies(pd.concat([tr, mal])[var + KATEGORISKE], columns=KATEGORISKE,
                           dtype=float)
        X_tr, X_mal = X.iloc[:len(tr)], X.iloc[len(tr):]
        gb = HistGradientBoostingRegressor(**GB_PARAMS)
        gb.fit(X_tr, tr["ln_pris"])
        mal[kol] = np.exp(gb.predict(X_mal))
        print(f"{kol}: {gb.n_iter_} træer")

    mal["p_ols"] = np.exp(mal["ln_hat_ols"])
    mal["p_ols_vej"] = np.exp(mal["ln_hat_ols"] + mal["vej_effekt"])
    mal["p_tidl"] = np.exp(mal["ln_tidl_fremskrevet"])
    mal["p_vurd"] = mal["ejendomvaerdibeloeb"]
    resultater.append(mal)

alle = pd.concat(resultater)
alle[["behandlingsid", "test", "salgsaar", "kvartal", "pris", "p_ols", "p_ols_vej", "p_gb",
      "p_geo", "p_nabo", "p_tidl", "p_vurd"]].to_csv(
    MAPPE / "forudsigelser_geo.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
print(f"  gemt: forudsigelser_geo.csv ({dk(len(alle))} handler)")

MODELLER = {"p_ols": "1) M4", "p_ols_vej": "2) M4 + vej", "p_gb": "3) Gradient boosting (fuld)",
            "p_geo": "4) + koordinater og kystafstand", "p_nabo": "5) + naboernes prisniveau",
            "p_vurd": "Offentlig vurdering"}
alle = alle.dropna(subset=list(MODELLER))

# =============================================================================
# VARIANT b OG c: SYMMETRISK NIVEAUKORREKTION (KUN FRA KALIBRERINGSHANDLER)
# =============================================================================
VARIANTER = {"a": "a) Strengt tidstro", "b": "b) Niveau pr. år", "c": "c) Niveau pr. kvartal"}
kal = alle[~alle["test"]]
for kol in MODELLER:
    alle[f"{kol}_a"] = alle[kol]
    for v, gruppe in [("b", "salgsaar"), ("c", "kvartal")]:
        faktor = (kal["pris"] / kal[kol]).groupby(kal[gruppe]).median()
        alle[f"{kol}_{v}"] = alle[kol] * alle[gruppe].map(faktor)
test = alle[alle["test"]].copy()
test["har_tidl_salg"] = test["p_tidl"].notna()

# =============================================================================
# TABEL 33: PRÆCISION PR. VARIANT
# =============================================================================
overskrift("Tabel 33 – tidstro præcision, trin 1–5 og vurdering")
print(f"Testhandler med forudsigelse fra alle modeller: {dk(len(test))}")
raekker = []
for v, vnavn in VARIANTER.items():
    for kol in MODELLER:
        raekker.append(praecision(test, f"{kol}_{v}").rename((vnavn, MODELLER[kol])))
t33 = pd.DataFrame(raekker)
t33.index = pd.MultiIndex.from_tuples(t33.index, names=["Variant", "Model"])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(t33[["N", "MdAPE (%)", "Median %-fejl", "Inden for ±10 % (andel)",
           "Inden for ±20 % (andel)"]].round(3).to_string())
gem_tabel(t33.round(4), "tabel_33_praediktion_geo.csv")

# Kontrol mod tabel 20 (praediktion_tidstro.py): trin 1–3 og vurderingen skal være ens
try:
    t20 = pd.read_csv(UD / "tabel_20_tidstro.csv", sep=";", decimal=",", encoding="utf-8-sig",
                      index_col=[0, 1])
    navne20 = {"p_ols": "M4", "p_ols_vej": "M4 + vej", "p_gb": "Gradient boosting (fuld)",
               "p_vurd": "Offentlig vurdering"}
    print("\nKontrol mod tabel 20 (MdAPE %, variant b) – bør være ens:")
    for kol, n20 in navne20.items():
        her = t33.loc[(VARIANTER["b"], MODELLER[kol]), "MdAPE (%)"]
        der = t20.loc[(VARIANTER["b"], n20), "MdAPE (%)"]
        print(f"  {n20:28s} her {her:6.3f}   tabel 20 {der:6.3f}")
    print("  (Små afvigelser kan skyldes, at enkelte testhandler uden koordinat er udeladt her.)")
except Exception as fejl:
    print(f"Kunne ikke sammenligne med tabel 20: {fejl}")

# =============================================================================
# TABEL 34: MdAPE PR. DELGRUPPE
# =============================================================================
overskrift("Tabel 34 – MdAPE (%) pr. delgruppe")
grupper = {"Alle": test, "Huse solgt før": test[test["har_tidl_salg"]],
           "Huse ikke solgt før": test[~test["har_tidl_salg"]],
           "Salg i 2020": test[test["salgsaar"] == 2020], "Salg i 2022": test[test["salgsaar"] == 2022]}
rk = {}
for v, vnavn in VARIANTER.items():
    for gnavn, g in grupper.items():
        rk[(vnavn, gnavn)] = {"N": len(g), **{MODELLER[k]: np.median(abs_pct(g[f"{k}_{v}"], g["pris"]))
                                             for k in MODELLER}}
t34 = pd.DataFrame(rk).T
print(t34.round(2).to_string())
gem_tabel(t34.round(3), "tabel_34_praediktion_geo_grupper.csv")

# =============================================================================
# TABEL 35: BOOTSTRAP – TRIN 5 MOD VURDERINGEN OG MOD TRIN 3
# =============================================================================
overskrift(f"Tabel 35 – bootstrap ({B} gentrækninger af testhandlerne)")
rk = []
bs_alle = {}
for v, vnavn in VARIANTER.items():
    for gnavn in ["Alle", "Huse solgt før", "Huse ikke solgt før"]:
        g = grupper[gnavn]
        res = bootstrap_mdape({k: abs_pct(g[f"{k}_{v}"], g["pris"]).to_numpy()
                               for k in ["p_gb", "p_nabo", "p_vurd"]})
        bs_alle[(v, gnavn)] = res
        for mod, navn_mod in [("p_vurd", "vurdering"), ("p_gb", "trin 3")]:
            diff = res["p_nabo"] - res[mod]
            m5 = np.median(abs_pct(g[f"p_nabo_{v}"], g["pris"]))
            mm = np.median(abs_pct(g[f"{mod}_{v}"], g["pris"]))
            rk.append(pd.Series({
                "MdAPE trin 5": m5, f"MdAPE sammenligning": mm,
                "Forskel (trin 5 − sammenligning), pp": m5 - mm,
                "95 %-KI nedre": diff.quantile(0.025), "95 %-KI øvre": diff.quantile(0.975),
                "Andel gentrækninger hvor trin 5 er bedst": (diff < 0).mean(),
            }, name=(vnavn, gnavn, f"mod {navn_mod}")))
t35 = pd.DataFrame(rk)
t35.index = pd.MultiIndex.from_tuples(t35.index, names=["Variant", "Testhandler", "Sammenligning"])
print(t35.round(2).to_string())
print("\nLæsning: negativ forskel = trin 5 har mindre fejl. Indeholder 95 %-intervallet 0, er "
      "forskellen ikke statistisk sikker.")
gem_tabel(t35.round(4), "tabel_35_praediktion_geo_bootstrap.csv")

# =============================================================================
# TABEL 36: KOMBINATION MED VURDERINGEN (K1 OG K2 MED TRIN 5-MODELLEN)
# =============================================================================
overskrift("Tabel 36 – kombination af vurdering og trin 5-modellen")
dd = alle.copy()
dd["ln_vurd"] = np.log(dd["p_vurd"])
dd["ln_model"] = np.log(dd["p_nabo"])
kal, tst = dd[~dd["test"]].copy(), dd[dd["test"]].copy()
faktor = (kal["pris"] / kal["p_vurd"]).groupby(kal["kvartal"]).median()
tst["V1"] = tst["p_vurd"] * tst["kvartal"].map(faktor)
faktor5 = (kal["pris"] / kal["p_nabo"]).groupby(kal["kvartal"]).median()
tst["M5"] = tst["p_nabo"] * tst["kvartal"].map(faktor5)
m_k1 = pf.feols("ln_pris ~ ln_vurd + ln_model | kvartal", data=kal, vcov={"CRV1": "postnr"})
tst["K1"] = np.exp(m_k1.predict(newdata=tst))
tk = m_k1.tidy()
print(f"K1: vægt på ln(vurdering) = {tk.loc['ln_vurd', 'Estimate']:.3f} "
      f"(std.fejl {tk.loc['ln_vurd', 'Std. Error']:.3f}), på ln(model) = "
      f"{tk.loc['ln_model', 'Estimate']:.3f} (std.fejl {tk.loc['ln_model', 'Std. Error']:.3f})")
m_k2 = pf.feols(f"ln_pris ~ ln_vurd + ln_model + {bolig_eksplicit} | kvartal + postnr", data=kal,
                vcov={"CRV1": "postnr"})
tst["K2"] = np.exp(m_k2.predict(newdata=tst))
tk2 = m_k2.tidy()
print(f"K2: vægt på ln(vurdering) = {tk2.loc['ln_vurd', 'Estimate']:.3f}, "
      f"på ln(model) = {tk2.loc['ln_model', 'Estimate']:.3f}")
KOMB = {"V1": "V1 Vurdering, niveau pr. kvartal", "M5": "Trin 5-model, niveau pr. kvartal",
        "K1": "K1 Kombination vurdering + trin 5", "K2": "K2 Kombination + boligkarakteristika"}
tst = tst.dropna(subset=list(KOMB))
res = bootstrap_mdape({k: abs_pct(tst[k], tst["pris"]).to_numpy() for k in KOMB})
rk = []
for k, navn in KOMB.items():
    s = praecision(tst, k)
    diff = res[k] - res["V1"]
    s["Forskel til V1, pp"] = s["MdAPE (%)"] - np.median(abs_pct(tst["V1"], tst["pris"]))
    s["95 %-KI nedre"], s["95 %-KI øvre"] = diff.quantile(0.025), diff.quantile(0.975)
    rk.append(s.rename(navn))
t36 = pd.DataFrame(rk)
print(t36[["N", "MdAPE (%)", "Median %-fejl", "Forskel til V1, pp", "95 %-KI nedre",
           "95 %-KI øvre"]].round(3).to_string())
print("NB: K1 og K2 bruger kalibreringshandler fra salgsåret – samme informationsgrundlag som "
      "variant c (niveau pr. kvartal), ikke strengt tidstro.")
gem_tabel(t36.round(4), "tabel_36_kombination_geo.csv")

# =============================================================================
# FIGUR 20: PRÆDIKTIONSTRAPPEN TRIN 1–5 (TIDSTRO, NIVEAU PR. ÅR) – UDEN TITEL
# =============================================================================
overskrift("Figur 20")
v = "b"
koler = ["p_ols", "p_ols_vej", "p_gb", "p_geo", "p_nabo", "p_vurd"]
etiketter = ["1) Hedonisk model (M4)", "2) + prisniveau på vejen",
             "3) Gradient boosting\n+ vej og tidligere salg", "4) + koordinater og kystafstand",
             "5) + naboernes prisniveau", "Offentlig vurdering\n(niveaukorrigeret)"]
vaerdier = [t33.loc[(VARIANTER[v], MODELLER[k]), "MdAPE (%)"] for k in koler]
fig, ax = plt.subplots(figsize=(16 / 2.54, 7 / 2.54))
yy = np.arange(len(koler))
farver = [BLAA] * 5 + [ORANGE]
ax.barh(yy, vaerdier, height=0.55, color=farver)
for y, val in zip(yy, vaerdier):
    # Tallet står til højre for den stiplede vurderingslinje, når søjlen slutter tæt på eller
    # før linjen, så linjen ikke går gennem tallet (rettet 19/11)
    xt = max(val, vaerdier[-1]) + 0.15
    ax.text(xt, y, dk(val, 1), va="center", fontsize=7.5, color=INK, zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.8))
ax.axvline(vaerdier[-1], color=ORANGE, lw=0.8, ls="--", zorder=1)
ax.set_yticks(yy, etiketter, fontsize=7.5)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(vaerdier) * 1.12)
ax.xaxis.set_major_locator(MultipleLocator(2))              # hele procenter på aksen
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_xlabel("Median absolut fejl i procent af salgsprisen (MdAPE), tidstro, niveau pr. år")
fig.savefig(UD / "fig_20_praediktionstrappe_geo.png")
plt.close(fig)
print("  gemt: fig_20_praediktionstrappe_geo.png")
print(f"\nAlle filer ligger i: {UD}")
