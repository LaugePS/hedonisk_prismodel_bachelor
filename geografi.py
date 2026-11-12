"""
geografi.py – Kystafstand, bebyggelsestæthed og sæson (udvidelse af analyse 5.1)

Bygger på hovedmodellen (M4 i hovedmodel.py) og ændrer IKKE stikprøven: de nye kolonner fra
dataudtrækket 27/10-2026 (koordinater og kystafstand) flettes på det rensede datasæt via
behandlingsid. Alle handler i data_renset.csv er handler med én opført bygning (valg B), så
nøglen er entydig.

Analyser
  1. Kystpræmie: M4 + afstandsintervaller til kysten (0–250 m, 250 m–1 km, 1–5 km, ref. over 5 km)
     – uden postnummer-FE (sammenligning på tværs af landet) og med postnummer-FE (inden for samme
     postnummer). Alternativ: ln(kystafstand). Variant med bebyggelsestæthed.
  2. Omitted variable bias-tjek: ændrer koefficienterne for energimærke, opførelsesperiode m.m. sig,
     når kystafstanden kommer med?
  3. Hvad består beliggenhedseffekten af? De estimerede postnummer-effekter fra M4 regresseres på
     postnummerets mediane kystafstand, tæthed og region.
  4. Tid: M4 med faste effekter for salgsmåned (år×måned) i stedet for kvartal, og sæsonmønstret i et
     hedonisk (kvalitetskorrigeret) månedsindeks.

Estimation
  Scriptet bruger kun numpy/pandas. Faste effekter fjernes med within-transformationen: y og alle
  forklarende variable trækkes fra deres gennemsnit inden for postnummer og inden for tidsperiode,
  gentaget til konvergens (Frisch–Waugh–Lovell). Det giver præcis de samme koefficienter som OLS med
  dummies – og som pyfixest i hovedmodel.py. Scriptet tjekker det selv ved at genestimere M4 og
  sammenligne med tabel_5_regression.csv. Postnumre med kun én handel fjernes (som i pyfixest).
  Standardfejl: klyngerobuste på postnummer (CRV1).

Tæthed
  Tæthedsindeks = antal opførte bygninger (alle anvendelser) med koordinat, som indgår i en handel
  2015–2026, inden for et kvadrat på 750 × 750 m omkring huset, omregnet til pr. km². Det er et
  indeks og ikke den faktiske bebyggelsestæthed, fordi bygninger, der ikke er handlet, ikke tælles.

Output (figurer_og_tabeller)
  tabel_27_kystpraemie.csv          kyst- og tæthedskoefficienter for modellerne G1–G4 (G = geografi; K1/K2 bruges til kombinationerne med vurderingen)
  tabel_28_kyst_deskriptiv.csv      handler og nøgletal pr. afstandsinterval
  tabel_29_ovb_kyst.csv             udvalgte koefficienter i M4 med og uden kystafstand
  tabel_30_postnr_dekomposition.csv postnummer-effekter forklaret af kyst, tæthed og region
  tabel_31_maaned_fe.csv            M4 med kvartals- vs. måneds-FE
  tabel_32_saeson.csv               sæsonmønster pr. kalendermåned
  fig_18_kystpraemie.png            kystpræmie med og uden postnummer-FE
  fig_19_saeson.png                 sæson i pris (hedonisk) og i antal handler
Data
  data/geografi_kolonner.csv        koordinater, kystafstand og tæthed pr. handel (genbruges)

Kræver kun numpy, pandas og matplotlib. faelles.py skal ligge i samme mappe.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from faelles import BLAA, INK2, MUTED, dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"      # bachelorprojekt/data
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
RAA = "ejendomshandler_energi (4).csv"                         # udtræk 27/10-2026 med koordinater
GEO = MAPPE / "geografi_kolonner.csv"

KYST_GRAENSER = [0, 250, 1000, 5000, np.inf]
KYST_NAVNE = ["0–250 m", "250 m–1 km", "1–5 km", "Over 5 km"]
KYST_REF = "Over 5 km"
CELLE = 250                     # gitterstørrelse i meter til tæthedsindekset (3 × 3 celler = 750 m)
MIN_HANDLER_POSTNR = 30         # postnumre med færre handler indgår ikke i dekompositionen
MAANEDER = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


# =============================================================================
# 1. GEOGRAFISKE KOLONNER (laves én gang og gemmes)
# =============================================================================
def lav_geografi():
    overskrift("Geografiske kolonner fra udtrækket 27/10-2026")
    ny = pd.read_csv(MAPPE / RAA, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False,
                     usecols=["behandlingsid", "bygning_id", "bygning_status", "bygning_x",
                              "bygning_y", "koordinat_kilde", "kystafstand_m"])
    ny = ny[ny["bygning_status"] == 6]          # kun opførte bygninger (som i rensningen)
    print(f"Rækker med opført bygning: {dk(len(ny))}")

    # Tæthedsindeks: tæl hver bygning én gang i et 250 m-gitter og læg de 3 × 3 naboceller sammen
    byg = ny.dropna(subset=["bygning_x", "bygning_y"]).drop_duplicates("bygning_id")
    cx = np.floor(byg["bygning_x"].to_numpy() / CELLE).astype(np.int64)
    cy = np.floor(byg["bygning_y"].to_numpy() / CELLE).astype(np.int64)
    taelling = pd.Series(1, index=cx * 100_000 + cy).groupby(level=0).sum()
    print(f"Unikke bygninger med koordinat: {dk(len(byg))}, besatte celler: {dk(len(taelling))}")

    # Kun handler med én opført bygning (valg B) – så er behandlingsid en entydig nøgle
    n = ny.groupby("behandlingsid")["bygning_id"].transform("size")
    en = ny[n == 1].copy()
    ex = np.floor(en["bygning_x"].to_numpy() / CELLE)
    ey = np.floor(en["bygning_y"].to_numpy() / CELLE)
    nabo = np.zeros(len(en))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            noegle = (ex + dx) * 100_000 + (ey + dy)
            nabo += taelling.reindex(noegle).fillna(0).to_numpy()
    areal_km2 = (3 * CELLE / 1000) ** 2
    en["taethed_km2"] = np.where(en["bygning_x"].notna(), nabo / areal_km2, np.nan)
    ud = en[["behandlingsid", "bygning_x", "bygning_y", "koordinat_kilde", "kystafstand_m",
             "taethed_km2"]]
    ud.to_csv(GEO, sep=";", decimal=",", encoding="utf-8-sig", index=False)
    print(f"Gemt: {GEO.name} ({dk(len(ud))} handler)")


if not GEO.exists():
    lav_geografi()

# =============================================================================
# 2. DATA: hovedmodellens stikprøve + geografi + måned
# =============================================================================
df, bolig = forbered_data(MAPPE)
geo = pd.read_csv(GEO, sep=";", decimal=",", encoding="utf-8-sig")
foer = len(df)
df = df.merge(geo, on="behandlingsid", how="left", validate="one_to_one")
assert len(df) == foer
dato = pd.to_datetime(df["koebsaftaledato"])
df["maaned"] = dato.dt.month
df["aarmaaned"] = dato.dt.to_period("M").astype(str)
df["kyst_km"] = df["kystafstand_m"] / 1000
df["ln_kyst_km"] = np.log(df["kyst_km"])
df["ln_taethed"] = np.log(df["taethed_km2"])
df["kystinterval"] = pd.cut(df["kystafstand_m"], KYST_GRAENSER, labels=KYST_NAVNE, right=False)
overskrift("Fletning")
print(f"Handler i hovedmodellens stikprøve: {dk(len(df))}")
print(f"Med kystafstand: {dk(df['kystafstand_m'].notna().sum())} "
      f"({100 * df['kystafstand_m'].notna().mean():.2f} %)")
print("Koordinatkilde:\n" + df["koordinat_kilde"].value_counts(dropna=False).to_string())

BOLIG_LED = [led.strip() for led in bolig.split(" + ")]


# =============================================================================
# ESTIMATOR: OLS med faste effekter via within-transformation
# =============================================================================
def design(data, led):
    """Bygger X ud fra formel-led. i(var, ref='x') bliver til dummies med navnene 'var::niveau'
    (samme navne som pyfixest), øvrige led bruges som de er."""
    kol = {}
    for l in led:
        m = re.fullmatch(r"i\((\w+), ref='([^']*)'\)", l)
        if m:
            var, ref = m.groups()
            v = data[var].astype(str)
            for niv in sorted(v.unique()):
                if niv != ref and niv != "nan":
                    kol[f"{var}::{niv}"] = (v == niv).astype(float).to_numpy()
        else:
            kol[l] = data[l].astype(float).to_numpy()
    return pd.DataFrame(kol, index=data.index)


def demean(M, grupper, tol=1e-10, maxit=1000):
    """Trækker gruppegennemsnit fra (skiftevis for hver gruppering) til alle gennemsnit er ~0."""
    M = M.copy()
    antal = [np.bincount(g) for g in grupper]
    for it in range(maxit):
        stoerst = 0.0
        for g, n in zip(grupper, antal):
            for j in range(M.shape[1]):
                gns = np.bincount(g, weights=M[:, j]) / n
                stoerst = max(stoerst, np.abs(gns).max())
                M[:, j] -= gns[g]
        if stoerst < tol:
            return M
    print(f"  ADVARSEL: demeaning ikke konvergeret efter {maxit} iterationer ({stoerst:.2e})")
    return M


def feols(data, ekstra_led, fe, navn):
    """OLS af ln_pris på boligkarakteristika + ekstra led med faste effekter fe.
    Klyngerobuste std.fejl på postnummer (CRV1, samme småstikprøvekorrektion som pyfixest)."""
    X = design(data, BOLIG_LED + ekstra_led)
    # manglende faste effekter (fx postnummer gemt som tekst "nan") fjernes – som i pyfixest
    fe_ok = data[fe].notna().all(axis=1) & ~data[fe].astype(str).eq("nan").any(axis=1)
    ok = X.notna().all(axis=1) & data["ln_pris"].notna() & fe_ok
    d = data[ok]
    # fjern postnumre (og perioder) med kun én handel – singletons – til ingen er tilbage
    while True:
        enlig = np.zeros(len(d), dtype=bool)
        for f in fe:
            enlig |= d.groupby(f)[f].transform("size").to_numpy() == 1
        if not enlig.any():
            break
        d = d[~enlig]
    X = X.loc[d.index]
    X = X.loc[:, X.std() > 0]                      # dummies uden observationer fjernes
    y = d["ln_pris"].to_numpy(float)
    grupper = [pd.factorize(d[f])[0] for f in fe]
    M = demean(np.column_stack([y, X.to_numpy(float)]), grupper)
    yd, Xd = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    b = XtX_inv @ (Xd.T @ yd)
    e = yd - Xd @ b
    N, K = Xd.shape
    kl = pd.factorize(d["postnr"])[0]
    G = kl.max() + 1
    U = np.column_stack([np.bincount(kl, weights=Xd[:, j] * e, minlength=G) for j in range(K)])
    V = G / (G - 1) * (N - 1) / (N - K) * XtX_inv @ (U.T @ U) @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    print(f"{navn}: N = {dk(N)}, R² = {r2:.3f}, FE = {' + '.join(fe)}")
    return {"navn": navn, "coef": pd.Series(b, X.columns), "se": pd.Series(se, X.columns),
            "N": N, "r2": r2, "data": d, "X": X, "fe": fe}


def faste_effekter(res, tol=1e-10, maxit=1000):
    """Genskaber de faste effekter ud fra residualet r = y − Xb (skiftevis gennemsnit)."""
    d = res["data"]
    r = d["ln_pris"].to_numpy(float) - res["X"].to_numpy(float) @ res["coef"].to_numpy()
    koder = [pd.factorize(d[f]) for f in res["fe"]]
    eff = [np.zeros(len(k[1])) for k in koder]
    for it in range(maxit):
        stoerst = 0.0
        for i, (g, _) in enumerate(koder):
            rest = r - sum(eff[j][koder[j][0]] for j in range(len(koder)) if j != i)
            ny = np.bincount(g, weights=rest) / np.bincount(g)
            stoerst = max(stoerst, np.abs(ny - eff[i]).max())
            eff[i] = ny
        if stoerst < tol:
            break
    return {f: pd.Series(eff[i], index=koder[i][1]) for i, f in enumerate(res["fe"])}


def pct(b):
    return 100 * (np.exp(b) - 1)


# =============================================================================
# 3. KONTROL: genskaber estimatoren M4 fra hovedmodel.py?
# =============================================================================
overskrift("Kontrol af estimatoren mod hovedmodellen (M4)")
m4_alle = feols(df, [], ["kvartal", "postnr"], "M4 (hele stikprøven)")
try:
    t5 = pd.read_csv(UD / "tabel_5_regression.csv", sep=";", decimal=",", encoding="utf-8-sig",
                     index_col=0)
    ref_b = t5["M4"].dropna().astype(str).str.extract(r"^(-?[\d.]+)")[0].astype(float).dropna()
    faelles_navne = ref_b.index.intersection(m4_alle["coef"].index)
    afv = (m4_alle["coef"][faelles_navne] - ref_b[faelles_navne]).abs()
    print(f"Sammenlignet {len(faelles_navne)} koefficienter med tabel_5 (afrundet til 4 decimaler).")
    print(f"Største afvigelse: {afv.max():.5f}  (≤ 0,00005 = identisk inden for afrundingen)")
    print(f"N i tabel_5: {t5.loc['N', 'M4']}  –  N her: {dk(m4_alle['N'])}")
except Exception as fejl:                                    # fx hvis filen ikke kan læses
    print(f"Kunne ikke sammenligne med tabel_5: {fejl}")

# Stikprøve med kystafstand (mister kun ganske få handler)
dfg = df[df["kystafstand_m"].notna()].copy()
median_pris = dfg["pris"].median()
print(f"\nStikprøve med kystafstand: {dk(len(dfg))} handler. Medianpris: {dk(median_pris)} kr.")

# =============================================================================
# 4. KYSTPRÆMIE
# =============================================================================
overskrift("Kystpræmie")
KYST_LED = [f"i(kystinterval, ref='{KYST_REF}')"]
modeller = {
    "M4": feols(dfg, [], ["kvartal", "postnr"], "M4 (samme stikprøve)"),
    "G1": feols(dfg, KYST_LED, ["kvartal"], "G1 kyst uden postnr-FE"),
    "G2": feols(dfg, KYST_LED, ["kvartal", "postnr"], "G2 kyst med postnr-FE"),
    "G3": feols(dfg, ["ln_kyst_km"], ["kvartal", "postnr"], "G3 ln(kystafstand) med postnr-FE"),
    "G4": feols(dfg, KYST_LED + ["ln_taethed"], ["kvartal", "postnr"], "G4 G2 + ln(tæthed)"),
}
raekker = []
for navn in ["G1", "G2", "G3", "G4"]:
    m = modeller[navn]
    for v in m["coef"].index:
        if v.startswith("kystinterval::") or v in ("ln_kyst_km", "ln_taethed"):
            b, s = m["coef"][v], m["se"][v]
            er_log = v.startswith("ln_")
            raekker.append({
                "model": navn, "variabel": v, "beta": b, "SE_klynge_postnr": s,
                "pct_effekt": np.nan if er_log else pct(b),
                "pct_KI_nedre": np.nan if er_log else pct(b - 1.96 * s),
                "pct_KI_oevre": np.nan if er_log else pct(b + 1.96 * s),
                "kr_ved_medianpris": np.nan if er_log else median_pris * (np.exp(b) - 1),
                "pct_ved_10pct_laengere": 100 * (1.1 ** b - 1) if er_log else np.nan,
                "N": m["N"], "R2": m["r2"]})
t27 = pd.DataFrame(raekker).set_index(["model", "variabel"])
print(t27.round(4).to_string())
gem_tabel(t27.round(5), "tabel_27_kystpraemie.csv")

# Deskriptivt pr. interval: er kystnære huse anderledes?
overskrift("Tabel 28 – handler pr. afstandsinterval")
grp = dfg.groupby("kystinterval", observed=True)
t28 = pd.DataFrame({
    "Handler": grp.size(),
    "Andel_pct": 100 * grp.size() / len(dfg),
    "Median_kr_m2": grp["kr_pr_m2"].median(),
    "Median_boligareal": grp["byg039bygningenssamledeboligareal"].median(),
    "Median_opfoerelsesaar": grp["byg026opfoerelsesaar"].median(),
    "Andel_F_G_pct": grp["energimaerke"].apply(lambda s: 100 * s.isin(["F", "G"]).mean()),
    "Andel_uden_maerke_pct": grp["energimaerke"].apply(lambda s: 100 * (s == "Intet").mean()),
    "Andel_raekkehus_pct": 100 * grp["raekkehus"].mean(),
    "Median_taethed_km2": grp["taethed_km2"].median(),
})
print(t28.round(1).to_string())
gem_tabel(t28.round(2), "tabel_28_kyst_deskriptiv.csv")

# =============================================================================
# 5. OMITTED VARIABLE BIAS: ændres de øvrige koefficienter?
# =============================================================================
overskrift("Tabel 29 – M4 med og uden kystafstand (samme stikprøve)")
b4, b2 = modeller["M4"]["coef"], modeller["G2"]["coef"]
navne = [v for v in b4.index if v.startswith(("energimaerke::", "opfoerelsesperiode::",
                                               "opvarmning::", "tag::", "ydervaeg::"))]
navne = ["ln_boligareal", "ln_grundareal"] + navne
t29 = pd.DataFrame({"beta_M4": b4[navne], "beta_G2": b2[navne]})
t29["aendring"] = t29["beta_G2"] - t29["beta_M4"]
t29["pct_M4"] = pct(t29["beta_M4"])
t29["pct_G2"] = pct(t29["beta_G2"])
print(t29.round(4).to_string())
print(f"\nStørste ændring i en koefficient: {t29['aendring'].abs().max():.4f} "
      f"({t29['aendring'].abs().idxmax()})")
gem_tabel(t29.round(5), "tabel_29_ovb_kyst.csv")

# =============================================================================
# 6. HVAD BESTÅR BELIGGENHEDSEFFEKTEN AF?
# =============================================================================
overskrift("Tabel 30 – postnummer-effekter forklaret af kyst, tæthed og region")
fe4 = faste_effekter(modeller["M4"])["postnr"]
pn = dfg.groupby("postnr").agg(
    handler=("pris", "size"), kyst_km=("kyst_km", "median"),
    taethed=("taethed_km2", "median"), region=("region", lambda s: s.mode().iat[0]))
pn["fe"] = fe4.reindex(pn.index)
pn = pn[(pn["handler"] >= MIN_HANDLER_POSTNR) & pn["fe"].notna()].copy()
pn["fe"] = pn["fe"] - pn["fe"].mean()        # niveauet er vilkårligt – centreres
pn["ln_kyst_km"] = np.log(pn["kyst_km"])
pn["ln_taethed"] = np.log(pn["taethed"])
print(f"Postnumre med mindst {MIN_HANDLER_POSTNR} handler: {len(pn)}")
print(f"Spredning i postnummer-effekterne: std = {pn['fe'].std():.3f} log-point "
      f"(p10–p90: {pct(pn['fe'].quantile(0.1)):.0f} % til {pct(pn['fe'].quantile(0.9)):.0f} %)")


def ols_hc1(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    V = n / (n - k) * XtX_inv @ (X.T * e ** 2) @ X @ XtX_inv
    return b, np.sqrt(np.diag(V)), 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


reg_dummies = pd.get_dummies(pn["region"]).drop(columns="Hovedstaden").astype(float)
spec = {
    "D1 region": reg_dummies,
    "D2 kyst + tæthed": pn[["ln_kyst_km", "ln_taethed"]],
    "D3 kyst + tæthed + region": pd.concat([pn[["ln_kyst_km", "ln_taethed"]], reg_dummies], axis=1),
}
raekker = []
for navn, X in spec.items():
    b, s, r2 = ols_hc1(pn["fe"].to_numpy(), X.to_numpy(float))
    for v, bi, si in zip(["konstant"] + list(X.columns), b, s):
        raekker.append({"model": navn, "variabel": v, "beta": bi, "SE_HC1": si, "R2": r2,
                        "N_postnumre": len(pn)})
    print(f"{navn}: R² = {r2:.3f}")
t30 = pd.DataFrame(raekker).set_index(["model", "variabel"])
print(t30.round(4).to_string())
gem_tabel(t30.round(5), "tabel_30_postnr_dekomposition.csv")

# =============================================================================
# 7. TID: MÅNEDS-FE OG SÆSON
# =============================================================================
overskrift("Tabel 31 – M4 med kvartals- vs. måneds-FE")
m4m = feols(dfg, [], ["aarmaaned", "postnr"], "M4 med år×måned-FE")
udv = ["ln_boligareal", "ln_grundareal", "energimaerke::A", "energimaerke::G",
       "energimaerke::Intet", "opvarmning::Olie", "opvarmning::Gas", "opfoerelsesperiode::2010+"]
t31 = pd.DataFrame({"beta_kvartal_FE": b4[udv], "beta_maaned_FE": m4m["coef"][udv]})
t31["aendring"] = t31["beta_maaned_FE"] - t31["beta_kvartal_FE"]
t31.loc["R²"] = [modeller["M4"]["r2"], m4m["r2"], m4m["r2"] - modeller["M4"]["r2"]]
t31.loc["N"] = [modeller["M4"]["N"], m4m["N"], 0]
print(t31.round(4).to_string())
print(f"Største ændring i en koefficient (alle): "
      f"{(m4m['coef'] - b4).abs().reindex(b4.index).max():.4f}")
gem_tabel(t31.round(5), "tabel_31_maaned_fe.csv")

overskrift("Tabel 32 – sæsonmønster")
# Hedonisk månedsindeks = de faste effekter for år×måned (kvalitetskorrigeret prisniveau)
idx = faste_effekter(m4m)["aarmaaned"].sort_index()
handler_pr_maaned = dfg.groupby("aarmaaned").size().reindex(idx.index)
idx = idx[handler_pr_maaned >= 1000]            # udelad ufuldstændige måneder (fx den sidste)
# Trend = centreret 2×12 glidende gennemsnit; sæson = afvigelsen fra trenden (klassisk dekomposition)
v = np.r_[0.5, np.ones(11), 0.5] / 12
trend = pd.Series(np.convolve(idx.to_numpy(), v, mode="same"), index=idx.index)
trend.iloc[:6], trend.iloc[-6:] = np.nan, np.nan
afv = (idx - trend).dropna()
kal = pd.PeriodIndex(afv.index, freq="M").month
saeson = afv.groupby(kal).mean()
saeson = saeson - saeson.mean()                  # sæsonfaktorerne summerer til 0 over året
# Handler: andel af årets handler pr. kalendermåned, gennemsnit for de hele år 2019–2025
hele = dfg[dfg["salgsaar"].between(2019, 2025)]
andel = (hele.groupby(["salgsaar", "maaned"]).size()
         / hele.groupby("salgsaar").size()).groupby(level="maaned").mean() * 100
t32 = pd.DataFrame({"saeson_pris_pct": pct(saeson), "andel_handler_pct": andel,
                    "antal_aar_i_saesonberegning": afv.groupby(kal).size()})
t32.index = MAANEDER
print(t32.round(2).to_string())
print(f"Forskel mellem højeste og laveste måned i pris: "
      f"{t32['saeson_pris_pct'].max() - t32['saeson_pris_pct'].min():.1f} procentpoint")
gem_tabel(t32.round(3), "tabel_32_saeson.csv")

# =============================================================================
# 8. FIGURER (uden titel – titlen står i figurteksten i Word)
# =============================================================================
overskrift("Figurer")
fig, ax = plt.subplots(figsize=(16 / 2.54, 5.5 / 2.54))
interv = [n for n in KYST_NAVNE if n != KYST_REF]
y = np.arange(len(interv))
for i, (navn, etiket, farve, mk, fyld) in enumerate([
        ("G1", "Uden postnummer-FE (på tværs af landet)", MUTED, "s", "full"),
        ("G2", "Med postnummer-FE (inden for samme postnummer)", BLAA, "o", "full")]):
    rk = t27.loc[navn].reindex([f"kystinterval::{n}" for n in interv])
    eff = rk["pct_effekt"].to_numpy()
    ax.errorbar(eff, y + (i - 0.5) * 0.28,
                xerr=[eff - rk["pct_KI_nedre"].to_numpy(), rk["pct_KI_oevre"].to_numpy() - eff],
                fmt=mk, color=farve, ecolor=farve, markersize=4.5, elinewidth=1.2, capsize=2,
                fillstyle=fyld, label=etiket)
    for xi, yi in zip(eff, y + (i - 0.5) * 0.28):
        ax.text(xi, yi - 0.17, f"{dk(xi, 1)} %", ha="center", va="bottom", fontsize=7, color=farve)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(y, interv)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(left=min(0, ax.get_xlim()[0]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{dk(x)} %"))
ax.set_xlabel("Forskel i salgspris i forhold til huse over 5 km fra kysten (%), 95 %-KI")
ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.savefig(UD / "fig_18_kystpraemie.png")
plt.close(fig)
print("  gemt: fig_18_kystpraemie.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 / 2.54, 6 / 2.54))
x = np.arange(12)
a1.bar(x, t32["saeson_pris_pct"], color=BLAA, width=0.7)
a1.axhline(0, color=MUTED, lw=1)
a1.set_title("a) Sæsonudsving i kvalitetskorrigeret pris (%)", loc="left", fontsize=8.5)
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{dk(v, 1)}"))
a2.bar(x, t32["andel_handler_pct"], color=INK2, width=0.7)
a2.axhline(100 / 12, color=MUTED, lw=1, ls="--")
a2.set_title("b) Andel af årets handler (%), stiplet = 1/12", loc="left", fontsize=8.5)
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{dk(v, 0)}"))
for a in (a1, a2):
    a.set_xticks(x, MAANEDER, fontsize=7)
    a.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(UD / "fig_19_saeson.png")
plt.close(fig)
print("  gemt: fig_19_saeson.png")

print(f"\nAlle filer ligger i: {UD}")
