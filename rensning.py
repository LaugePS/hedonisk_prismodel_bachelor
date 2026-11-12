"""
rensning.py – Rensning af ejendomshandler til bachelorprojektet

Følger rensningsreglerne i arbejdsnoterne (afsnit 4), i samme rækkefølge.
Output (gemmes i samme mappe som datafilen):
  - data_renset.csv          én række pr. handel, klar til analyse (valg B)
  - data_flere_bygninger.csv  handler med flere bygninger til robusthedstjek (valg A)
  - data_renset_inkl_ekstreme.csv  som data_renset.csv, men uden grænserne for ekstreme huse
                                   (trin 10a–10c) – til robusthedstjek
  - frasorteringstabel.csv    antal handler før/efter hvert trin (til dataafsnittet)

Kør i VS Code. Ret kun STI og evt. DATO_UDTRAEK. Send hele outputtet tilbage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# INDSTILLINGER
# =============================================================================
STI = str(Path(__file__).resolve().parent.parent / "data" / "ejendomshandler_energi (2).csv")  # rådata i bachelorprojekt/data
DATO_UDTRAEK = pd.Timestamp("2026-10-19")  # handler efter denne dato fjernes

# Bygningsstatus (BBR's livscyklus-kodeliste). Kun opførte bygninger beholdes.
BYGNING_STATUS_BEHOLD = [6]
BYGNING_STATUS_NAVN = {2: "Projekteret", 3: "Under opførelse", 6: "Opført", 9: "Afsluttet",
                       10: "Historisk", 11: "Fejlregistreret", 12: "Midlertidig afsluttet",
                       13: "Delvis afsluttet", 14: "Henlagt"}

ANVENDELSE_FRITLIGGENDE = [120, 121, 122]
ANVENDELSE_RAEKKEHUS = [130, 131, 132]

MAX_ANDEL_MANGLENDE_KONTANT = 0.02  # kontantkøbesum bruges, hvis højst 2 % mangler fra 2019
TRIM_ANDEL = 0.01                   # 1 % i hver ende af kr./m² pr. salgsår
HOVEDPERIODE_FRA = 2019

# Opførelsesår før denne grænse tolkes som fejl/ukendt og sættes til manglende
MIN_OPFOERELSESAAR = 1600

# Grænser for ekstreme huse (fastlagt 19/10 ud fra fordelingerne i
# "Diagnose af ekstreme værdier"). Sæt til None for at slå en grænse fra.
BOLIGAREAL_MIN = 40      # under 0,1 %-fraktilen (49 m²)
BOLIGAREAL_MAX = 500     # over 99,9 %-fraktilen (422 m²)
GRUNDAREAL_MAX = 20_000  # ca. 99 %-fraktilen (18.700 m²); fanger landbrugslignende grunde

# Antal over disse grænser er registreringsfejl og sættes til manglende
MAX_VAERELSER = 20       # 99,9 %-fraktilen er 16
MAX_BADEVAERELSER = 10   # 99,9 %-fraktilen er 5
MAX_TOILETTER = 10       # 99,9 %-fraktilen er 6

# Kommunekode -> region (98 kommuner + Christiansø)
REGION = {}
for navn, koder in {
    "Hovedstaden": [101, 147, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 173, 175, 183,
                    185, 187, 190, 201, 210, 217, 219, 223, 230, 240, 250, 260, 270, 400, 411],
    "Sjælland": [253, 259, 265, 269, 306, 316, 320, 326, 329, 330, 336, 340, 350, 360, 370,
                 376, 390],
    "Syddanmark": [410, 420, 430, 440, 450, 461, 479, 480, 482, 492, 510, 530, 540, 550, 561,
                   563, 573, 575, 580, 607, 621, 630],
    "Midtjylland": [615, 657, 661, 665, 671, 706, 707, 710, 727, 730, 740, 741, 746, 751, 756,
                    760, 766, 779, 791],
    "Nordjylland": [773, 787, 810, 813, 820, 825, 840, 846, 849, 851, 860],
}.items():
    for k in koder:
        REGION[k] = navn


# =============================================================================
# HJÆLPEFUNKTIONER
# =============================================================================
def overskrift(tekst):
    print("\n" + "=" * 75 + f"\n{tekst}\n" + "=" * 75)


frasortering = []


def log_trin(trin, beskrivelse, df):
    """Gemmer antal handler og rækker efter et rensningstrin."""
    handler = df["behandlingsid"].nunique()
    forrige = frasortering[-1]["handler_efter"] if frasortering else handler
    frasortering.append({
        "trin": trin,
        "beskrivelse": beskrivelse,
        "handler_efter": handler,
        "raekker_efter": len(df),
        "handler_fjernet": forrige - handler,
    })
    print(f"Trin {trin:>4}: {beskrivelse:60s} handler: {handler:>9,}  (fjernet {forrige - handler:,})")


def til_tal(s):
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# 0. INDLÆSNING
# =============================================================================
overskrift("0. Indlæsning")
df = pd.read_csv(STI, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)
df.columns = df.columns.str.strip().str.strip('"').str.replace("﻿", "").str.lower()
print("Dimensioner:", df.shape)

df["koebsaftaledato"] = pd.to_datetime(df["koebsaftaledato"], errors="coerce")
df["salgsaar"] = df["koebsaftaledato"].dt.year

for col in ["samletkoebesum", "kontantkoebesum", "ejendomstype", "bygning_status",
            "byg021bygningensanvendelse", "byg026opfoerelsesaar", "byg027omtilbygningsaar",
            "byg039bygningenssamledeboligareal", "byg042arealindbyggetgarage",
            "byg043arealindbyggetcarport", "byg044arealindbyggetudhus",
            "byg045arealindbyggetudestueellerlign", "sum_vaerelser", "sum_toiletter",
            "sum_badevaerelser", "byg054antaletager", "grundareal_m2",
            "kommuneinddelingkommunekode", "vurderingsaar", "ejendomvaerdibeloeb",
            "grundvaerdibeloeb"]:
    df[col] = til_tal(df[col])

log_trin("0", "Rådata", df)

# =============================================================================
# 1–7. FRASORTERING PÅ RÆKKENIVEAU
# =============================================================================
overskrift("1–7. Frasortering")

# 1. Datoer efter dataudtrækket
df = df[df["koebsaftaledato"] <= DATO_UDTRAEK]
log_trin("1", f"Aftaledato senest {DATO_UDTRAEK.date()}", df)

# 2. Kun almindelig fri handel
df = df[df["overdragelsesmaade"].astype(str).str.strip() == "Almindelig fri handel"]
log_trin("2", "Kun almindelig fri handel", df)

# 3. Fjern betingede handler (manglende værdier beholdes)
betinget = df["betinget"].astype(str).str.strip().str.lower().isin(["true", "1", "t"])
df = df[~betinget]
log_trin("3", "Betingede handler fjernet", df)

# 4. Kun samlet fast ejendom
df = df[df["ejendomstype"] == 1]
log_trin("4", "Kun ejendomstype 1 (samlet fast ejendom)", df)

# 5. Kun handler med én ejendom (BFE)
n_bfe = df.groupby("behandlingsid")["bestemtfastejendombfenr"].transform("nunique")
df = df[n_bfe == 1]
log_trin("5", "Handler med flere ejendomme fjernet", df)

# 6. Kun handler med én købesum og én aftaledato
n_pris = df.groupby("behandlingsid")["samletkoebesum"].transform("nunique")
n_dato = df.groupby("behandlingsid")["koebsaftaledato"].transform("nunique")
df = df[(n_pris == 1) & (n_dato == 1)]
log_trin("6", "Handler med flere købesummer/datoer fjernet", df)

# Ægte dubletter af (handel, bygning) fjernes, så hver bygning kun tælles én gang
df = df.drop_duplicates(subset=["behandlingsid", "bygning_id"])
log_trin("6b", "Dubletter af (handel, bygning) fjernet", df)

# 7. Bygningsstatus
print("\nFordeling af bygning_status (inden filter):")
fordeling = df["bygning_status"].value_counts(dropna=False).rename("antal").to_frame()
fordeling["navn"] = fordeling.index.map(BYGNING_STATUS_NAVN)
print(fordeling)
print("\nBygningsstatus pr. salgsår (andel af rækker):")
print(pd.crosstab(df["salgsaar"], df["bygning_status"], normalize="index").round(3))

foer_byg = df.groupby("behandlingsid")["bygning_id"].nunique()
if BYGNING_STATUS_BEHOLD is not None:
    df = df[df["bygning_status"].isin(BYGNING_STATUS_BEHOLD)]
    log_trin("7", f"Kun bygningsstatus {BYGNING_STATUS_BEHOLD}", df)
else:
    log_trin("7", "Bygningsstatus: intet filter endnu (se fordeling ovenfor)", df)

efter_byg = df.groupby("behandlingsid")["bygning_id"].nunique()
blev_enkelt = ((foer_byg.reindex(efter_byg.index) > 1) & (efter_byg == 1)).sum()
print(f"Handler, der gik fra flere bygninger til én pga. statusfilteret: {blev_enkelt:,}")

# =============================================================================
# 8. VALG B: KUN HANDLER MED ÉN BYGNING
# =============================================================================
overskrift("8. Valg B – kun handler med én bygning")
n_byg = df.groupby("behandlingsid")["bygning_id"].transform("nunique")
df["antal_bygninger"] = n_byg
flere_bygninger = df[n_byg > 1].copy()   # gemmes til robusthedstjek (valg A)
df = df[n_byg == 1].copy()
log_trin("8", "Kun handler med én bygning (valg B)", df)
print(f"Handler med flere bygninger gemt til robusthedstjek: "
      f"{flere_bygninger['behandlingsid'].nunique():,}")

# =============================================================================
# 9–10. BOLIGTYPE OG UMULIGE VÆRDIER
# =============================================================================
overskrift("9–10. Boligtype og umulige værdier")
anvendelse_ok = ANVENDELSE_FRITLIGGENDE + ANVENDELSE_RAEKKEHUS
df = df[df["byg021bygningensanvendelse"].isin(anvendelse_ok)]
log_trin("9", "Kun enfamiliehuse og række-/kæde-/dobbelthuse", df)

df = df[df["byg039bygningenssamledeboligareal"] > 0]
log_trin("10", "Boligareal > 0", df)

# Opførelsesår: urealistisk tidlige år er fejl/ukendt -> manglende
print("\nOpførelsesår før 1800 (inden rettelse):")
print(df.loc[df["byg026opfoerelsesaar"] < 1800, "byg026opfoerelsesaar"]
      .value_counts().sort_index().head(20))
for_tidligt = (df["byg026opfoerelsesaar"] < MIN_OPFOERELSESAAR).sum()
print(f"  {for_tidligt} opførelsesår før {MIN_OPFOERELSESAAR} sat til manglende")
df.loc[df["byg026opfoerelsesaar"] < MIN_OPFOERELSESAAR, "byg026opfoerelsesaar"] = np.nan

# Urealistisk mange værelser/badeværelser/toiletter er registreringsfejl -> manglende
for col, graense in [("sum_vaerelser", MAX_VAERELSER), ("sum_badevaerelser", MAX_BADEVAERELSER),
                     ("sum_toiletter", MAX_TOILETTER)]:
    for_mange = (df[col] > graense).sum()
    print(f"  {col}: {for_mange} værdier over {graense} sat til manglende")
    df.loc[df[col] > graense, col] = np.nan

# Ekstreme huse (grænser sættes i INDSTILLINGER, når fordelingerne er set).
# En kopi UDEN disse grænser gemmes til robusthedstjekket (data_renset_inkl_ekstreme.csv).
df_alle = df.copy()
if BOLIGAREAL_MIN is not None:
    df = df[df["byg039bygningenssamledeboligareal"] >= BOLIGAREAL_MIN]
    log_trin("10a", f"Boligareal mindst {BOLIGAREAL_MIN} m²", df)
if BOLIGAREAL_MAX is not None:
    df = df[df["byg039bygningenssamledeboligareal"] <= BOLIGAREAL_MAX]
    log_trin("10b", f"Boligareal højst {BOLIGAREAL_MAX} m²", df)
if GRUNDAREAL_MAX is not None:
    df = df[~(df["grundareal_m2"] > GRUNDAREAL_MAX)]
    log_trin("10c", f"Grundareal højst {GRUNDAREAL_MAX:,} m²", df)

# Negative arealer for garage o.l. og negative antal er registreringsfejl -> manglende
for col in ["byg042arealindbyggetgarage", "byg043arealindbyggetcarport",
            "byg044arealindbyggetudhus", "byg045arealindbyggetudestueellerlign",
            "sum_vaerelser", "sum_toiletter", "sum_badevaerelser", "grundareal_m2"]:
    negative = (df[col] < 0).sum()
    if negative:
        print(f"  {col}: {negative} negative værdier sat til manglende")
    df.loc[df[col] < 0, col] = np.nan
    df_alle.loc[df_alle[col] < 0, col] = np.nan

# =============================================================================
# 10d. VALG AF PRISVARIABEL
# =============================================================================
overskrift("10d. Prisvariabel")
print("Andel manglende kontantkøbesum pr. salgsår:")
print(df.groupby("salgsaar")["kontantkoebesum"].apply(lambda s: s.isna().mean()).round(3))

andel_mangler = df.loc[df["salgsaar"] >= HOVEDPERIODE_FRA, "kontantkoebesum"].isna().mean()
print(f"\nAndel manglende kontantkøbesum fra {HOVEDPERIODE_FRA}: {andel_mangler:.2%}")

if andel_mangler <= MAX_ANDEL_MANGLENDE_KONTANT:
    PRISVARIABEL = "kontantkoebesum"
    df = df[df["kontantkoebesum"] > 0]
    df_alle = df_alle[df_alle["kontantkoebesum"] > 0]
    log_trin("10d", "Kontantkøbesum findes og er > 0", df)
else:
    PRISVARIABEL = "samletkoebesum"
    log_trin("10d", "Samlet købesum bruges (kontant for ufuldstændig)", df)
print(f"Prisvariabel i hovedmodellen: {PRISVARIABEL}")

for d in (df, df_alle):
    d["pris"] = d[PRISVARIABEL]
    d["kr_pr_m2"] = d["pris"] / d["byg039bygningenssamledeboligareal"]

# =============================================================================
# 11. OUTLIERS: 1 % TRIMNING AF KR./M² PR. SALGSÅR
# =============================================================================
overskrift("11. Outliers")
nedre = df.groupby("salgsaar")["kr_pr_m2"].transform(lambda s: s.quantile(TRIM_ANDEL))
oevre = df.groupby("salgsaar")["kr_pr_m2"].transform(lambda s: s.quantile(1 - TRIM_ANDEL))
graenser = df.assign(nedre=nedre, oevre=oevre).groupby("salgsaar")[["nedre", "oevre"]].first()
print("Trimningsgrænser for kr./m² pr. salgsår:")
print(graenser.round(0).astype(int))
df = df[(df["kr_pr_m2"] >= nedre) & (df["kr_pr_m2"] <= oevre)].copy()
log_trin("11", f"Trimning af kr./m² ({TRIM_ANDEL:.0%} i hver ende pr. år)", df)

# Samme trimning af kopien uden grænser for ekstreme huse (egne fraktiler)
n_ = df_alle.groupby("salgsaar")["kr_pr_m2"].transform(lambda s: s.quantile(TRIM_ANDEL))
o_ = df_alle.groupby("salgsaar")["kr_pr_m2"].transform(lambda s: s.quantile(1 - TRIM_ANDEL))
df_alle = df_alle[(df_alle["kr_pr_m2"] >= n_) & (df_alle["kr_pr_m2"] <= o_)].copy()

# =============================================================================
# AFLEDTE VARIABLE
# =============================================================================
overskrift("Afledte variable")


def afled_variable(d, vis=True):
    """Danner de afledte variable. vis=False undertrykker udskrift (bruges til kopien)."""
    import builtins
    print = builtins.print if vis else (lambda *a, **k: None)

    # Tid
    d["kvartal"] = d["koebsaftaledato"].dt.to_period("Q").astype(str)
    d["hovedperiode"] = (d["salgsaar"] >= HOVEDPERIODE_FRA).astype(int)

    # Boligtype
    d["boligtype"] = np.where(d["byg021bygningensanvendelse"].isin(ANVENDELSE_FRITLIGGENDE),
                               "Fritliggende", "Række/kæde/dobbelt")

    # Geografi
    d["region"] = d["kommuneinddelingkommunekode"].map(REGION)
    mangler_region = d.loc[d["region"].isna(), "kommuneinddelingkommunekode"].unique()
    print("Kommunekoder uden region:", mangler_region if len(mangler_region) else "ingen")
    print("Antal kommuner:", d["kommuneinddelingkommunekode"].nunique(),
          " Antal postnumre:", d["postnr"].nunique())

    # Alder og opførelsesperiode
    d["boligalder"] = d["salgsaar"] - d["byg026opfoerelsesaar"]
    d["opfoerelsesperiode"] = pd.cut(
        d["byg026opfoerelsesaar"],
        bins=[0, 1899, 1929, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2100],
        labels=["før 1900", "1900–29", "1930–49", "1950–59", "1960–69", "1970–79",
                "1980–89", "1990–99", "2000–09", "2010+"])

    # Kontrol af den bitemporale kobling
    opfoert_efter = (d["byg026opfoerelsesaar"] > d["salgsaar"]).mean()
    ombygget_efter = (d["byg027omtilbygningsaar"] > d["salgsaar"]).mean()
    print(f"\nKontrol af bitemporal kobling:")
    print(f"  Andel med opførelsesår efter salgsår: {opfoert_efter:.2%}")
    print(f"  Andel med ombygningsår efter salgsår: {ombygget_efter:.2%}")

    # Ombygget (manglende = aldrig ombygget, jf. SEGES). Ombygning efter salget tæller ikke.
    d["ombygget"] = (d["byg027omtilbygningsaar"].notna()
                      & (d["byg027omtilbygningsaar"] <= d["salgsaar"])).astype(int)

    # Garage o.l. (manglende = findes ikke, jf. SEGES)
    for col, navn in [("byg042arealindbyggetgarage", "har_garage"),
                      ("byg043arealindbyggetcarport", "har_carport"),
                      ("byg044arealindbyggetudhus", "har_udhus"),
                      ("byg045arealindbyggetudestueellerlign", "har_udestue")]:
        d[navn] = (d[col] > 0).astype(int)

    # Badeværelser: manglende kan ikke tolkes som 0 -> dummy for manglende
    d["mangler_bad"] = d["sum_badevaerelser"].isna().astype(int)

    # Energimærke ved handlen: A2010/A2015/A2020 -> A, undertyper (C2, D1 ...) -> hovedbogstav
    foerste = d["energimaerke_klasse"].astype(str).str.strip().str.upper().str[0]
    d["energimaerke"] = np.where(foerste.isin(list("ABCDEFG")), foerste, "Intet")
    print("\nEnergimærke efter omkodning:")
    print(d["energimaerke"].value_counts())

    # Logaritmer
    d["ln_pris"] = np.log(d["pris"])
    d["ln_boligareal"] = np.log(d["byg039bygningenssamledeboligareal"])
    d["ln_grundareal"] = np.log(d["grundareal_m2"].where(d["grundareal_m2"] > 0))

    # Offentlig vurdering (kun til sammenligning – aldrig forklarende variabel)
    d["vurd_matcher_salgsaar"] = (d["vurderingsaar"] == d["salgsaar"]).astype(int)
    d["vurderingsgab"] = d["pris"] / d["ejendomvaerdibeloeb"].where(d["ejendomvaerdibeloeb"] > 0)
    return d


df = afled_variable(df)
df_alle = afled_variable(df_alle, vis=False)

# =============================================================================
# TRIN 12: HOVEDPERIODE (markeres, slettes ikke – 2015–2018 bruges til robusthed)
# =============================================================================
log_trin("12", f"Heraf hovedperiode {HOVEDPERIODE_FRA}+", df[df["hovedperiode"] == 1])

# =============================================================================
# DIAGNOSE AF EKSTREME VÆRDIER (bruges til at fastlægge grænserne i INDSTILLINGER)
# =============================================================================
overskrift("Diagnose af ekstreme værdier")
pct = [.001, .005, .01, .05, .5, .95, .99, .995, .999]
print(df[["byg039bygningenssamledeboligareal", "grundareal_m2", "sum_vaerelser",
          "sum_badevaerelser", "sum_toiletter", "byg054antaletager", "pris"]]
      .describe(percentiles=pct).T.drop(columns=["mean", "std"]).round(0).to_string())

print("\nAntal handler over forskellige grundarealer:")
for g in [5_000, 10_000, 20_000, 50_000, 100_000]:
    print(f"  grundareal > {g:>7,} m²: {(df['grundareal_m2'] > g).sum():>7,}")

print("\nAntal handler over forskellige boligarealer:")
for b in [300, 400, 500, 750, 1000]:
    print(f"  boligareal > {b:>5} m²: {(df['byg039bygningenssamledeboligareal'] > b).sum():>7,}")

print("\nVurderingens benyttelseskode: antal, median grundareal og median pris:")
print(df.groupby("vur_benyttelsekode", dropna=False)
      .agg(antal=("pris", "size"), median_grund=("grundareal_m2", "median"),
           median_pris=("pris", "median"))
      .sort_values("antal", ascending=False).head(12).round(0))

# =============================================================================
# OVERBLIK OG GEM
# =============================================================================
overskrift("Overblik over det rensede datasæt")
print(f"Handler i alt: {len(df):,}   heraf {HOVEDPERIODE_FRA}+: {df['hovedperiode'].sum():,}")
print("\nHandler pr. salgsår:")
print(df["salgsaar"].value_counts().sort_index())
print("\nHandler pr. region:")
print(df["region"].value_counts(dropna=False))
print("\nBoligtype:")
print(df["boligtype"].value_counts())
print("\nNøgletal:")
print(df[["pris", "kr_pr_m2", "byg039bygningenssamledeboligareal", "grundareal_m2",
          "boligalder", "sum_vaerelser", "sum_badevaerelser"]]
      .describe().T[["count", "mean", "50%", "min", "max"]].round(0))
print("\nAndel manglende i nøglevariable:")
print(df[["grundareal_m2", "byg026opfoerelsesaar", "sum_vaerelser", "sum_badevaerelser",
          "ejendomvaerdibeloeb", "postnr", "region"]].isna().mean().round(3))
print("\nAndel med vurderingsår = salgsår pr. salgsår:")
print(df.groupby("salgsaar")["vurd_matcher_salgsaar"].mean().round(3))

kolonner = [
    # Id og tid
    "behandlingsid", "bestemtfastejendombfenr", "koebsaftaledato", "salgsaar", "kvartal",
    "hovedperiode",
    # Geografi
    "kommuneinddelingkommunekode", "kommunenavn", "postnr", "postnummer_navn", "region",
    "vejnavn",
    # Pris
    "pris", "ln_pris", "kr_pr_m2", "samletkoebesum", "kontantkoebesum",
    # Bolig
    "boligtype", "byg021bygningensanvendelse", "byg039bygningenssamledeboligareal",
    "ln_boligareal", "grundareal_m2", "ln_grundareal", "byg026opfoerelsesaar", "boligalder",
    "opfoerelsesperiode", "ombygget", "byg054antaletager", "sum_vaerelser", "sum_toiletter",
    "sum_badevaerelser", "mangler_bad", "har_garage", "har_carport", "har_udhus",
    "har_udestue", "byg032ydervaeggensmateriale", "byg033tagdaekningsmateriale",
    "byg056varmeinstallation", "byg057opvarmningsmiddel", "energimaerke",
    # Vurdering (kun sammenligning)
    "vurderingsaar", "vur_benyttelsekode", "ejendomvaerdibeloeb", "grundvaerdibeloeb", "vurd_matcher_salgsaar",
    "vurderingsgab",
]
mappe = Path(STI).parent
df[kolonner].to_csv(mappe / "data_renset.csv", index=False, sep=";", decimal=",",
                    encoding="utf-8-sig")
df_alle[kolonner].to_csv(mappe / "data_renset_inkl_ekstreme.csv", index=False, sep=";",
                         decimal=",", encoding="utf-8-sig")
print(f"Robusthedsdata uden grænser for ekstreme huse: {len(df_alle):,} handler "
      f"(heraf {HOVEDPERIODE_FRA}+: {df_alle['hovedperiode'].sum():,})")
flere_bygninger.to_csv(mappe / "data_flere_bygninger.csv", index=False, sep=";", decimal=",",
                       encoding="utf-8-sig")
tabel = pd.DataFrame(frasortering)
tabel.to_csv(mappe / "frasorteringstabel.csv", index=False, sep=";", encoding="utf-8-sig")

overskrift("Frasorteringstabel")
print(tabel.to_string(index=False))
print(f"\nFiler gemt i: {mappe}")
