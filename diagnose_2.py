"""
Diagnose 2 – struktur i data før rensning
Kør i VS Code. Ret STI (og evt. SEP) til din egen fil, og send hele outputtet tilbage.
"""
import sys
from pathlib import Path

import pandas as pd

STI = str(Path(__file__).resolve().parent.parent / "data" / "ejendomshandler_energi (2).csv")  # rådata i bachelorprojekt/data

# Find separatoren ud fra første linje i filen (mere robust end pandas' gæt)
with open(STI, encoding="utf-8-sig") as f:
    foerste_linje = f.readline()
SEP = max([",", ";", "\t", "|"], key=foerste_linje.count)
print(f"Separator fundet: {SEP!r}")

df = pd.read_csv(STI, sep=SEP, encoding="utf-8-sig", low_memory=False)

# Rens kolonnenavne for mellemrum, anførselstegn og usynlige tegn
df.columns = df.columns.str.strip().str.strip('"').str.replace("﻿", "").str.lower()

print("Dimensioner:", df.shape)
if "behandlingsid" not in df.columns:
    print("Kolonnen 'behandlingsid' blev ikke fundet. Kolonnerne er:")
    print(df.columns.tolist())
    sys.exit()

df["koebsaftaledato"] = pd.to_datetime(df["koebsaftaledato"], errors="coerce")
df["salgsaar"] = df["koebsaftaledato"].dt.year

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)


def overskrift(tekst):
    print("\n" + "=" * 75 + f"\n{tekst}\n" + "=" * 75)


# ---------------------------------------------------------------------------
# 1. Hvorfor er der flere rækker pr. handel?
# ---------------------------------------------------------------------------
overskrift("1. Rækker pr. handel")
n_pr_handel = df.groupby("behandlingsid").size()
print(n_pr_handel.value_counts().sort_index().head(10))

multi = df[df["behandlingsid"].isin(n_pr_handel[n_pr_handel > 1].index)]

print("\nForskellige ejendomme (BFE) pr. handel – kun handler med flere rækker:")
print(multi.groupby("behandlingsid")["bestemtfastejendombfenr"].nunique().value_counts().head())

print("\nForskellige bygninger pr. handel – kun handler med flere rækker:")
print(multi.groupby("behandlingsid")["bygning_id"].nunique().value_counts().head())

print("\nForskellige købesummer pr. handel – kun handler med flere rækker:")
print(multi.groupby("behandlingsid")["samletkoebesum"].nunique().value_counts().head())

print("\nRækker pr. (handel, bygning) – over 1 tyder på dubletter fra fletning (fx energimærker):")
print(df.groupby(["behandlingsid", "bygning_id"]).size().value_counts().head())

print("\nBygningsanvendelse i handler med flere rækker:")
print(multi["byg021bygningensanvendelse"].value_counts().head(10))

print("\nTo eksempler på handler med flere rækker:")
kolonner = ["behandlingsid", "bestemtfastejendombfenr", "bygning_id", "byg021bygningensanvendelse",
            "samletkoebesum", "byg039bygningenssamledeboligareal", "energimaerke_klasse",
            "energimaerke_validfrom", "vurderingsaar"]
for eks in multi["behandlingsid"].drop_duplicates().iloc[[0, 1000]]:
    print(df.loc[df["behandlingsid"] == eks, kolonner].to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Datoer
# ---------------------------------------------------------------------------
overskrift("2. Datoer")
idag = pd.Timestamp("2026-10-19")
print("Rækker med aftaledato efter i dag:", (df["koebsaftaledato"] > idag).sum())
print("\nUnikke handler pr. salgsår:")
print(df.drop_duplicates("behandlingsid")["salgsaar"].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 3. Mangler 'betinget' i bestemte år?
# ---------------------------------------------------------------------------
overskrift("3. Andel manglende 'betinget' pr. salgsår")
print(df.groupby("salgsaar")["betinget"].apply(lambda s: s.isna().mean()).round(3))

# ---------------------------------------------------------------------------
# 4. Hvilken vurdering hører til hvilket salgsår?
# ---------------------------------------------------------------------------
overskrift("4. Vurderingsår (kolonner) mod salgsår (rækker)")
print(pd.crosstab(df["salgsaar"], df["vurderingsaar"]))

# ---------------------------------------------------------------------------
# 5. Pris og kr./m²
# ---------------------------------------------------------------------------
overskrift("5. Pris og kr./m²")
pct = [.001, .01, .05, .5, .95, .99, .999]
print("samletkoebesum:")
print(df["samletkoebesum"].describe(percentiles=pct).apply(lambda x: f"{x:,.0f}"))
kvm = df["samletkoebesum"] / df["byg039bygningenssamledeboligareal"]
print("\nkr. pr. m² boligareal:")
print(kvm.describe(percentiles=pct).apply(lambda x: f"{x:,.0f}"))
begge = df[["samletkoebesum", "kontantkoebesum"]].dropna()
print("\nAndel hvor kontantkøbesum = samlet købesum (hvor begge findes):",
      round((begge["samletkoebesum"] == begge["kontantkoebesum"]).mean(), 3))

# ---------------------------------------------------------------------------
# 6. Betyder NaN 'ingen'? (findes der nuller i forvejen?)
# ---------------------------------------------------------------------------
overskrift("6. NaN vs. nuller")
for col in ["byg042arealindbyggetgarage", "byg043arealindbyggetcarport",
            "byg044arealindbyggetudhus", "byg045arealindbyggetudestueellerlign",
            "byg027omtilbygningsaar", "byg070fredning", "sum_badevaerelser", "sum_toiletter"]:
    s = df[col]
    print(f"{col:40s} NaN: {s.isna().mean():6.1%}   nuller: {(s == 0).mean():6.1%}   min: {s.min()}")

# ---------------------------------------------------------------------------
# 7. Arealvariable
# ---------------------------------------------------------------------------
overskrift("7. Arealvariable")
arealer = ["byg038samletbygningsareal", "byg039bygningenssamledeboligareal", "byg041bebyggetareal",
           "sum_enhedsareal", "sum_beboelsesareal_enhed", "vurderet_areal_m2", "grundareal_m2"]
print(df[arealer].describe().T[["count", "mean", "50%", "min", "max"]].round(0))
print("\nKorrelationer:")
print(df[arealer].corr().round(2))

# ---------------------------------------------------------------------------
# 8. Energimærker
# ---------------------------------------------------------------------------
overskrift("8. Energimærker")
print(df["energimaerke_status"].value_counts(dropna=False))
print("\nAndel uden energimærke pr. salgsår:")
print(df.groupby("salgsaar")["energimaerke_klasse"].apply(lambda s: s.isna().mean()).round(3))

# ---------------------------------------------------------------------------
# 9. Koder og geografi
# ---------------------------------------------------------------------------
overskrift("9. Koder og geografi")
print("Bygningsanvendelse (rækker) mod ejendomstype (kolonner):")
print(pd.crosstab(df["byg021bygningensanvendelse"], df["ejendomstype"].fillna(-1)))
print("\nvur_benyttelsekode:")
print(df["vur_benyttelsekode"].value_counts(dropna=False).head(10))
print("\nAntal kommuner:", df["kommuneinddelingkommunekode"].nunique(),
      "  Antal postnumre:", df["postnr"].nunique())
