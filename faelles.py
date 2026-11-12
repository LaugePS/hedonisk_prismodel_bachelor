"""
faelles.py – fælles indstillinger, hjælpefunktioner og dataforberedelse

Bruges af hovedmodel.py og regional.py, så alle modeller bygger på præcis de samme
variable og den samme stikprøve. Køres ikke selv.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOVEDPERIODE_FRA = 2019
REGION_RAEKKEFOELGE = ["Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland", "Nordjylland"]

# Referencekategorier (dummies fortolkes altid i forhold til disse)
REF = {
    "energimaerke": "D",            # midt på skalaen
    "opfoerelsesperiode": "1960–69",  # stor og "typisk" parcelhusperiode
    "badevaerelser_kat": "1",
    "opvarmning": "Fjernvarme",     # mest udbredte
    "ydervaeg": "Mursten",
    "tag": "Tegl",
}

INK, INK2, MUTED, GRID, BLAA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def dk(x, dec=0):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def overskrift(tekst):
    print("\n" + "=" * 75 + f"\n{tekst}\n" + "=" * 75)


def forbered_data(mappe, fra_aar=HOVEDPERIODE_FRA, fil="data_renset.csv"):
    """Indlæser det rensede datasæt, danner modelvariablene og returnerer (df, bolig-formel).

    Stikprøve: handler fra og med fra_aar (standard: hovedperioden 2019+) og kun handler med
    alle nødvendige variable. fra_aar og fil ændres kun i robusthedstjekkene.
    """
    overskrift("Data")
    df = pd.read_csv(mappe / fil, sep=";", decimal=",", encoding="utf-8-sig",
                     low_memory=False)
    df = df[df["salgsaar"] >= fra_aar].copy()
    print(f"Fil: {fil}   Handler fra {fra_aar}: {dk(len(df))}")

    # Areal og grund
    df["ln_grundareal"] = np.log(df["grundareal_m2"].where(df["grundareal_m2"] > 0))
    df["raekkehus"] = (df["boligtype"] != "Fritliggende").astype(int)
    # Grundareal betyder noget andet for række-/dobbelthuse (lille egen parcel) -> interaktion.
    # ln(grundareal) centreres om gennemsnittet før interaktionen. Det ændrer ikke elasticiteten,
    # men gør, at række-dummyen måler forskellen ved et gennemsnitligt grundareal, og det
    # fjerner den kunstige multikollinearitet mellem dummyen og interaktionsleddet.
    df["ln_grundareal"] = df["ln_grundareal"] - df["ln_grundareal"].mean()
    df["ln_grund_x_raekkehus"] = df["ln_grundareal"] * df["raekkehus"]

    # Værelser: 0 er en registreringsfejl
    df["vaerelser"] = df["sum_vaerelser"].where(df["sum_vaerelser"] > 0)

    # Badeværelser som kategorier (1, 2, 3+, ukendt). 0 og manglende -> "Ukendt"
    b = df["sum_badevaerelser"]
    df["badevaerelser_kat"] = np.select([b == 1, b == 2, b >= 3], ["1", "2", "3+"], "Ukendt")

    df["to_etager_plus"] = (df["byg054antaletager"] >= 2).astype(int)

    # Opvarmning (BBR byg056 varmeinstallation + byg057 opvarmningsmiddel)
    v, m = df["byg056varmeinstallation"], df["byg057opvarmningsmiddel"]
    central = v.isin([2, 6])
    df["opvarmning"] = np.select(
        [v == 1, v == 5, (v == 7) | (central & (m == 1)), central & m.isin([2, 7, 10]) | (v == 8),
         central & (m == 3), v.notna()],
        ["Fjernvarme", "Varmepumpe", "Elvarme", "Gas", "Olie", "Andet"], "Ukendt")

    # Ydervæg (byg032) og tag (byg033) i få grupper.
    # Koderne følger BBR's kodelister (bekræftet via Danmarks Statistiks TIMES-dokumentation).
    # OBS: variabeloversigtens kodeforklaring for tag er forkert.
    #   byg032: 1 mursten, 2 letbeton, 3 fibercement inkl. asbest, 4 bindingsværk, 5 træ,
    #           6 betonelementer, 8 metal, 10 fibercement asbestfri, 11 PVC, 12 glas, 90 andet
    #   byg033: 1 built-up (fladt tag), 2 tagpap med hældning, 3 fibercement inkl. asbest,
    #           4 cementsten (betontagsten), 5 tegl, 6 metal, 7 strå, 10 fibercement asbestfri,
    #           11 PVC, 12 glas, 90 andet
    w = df["byg032ydervaeggensmateriale"]
    df["ydervaeg"] = np.select(
        [w == 1, w == 5, w.isin([2, 6]), w == 4, w.isin([3, 10]), w.notna()],
        ["Mursten", "Træ", "Letbeton/beton", "Bindingsværk", "Fibercement", "Andet"], "Ukendt")
    t = df["byg033tagdaekningsmateriale"]
    df["tag"] = np.select(
        [t == 5, t == 4, t.isin([3, 10]), t.isin([1, 2]), t == 7, t == 6, t.notna()],
        ["Tegl", "Betontagsten", "Fibercement", "Tagpap", "Stråtag", "Metal", "Andet"], "Ukendt")

    df["postnr"] = df["postnr"].astype(str)
    df["kvartal"] = df["kvartal"].astype(str)
    df["opfoerelsesperiode"] = df["opfoerelsesperiode"].astype(str)

    for col in ["opvarmning", "ydervaeg", "tag", "badevaerelser_kat", "energimaerke",
                "opfoerelsesperiode"]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())

    # Tjek at referencekategorierne findes i data (ellers bruges den hyppigste kategori)
    for col, ref in REF.items():
        if ref not in set(df[col]):
            ny = df[col].value_counts().index[0]
            print(f"ADVARSEL: referencen '{ref}' findes ikke i {col} – bruger '{ny}' i stedet")
            REF[col] = ny

    # Samme stikprøve i alle fire modeller (ellers kan ændringer skyldes forskellige data)
    noedvendige = ["ln_pris", "ln_boligareal", "ln_grundareal", "vaerelser", "postnr", "kvartal",
                   "opfoerelsesperiode"]
    foer = len(df)
    df = df.dropna(subset=noedvendige)
    df = df[df["opfoerelsesperiode"] != "nan"]
    print(f"\nHandler med alle variable: {dk(len(df))} (fjernet {dk(foer - len(df))} pga. manglende værdier)")

    bolig = ("ln_boligareal + ln_grundareal + ln_grund_x_raekkehus + raekkehus + vaerelser"
             " + i(badevaerelser_kat, ref='{badevaerelser_kat}') + to_etager_plus"
             " + i(opfoerelsesperiode, ref='{opfoerelsesperiode}') + ombygget"
             " + har_garage + har_carport + har_udhus + har_udestue"
             " + i(energimaerke, ref='{energimaerke}') + i(opvarmning, ref='{opvarmning}')"
             " + i(ydervaeg, ref='{ydervaeg}') + i(tag, ref='{tag}')").format(**REF)
    return df, bolig
