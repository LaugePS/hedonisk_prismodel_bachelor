"""
deskriptiv.py – Deskriptiv statistik og grafer til bachelorprojektet

Læser data_renset.csv (fra rensning.py) og kommuner.geojson og laver:
  Tabeller (CSV, semikolon og decimalkomma – åbner direkte i Excel):
    tabel_1_deskriptiv.csv        nøgletal for hovedperioden 2019+
    tabel_2_region.csv            nøgletal pr. region (2019+)
    tabel_3_energimaerke.csv      fordeling og median kr./m² pr. energimærke (2019+)
    tabel_4_kommune.csv           nøgletal pr. kommune (bruges også til kortene)
  Figurer (PNG, 300 dpi, 16 cm brede – klar til Word):
    fig_1_prisudvikling_region.png   median kr./m² pr. kvartal pr. region
    fig_2a_kort_kr_pr_m2.png         median kr./m² pr. kommune (2019+)
    fig_2b_kort_vurderingsgab.png    median salgspris/offentlig vurdering pr. kommune
    fig_3_pris_areal.png             log(pris) mod log(boligareal)
    fig_4a_boxplot_energimaerke.png  kr./m² fordelt på energimærke
    fig_4b_boxplot_opfoerelse.png    kr./m² fordelt på opførelsesperiode
    fig_8_handler_kvartal.png        antal handler pr. kvartal

Kræver: pip3 install pandas numpy matplotlib geopandas
Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Send outputtet i terminalen tilbage.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
HOVEDPERIODE_FRA = 2019
SIDSTE_HELE_KVARTAL = "2026Q2"    # sidste komplette kvartal (bruges i figur 1)
UFULDSTAENDIGT_KVARTAL = "2026Q3"  # kun ca. halvt så mange handler som normalt pga. tinglysnings-
                                   # forsinkelse (data trukket 19/10-2026); vises markeret i figur 8
MIN_HANDLER_KOMMUNE = 30         # kommuner med færre handler vises som "for få handler"
VURDERINGSAAR_KORT = 2020        # vurderingsgab: salg i 2020 med 2020-vurdering

UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)

# ---- Farver (valideret kategorisk palet; rækkefølgen er fast) ----
REGION_RAEKKEFOELGE = ["Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland", "Nordjylland"]
REGION_FARVE = {"Hovedstaden": "#2a78d6", "Sjælland": "#eb6834", "Syddanmark": "#1baf7a",
                "Midtjylland": "#eda100", "Nordjylland": "#e87ba4"}
# Linjetyper og markører, så graferne også kan læses i sort-hvid
REGION_STIL = {"Hovedstaden": ("-", "o"), "Sjælland": ("--", "s"), "Syddanmark": ("-.", "^"),
               "Midtjylland": (":", "D"), "Nordjylland": ((0, (5, 1, 1, 1)), "v")}
BLAA = "#2a78d6"
BLAA_LYS = "#b7d3f6"
BLAA_RAMPE = LinearSegmentedColormap.from_list(
    "blaa", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGERENDE = LinearSegmentedColormap.from_list(
    "blaa_roed", ["#184f95", "#6da7ec", "#f0efec", "#ef8b8a", "#b2302f"])
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BREDDE = 16 / 2.54   # 16 cm i tommer


def dk(x, dec=0):
    """Dansk talformat: 1.234.567,8"""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


FMT_TAL = FuncFormatter(lambda x, _: dk(x))


def gem(fig, navn):
    fig.savefig(UD / navn)
    plt.close(fig)
    print(f"  gemt: {navn}")


def gem_tabel(df, navn):
    df.to_csv(UD / navn, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"  gemt: {navn}")


def overskrift(tekst):
    print("\n" + "=" * 75 + f"\n{tekst}\n" + "=" * 75)


# =============================================================================
# INDLÆSNING
# =============================================================================
overskrift("Indlæsning")
df = pd.read_csv(MAPPE / "data_renset.csv", sep=";", decimal=",", encoding="utf-8-sig",
                 low_memory=False)
df["koebsaftaledato"] = pd.to_datetime(df["koebsaftaledato"], errors="coerce")
print("Handler i alt:", dk(len(df)))
hp = df[df["salgsaar"] >= HOVEDPERIODE_FRA].copy()
print(f"Handler {HOVEDPERIODE_FRA}+:", dk(len(hp)))

# =============================================================================
# TABEL 1: DESKRIPTIV STATISTIK (2019+)
# =============================================================================
overskrift(f"Tabel 1 – deskriptiv statistik ({HOVEDPERIODE_FRA}+)")
variable = {
    "pris": "Salgspris (kr.)",
    "kr_pr_m2": "Pris pr. m² boligareal (kr.)",
    "byg039bygningenssamledeboligareal": "Boligareal (m²)",
    "grundareal_m2": "Grundareal (m²)",
    "boligalder": "Boligens alder ved salg (år)",
    "sum_vaerelser": "Antal værelser",
    "sum_badevaerelser": "Antal badeværelser",
    "ombygget": "Ombygget (andel)",
    "har_garage": "Indbygget garage (andel)",
    "har_carport": "Indbygget carport (andel)",
    "har_udhus": "Indbygget udhus (andel)",
    "har_udestue": "Udestue (andel)",
}
t1 = pd.DataFrame({
    "N": hp[list(variable)].count(),
    "Gennemsnit": hp[list(variable)].mean(),
    "Median": hp[list(variable)].median(),
    "Std.afv.": hp[list(variable)].std(),
    "Min": hp[list(variable)].min(),
    "Maks": hp[list(variable)].max(),
}).rename(index=variable)
t1["Fritliggende (andel)"] = np.nan
fritl = pd.DataFrame({"N": [hp["boligtype"].notna().sum()],
                      "Gennemsnit": [(hp["boligtype"] == "Fritliggende").mean()]},
                     index=["Fritliggende hus (andel)"])
t1 = pd.concat([t1.drop(columns="Fritliggende (andel)"), fritl])
print(t1.round(2).to_string())
gem_tabel(t1.round(3), "tabel_1_deskriptiv.csv")

# =============================================================================
# TABEL 2 OG 3: REGION OG ENERGIMÆRKE (2019+)
# =============================================================================
overskrift("Tabel 2 – pr. region og tabel 3 – pr. energimærke")
t2 = (hp.groupby("region")
      .agg(Handler=("pris", "size"), Median_pris=("pris", "median"),
           Median_kr_m2=("kr_pr_m2", "median"),
           Median_boligareal=("byg039bygningenssamledeboligareal", "median"),
           Median_grundareal=("grundareal_m2", "median"),
           Median_alder=("boligalder", "median"),
           Andel_fritliggende=("boligtype", lambda s: (s == "Fritliggende").mean()))
      .reindex(REGION_RAEKKEFOELGE))
print(t2.round(2).to_string())
gem_tabel(t2.round(3), "tabel_2_region.csv")

mærker = ["A", "B", "C", "D", "E", "F", "G", "Intet"]
t3 = (hp.groupby("energimaerke")
      .agg(Handler=("pris", "size"), Median_kr_m2=("kr_pr_m2", "median"),
           Median_alder=("boligalder", "median"),
           Median_boligareal=("byg039bygningenssamledeboligareal", "median"))
      .reindex(mærker))
t3.insert(1, "Andel", t3["Handler"] / t3["Handler"].sum())
print(t3.round(3).to_string())
gem_tabel(t3.round(3), "tabel_3_energimaerke.csv")

# =============================================================================
# KONTROL AF PUNKTER FRA PARKERINGSLISTEN
# =============================================================================
overskrift("Kontrol: små grundarealer og 0 værelser")
for typ in ["Fritliggende", "Række/kæde/dobbelt"]:
    s = df.loc[df["boligtype"] == typ, "grundareal_m2"]
    print(f"{typ:20s} grundareal < 50 m²: {dk((s < 50).sum()):>7}  "
          f"< 100 m²: {dk((s < 100).sum()):>7}  median: {dk(s.median())} m²")
print("Handler med 0 værelser:", dk((df["sum_vaerelser"] == 0).sum()))

# =============================================================================
# FIGUR 1: PRISUDVIKLING PR. REGION
# =============================================================================
overskrift("Figurer")
kv = df[df["kvartal"] <= SIDSTE_HELE_KVARTAL]
udv = kv.groupby(["kvartal", "region"])["kr_pr_m2"].median().unstack()[REGION_RAEKKEFOELGE]
x = pd.PeriodIndex(udv.index, freq="Q").to_timestamp(how="end")

fig, ax = plt.subplots(figsize=(BREDDE, 3.4))
for r in REGION_RAEKKEFOELGE:
    ls, mk = REGION_STIL[r]
    ax.plot(x, udv[r], color=REGION_FARVE[r], lw=2, ls=ls, marker=mk, markersize=3.5,
            markevery=4, label=r)
ax.axvline(pd.Timestamp(f"{HOVEDPERIODE_FRA}-01-01"), color=MUTED, lw=1, ls="--")
ax.text(pd.Timestamp(f"{HOVEDPERIODE_FRA}-02-01"), ax.get_ylim()[1], "Hovedperiode →",
        va="top", fontsize=8, color=MUTED)
ax.set_ylabel("Median kr. pr. m² (nominelt)")
ax.yaxis.set_major_formatter(FMT_TAL)
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Median pris pr. m² boligareal pr. kvartal, fordelt på region", loc="left")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=60))
gem(fig, "fig_1_prisudvikling_region.png")

# =============================================================================
# FIGUR 2: KORT PR. KOMMUNE
# =============================================================================
kom = gpd.read_file(MAPPE / "kommuner.geojson").to_crs(25832)
# Bornholm og Christiansø flyttes tættere på resten af landet (almindelig praksis på
# danmarkskort). Det ændrer kun placeringen på kortet, ikke data.
oe = kom["kommunekode"].isin([400, 411])
kom.loc[oe, "geometry"] = kom.loc[oe, "geometry"].translate(xoff=-142_000, yoff=190_000)

t4 = (hp.groupby("kommuneinddelingkommunekode")
      .agg(Kommune=("kommunenavn", "first"), Region=("region", "first"),
           Handler=("pris", "size"), Median_pris=("pris", "median"),
           Median_kr_m2=("kr_pr_m2", "median")))
vg = df[(df["salgsaar"] == VURDERINGSAAR_KORT) & (df["vurd_matcher_salgsaar"] == 1)
        & (df["vurderingsgab"] > 0)]
t4 = t4.join(vg.groupby("kommuneinddelingkommunekode")
             .agg(Handler_vurd=("vurderingsgab", "size"),
                  Median_vurderingsgab=("vurderingsgab", "median")))
gem_tabel(t4.round(3), "tabel_4_kommune.csv")
print("Kommuner uden match til kortet:",
      sorted(set(t4.index) - set(kom["kommunekode"])) or "ingen")

kort = kom.merge(t4, left_on="kommunekode", right_index=True, how="left")


def tegn_kort(kolonne, n_kolonne, cmap, norm, titel, label, fmt, navn):
    fig, ax = plt.subplots(figsize=(BREDDE, BREDDE * 0.78))
    ok = kort[n_kolonne] >= MIN_HANDLER_KOMMUNE
    kort[~ok].plot(ax=ax, color="#f0efec", edgecolor="white", linewidth=0.4, hatch="////")
    kort[ok].plot(ax=ax, column=kolonne, cmap=cmap, norm=norm, edgecolor="white",
                  linewidth=0.4)
    # ramme om det flyttede Bornholm
    x0, y0, x1, y1 = kom[oe].total_bounds
    ax.add_patch(plt.Rectangle((x0 - 8000, y0 - 8000), x1 - x0 + 16000, y1 - y0 + 16000,
                               fill=False, ec=MUTED, lw=0.6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label(label, color=INK2)
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.outline.set_visible(False)
    # Forklaringen vises kun, hvis der er synligt skraverede kommuner, dvs. kommuner med
    # mellem 1 og 29 handler. Christiansø har ingen handler og er for lille til at ses (rettet 19/11)
    faa = ~ok & (kort[n_kolonne].fillna(0) > 0)
    print(f"  {navn}: kommuner med 1-{MIN_HANDLER_KOMMUNE - 1} handler: {int(faa.sum())}")
    if faa.any():
        ax.legend(handles=[Patch(facecolor="#f0efec", edgecolor=MUTED, hatch="////",
                                 label=f"Under {MIN_HANDLER_KOMMUNE} handler")],
                  loc="upper right", fontsize=8)
    # Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
    # ax.set_title(titel, loc="left")
    ax.set_axis_off()
    gem(fig, navn)


v = kort.loc[kort["Handler"] >= MIN_HANDLER_KOMMUNE, "Median_kr_m2"]
tegn_kort("Median_kr_m2", "Handler", BLAA_RAMPE,
          plt.Normalize(v.min(), v.max()),
          f"Median pris pr. m² boligareal pr. kommune, {HOVEDPERIODE_FRA}–2026",
          "Kr. pr. m²", FMT_TAL, "fig_2a_kort_kr_pr_m2.png")

kort["Handler_vurd"] = kort["Handler_vurd"].fillna(0)
g = kort.loc[kort["Handler_vurd"] >= MIN_HANDLER_KOMMUNE, "Median_vurderingsgab"]
spaend = max(abs(g.min() - 1), abs(g.max() - 1))
tegn_kort("Median_vurderingsgab", "Handler_vurd", DIVERGERENDE,
          TwoSlopeNorm(vcenter=1, vmin=1 - spaend, vmax=1 + spaend),
          f"Salgspris i forhold til offentlig vurdering, salg i {VURDERINGSAAR_KORT}\n"
          f"(median af salgspris / ejendomsværdi; 1 = salg til vurderingen)",
          "Salgspris / vurdering", FuncFormatter(lambda x, _: dk(x, 2)),
          "fig_2b_kort_vurderingsgab.png")

# =============================================================================
# FIGUR 3: LOG(PRIS) MOD LOG(BOLIGAREAL)
# =============================================================================
x3 = hp["ln_boligareal"]
y3 = hp["ln_pris"]
haeldning, konstant = np.polyfit(x3, y3, 1)

fig, ax = plt.subplots(figsize=(BREDDE, 3.8))
hb = ax.hexbin(x3, y3, gridsize=70, cmap=BLAA_RAMPE, mincnt=5, bins="log", linewidths=0)
# gennemsnitlig log-pris i 20 lige store grupper af boligareal (binned scatter)
grupper = pd.qcut(x3, 20)
bin_x = x3.groupby(grupper, observed=True).mean()
bin_y = y3.groupby(grupper, observed=True).mean()
ax.plot(bin_x, bin_y, "o", color="#eb6834", markersize=5, mec="white", mew=1,
        label="Gennemsnit i 20 arealgrupper")
xs = np.linspace(x3.min(), x3.max(), 50)
ax.plot(xs, konstant + haeldning * xs, color=INK, lw=1.5,
        label=f"Simpel OLS-linje (hældning = {dk(haeldning, 2)})")
ax.set_xlabel("ln(boligareal i m²)")
ax.set_ylabel("ln(salgspris i kr.)")
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title(f"Salgspris og boligareal på log-skala, {HOVEDPERIODE_FRA}–2026", loc="left")
cb = fig.colorbar(hb, ax=ax, pad=0.01)
cb.set_label("Antal handler (log-skala)", color=INK2)
cb.ax.yaxis.set_major_formatter(FMT_TAL)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x, 1)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: dk(x, 0)))
cb.outline.set_visible(False)
ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.9)   # hvid baggrund, så teksten ikke ligger direkte på punkterne
gem(fig, "fig_3_pris_areal.png")
print(f"  Simpel elasticitet af pris mht. boligareal (ingen kontroller): {haeldning:.3f}")

# =============================================================================
# FIGUR 4: BOXPLOTS
# =============================================================================
def boxplot(grupper, data, titel, xlabel, navn):
    vaerdier = [data.loc[data["gruppe"] == g, "kr_pr_m2"].dropna() for g in grupper]
    fig, ax = plt.subplots(figsize=(BREDDE, 3.4))
    bp = ax.boxplot(vaerdier, whis=(5, 95), showfliers=False, patch_artist=True,
                    widths=0.6, medianprops=dict(color=INK, lw=1.8),
                    boxprops=dict(facecolor=BLAA_LYS, edgecolor=BLAA, lw=1),
                    whiskerprops=dict(color=BLAA, lw=1), capprops=dict(color=BLAA, lw=1))
    lange = max(len(str(g)) for g in grupper) > 5
    # Antal handler står i akseteksten under hver kategori, så det ikke ligger oven på
    # x-aksen og knurhårene (rettet 19/11)
    n_tekst = [f"n={dk(len(v) / 1000, 0)}k" for v in vaerdier]
    etiketter = [f"{g} ({n})" if lange else f"{g}\n{n}" for g, n in zip(grupper, n_tekst)]
    ax.set_xticks(range(1, len(grupper) + 1), etiketter, rotation=30 if lange else 0,
                  ha="right" if lange else "center")
    ax.set_ylabel("Kr. pr. m² boligareal")
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(FMT_TAL)
    # Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
    # ax.set_title(titel, loc="left")
    ax.grid(axis="x", visible=False)
    gem(fig, navn)


hp["gruppe"] = hp["energimaerke"]
boxplot(mærker, hp, f"Pris pr. m² fordelt på energimærke, {HOVEDPERIODE_FRA}–2026\n"
        "Boks: 25–75 %-fraktil, streg: median, knurhår: 5–95 %-fraktil", "Energimærke ved handlen",
        "fig_4a_boxplot_energimaerke.png")

perioder = ["før 1900", "1900–29", "1930–49", "1950–59", "1960–69", "1970–79", "1980–89",
            "1990–99", "2000–09", "2010+"]
hp["gruppe"] = hp["opfoerelsesperiode"]
boxplot(perioder, hp, f"Pris pr. m² fordelt på opførelsesperiode, {HOVEDPERIODE_FRA}–2026\n"
        "Boks: 25–75 %-fraktil, streg: median, knurhår: 5–95 %-fraktil", "Opførelsesperiode",
        "fig_4b_boxplot_opfoerelse.png")

# =============================================================================
# FIGUR 8: ANTAL HANDLER PR. KVARTAL
# =============================================================================
antal = df[df["kvartal"] <= UFULDSTAENDIGT_KVARTAL].groupby("kvartal").size()
x8 = pd.PeriodIndex(antal.index, freq="Q").to_timestamp()
fig, ax = plt.subplots(figsize=(BREDDE, 3.0))
farver = [BLAA if d.year >= HOVEDPERIODE_FRA else BLAA_LYS for d in x8]
ufuld = np.array([k == UFULDSTAENDIGT_KVARTAL for k in antal.index])
ax.bar(x8[~ufuld], antal.values[~ufuld], width=80, color=np.array(farver)[~ufuld], align="edge")
ax.bar(x8[ufuld], antal.values[ufuld], width=80, color="white", edgecolor=BLAA, hatch="////",
       linewidth=0.8, align="edge")
if UFULDSTAENDIGT_KVARTAL in antal.index:
    print(f"Handler i {UFULDSTAENDIGT_KVARTAL} (ufuldstændigt): {dk(antal[UFULDSTAENDIGT_KVARTAL])}   "
          f"gennemsnit af de 4 foregående kvartaler: {dk(antal.iloc[-5:-1].mean())}")
ax.set_ylabel("Antal handler")
ax.yaxis.set_major_formatter(FMT_TAL)
# Titel fjernet (vejledermøde 28/10) – står i figurteksten i Word:
# ax.set_title("Antal handler pr. kvartal i det rensede datasæt", loc="left")
ax.legend(handles=[Patch(color=BLAA_LYS, label="2015–2018 (robusthed)"),
                   Patch(color=BLAA, label=f"{HOVEDPERIODE_FRA}+ (hovedperiode)"),
                   Patch(facecolor="white", edgecolor=BLAA, hatch="////",
                         label="Ufuldstændigt kvartal (tinglysningsforsinkelse)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5)
ax.grid(axis="x", visible=False)
gem(fig, "fig_8_handler_kvartal.png")

print(f"\nAlle filer ligger i: {UD}")
