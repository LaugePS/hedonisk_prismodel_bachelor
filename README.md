# Hvad betaler markedet for?
**En hedonisk prisanalyse af danske enfamiliehuse – og af den offentlige ejendomsvurdering**

Kildekode til bachelorprojektet i Erhvervsøkonomi med tilvalg, Aarhus BSS, Aarhus Universitet (afleveret 1. december 2026).
Forfatter: Lauge P. Skodborggaard. Vejleder: Jonas Søndergaard Sørensen.

Repository'et indeholder alle Python-scripts, der ligger bag opgavens tal, tabeller og figurer. Hvert script kan køres for sig i den rækkefølge, der er angivet nedenfor, og skriver sine resultater til mappen `figurer_og_tabeller/`.

---

## Indhold
1. [Formål](#formål)
2. [Data](#data)
3. [Mappestruktur](#mappestruktur)
4. [Installation](#installation)
5. [Kørsel – rækkefølge](#kørsel--rækkefølge)
6. [Scripts](#scripts)
7. [Fra script til opgave – figurer og tabeller](#fra-script-til-opgave--figurer-og-tabeller)
8. [Kontrol og test](#kontrol-og-test)
9. [Pakkeversioner](#pakkeversioner)
10. [Brug af kunstig intelligens](#brug-af-kunstig-intelligens)

---

## Formål
Opgaven undersøger:

> *Hvilke boligkarakteristika er forbundet med salgsprisen på danske enfamiliehuse, og hvordan forholder en hedonisk prismodel baseret på registerdata sig til den offentlige ejendomsvurdering – både i prissætningen af boligens karakteristika og i evnen til at forudsige salgspriser?*

Analysen består af fem dele:
- **Hovedmodel (prisdrivere):** log-lineær hedonisk prismodel med postnummer- og kvartalsfaste effekter (M1–M4) estimeret med OLS.
- **Beliggenhed:** udvidelse med afstand til kysten og bebyggelsestæthed (G1–G4).
- **Regionale forskelle:** interaktioner mellem region og boligareal, energimærke og oliefyr med F-test.
- **Vurderingsgabet:** markedets og vurderingens prissætning af de samme karakteristika (dekomposition, LPM og logit).
- **Prædiktion:** tidstro sammenligning af en registerdatamodel (fem trin op til gradient boosting med koordinater og naboernes prisniveau) med vurderingen, med bootstrap-konfidensintervaller, samt korrektion og kombination af vurderingen.

---

## Data
**Rådata ligger ikke i repository'et.** Data er et udtræk fra SEGES Innovation fra de officielle registre på Datafordeleren (Ejerfortegnelsen, BBR, Matriklen, Danmarks Adresseregister, Ejendomsvurderingsregistret og DAGI) suppleret med energimærker fra Energistyrelsen. Udtrækket dækker handler fra 1. januar 2015 til oktober 2026 (774.528 rækker, 656.584 handler). Data indeholder ikke personoplysninger, men filerne er for store til GitHub og er ikke offentligt tilgængelige.

For at køre scripts skal følgende filer ligge i `data/`:

| Fil | Indhold | Dannes af |
|---|---|---|
| Rå CSV-fil fra SEGES (oktober 2026) | Handler beriget med BBR, DAR, MAT, VUR og energimærker | Udleveret af SEGES Innovation |
| `ejendomshandler_energi (4).csv` | Supplerende udtræk (27/10-2026) med bygningens koordinat og afstand til kysten | Udleveret af SEGES Innovation |
| `kommuner.geojson` | Kommunegrænser (DAGI via [Neogeografen/dagi](https://github.com/Neogeografen/dagi)), forenklet | Ligger i repository'et |
| `data_renset.csv`, `data_renset_inkl_ekstreme.csv`, `data_flere_bygninger.csv`, `frasorteringstabel.csv` | Renset datasæt (558.494 handler, heraf 385.334 fra 2019) | `rensning.py` |
| `geografi_kolonner.csv` | Kystafstand, koordinater og tæthedsindeks koblet på det rensede datasæt | `geografi.py` |
| `forudsigelser_tidstro.csv`, `forudsigelser_geo.csv` | Forudsigelser fra de tidstro prædiktionsmodeller | `praediktion_tidstro.py`, `praediktion_geo.py` |

CSV-format i alle filer: separator `;`, decimaltegn `,`, encoding UTF-8 med BOM (`utf-8-sig`), så filerne åbner direkte i Excel.

---

## Mappestruktur
```
bachelorprojekt/
├── scripts/                 # alle Python-scripts (dette repository)
├── data/                    # rådata og mellemfiler (ikke i repository'et, undtagen kommuner.geojson)
└── figurer_og_tabeller/     # output: figurer (PNG, 300 dpi) og tabeller (CSV)
```
Alle scripts finder selv stierne ud fra denne struktur, så ingen stier skal rettes.

---

## Installation
Python 3.13 og pakkerne i `requirements.txt`:
```bash
pip3 install -r requirements.txt
```
`pyfixest` bruges til modellerne med faste effekter. `geopandas` bruges kun til kortene i `deskriptiv.py`.

---

## Kørsel – rækkefølge
```bash
cd scripts
python3 diagnose_2.py            # valgfri: diagnose af rådata (kun udskrift)
python3 rensning.py
python3 deskriptiv.py
python3 hovedmodel.py
python3 geografi.py
python3 regional.py
python3 robusthed.py
python3 vurderingsgab.py
python3 praediktion.py
python3 praediktion_v2.py
python3 praediktion_tidstro.py
python3 praediktion_geo.py       # kræver geografi_kolonner.csv (geografi.py) – ca. 15–30 min.
python3 vurdering_korrigeret.py  # kræver forudsigelser_geo.csv (praediktion_geo.py)
python3 robusthed_benyttelse.py  # kræver forudsigelser_tidstro.csv
python3 figurer_praediktion.py   # laver figurer ud fra gemte tabeller, kører ingen modeller
```
`faelles.py` køres ikke selv. Den indeholder den fælles dataforberedelse, referencekategorier og hjælpefunktioner, så alle modeller bygger på præcis de samme variable og den samme stikprøve.

---

## Scripts

| Script | Formål | Vigtigste output |
|---|---|---|
| `diagnose_2.py` | Diagnose af rådata: struktur, datoer, koder og manglende værdier | Kun udskrift i terminalen |
| `rensning.py` | Rensning i 12 trin (fri handel, én ejendom og pris, bygningsstatus "opført", én bygning, enfamiliehuse, grænser for ekstreme huse, trimning af kr./m² pr. år) og afledte variable | `data_renset.csv`, `frasorteringstabel.csv` |
| `faelles.py` | Fælles dataforberedelse (`forbered_data`), referencekategorier (energimærke D, opført 1960–69, fjernvarme, mursten, tegl) og hjælpefunktioner | – |
| `deskriptiv.py` | Nøgletal, prisudvikling, kort og rå sammenhænge | Tabel 1–4, fig_1–4, fig_8 |
| `hovedmodel.py` | Modeltrappen M1–M4 (OLS med postnummer- og kvartalsfaste effekter), robuste og klyngede standardfejl, VIF | Tabel 5–7, fig_5 |
| `geografi.py` | Kystafstand i intervaller og som logaritme (G1–G4), tæthedsindeks, dekomposition af postnummer-effekterne, månedsfaste effekter og sæson | Tabel 27–32, fig_18, fig_19, `geografi_kolonner.csv` |
| `regional.py` | Interaktioner med region og F-test af ens effekter | Tabel 8–9, fig_6 |
| `robusthed.py` | Robusthedstjek R0–R6 og standardfejl på tre måder | Tabel 12–13, fig_9 |
| `vurderingsgab.py` | Dekomposition af markedets og vurderingens prissætning samt LPM og logit for salg over vurderingen | Tabel 14–16, fig_10–12 |
| `praediktion.py` | Første prædiktion: M4 og random forest mod vurderingen | Tabel 10–11, fig_7a–7b |
| `praediktion_v2.py` | Vej-effekt med shrinkage, tidligere salgspris og gradient boosting (trænet på hele perioden) | Tabel 17–19, fig_13–14 |
| `praediktion_tidstro.py` | Tidstro design: én model pr. testår trænet kun på salg før vurderingsdatoen, varianter a/b/c og bootstrap | Tabel 20–22, fig_15, `forudsigelser_tidstro.csv` |
| `praediktion_geo.py` | Prædiktionstrappen trin 1–5: + koordinater og kystafstand (trin 4) og naboernes prisniveau (trin 5), bootstrap mod vurderingen | Tabel 33–36, fig_20, `forudsigelser_geo.csv` |
| `vurdering_korrigeret.py` | Korrektion af vurderingen for energimærket (OLS og median) og kombination af vurdering og trin 5-model | Tabel 23–25, fig_16 |
| `robusthed_benyttelse.py` | Robusthedstjek kun for vurderingens benyttelseskode 1 (beboelse) | Tabel 26 |
| `figurer_praediktion.py` | Figurer ud fra gemte tabeller | fig_13 (rettet), fig_17 |

---

## Fra script til opgave – figurer og tabeller
Numrene i opgaven svarer ikke til filnavnene. Oversigt over hovedteksten:

| I opgaven | Fil | Script |
|---|---|---|
| Figur 1 | `fig_1_prisudvikling_region.png` | `deskriptiv.py` |
| Figur 2 | `fig_2a_kort_kr_pr_m2.png` | `deskriptiv.py` |
| Figur 3 | `fig_3_pris_areal.png` | `deskriptiv.py` |
| Figur 4 | `fig_4a_boxplot_energimaerke.png` | `deskriptiv.py` |
| Figur 5 | `fig_5_koefficientplot.png` | `hovedmodel.py` |
| Figur 6 | `fig_18` (kystpræmie) | `geografi.py` |
| Figur 7 | `fig_6_regionale_forskelle.png` | `regional.py` |
| Figur 8 | `fig_11_marked_vs_vurdering.png` | `vurderingsgab.py` |
| Figur 9 | `fig_20` (prædiktionstrappe) | `praediktion_geo.py` |
| Figur 10 | `fig_16_korrigeret_vurdering.png` | `vurdering_korrigeret.py` |
| Figur 11 | `fig_9_robusthed.png` | `robusthed.py` |
| Tabel 1 | `frasorteringstabel.csv` | `rensning.py` |
| Tabel 2 | `tabel_1_deskriptiv.csv` | `deskriptiv.py` |
| Tabel 3 | `tabel_5_regression.csv` | `hovedmodel.py` |
| Tabel 4 | `tabel_8_regionale_effekter.csv` | `regional.py` |
| Tabel 5 | `tabel_14_dekomposition.csv` | `vurderingsgab.py` |
| Tabel 6 | `tabel_15_lpm_logit.csv` | `vurderingsgab.py` |
| Tabel 7 | `tabel_33`–`tabel_35` | `praediktion_geo.py` |
| Tabel 8 | `tabel_23_korrigeret_vurdering.csv`, `tabel_25_korrigeret_bootstrap.csv` | `vurdering_korrigeret.py` |

Øvrige tabeller og figurer indgår i bilagene.

---

## Kontrol og test
- **Syntetiske data:** alle scripts er testet på syntetiske data med kendt facit, før de blev kørt på de rigtige data. Fx genfinder dekompositionen i `vurderingsgab.py` sammenhængen β̂(pris) − β̂(vurdering) = β̂(gab) med en afvigelse på 10⁻¹⁶, og F-testene i `regional.py` afviser ikke hypotesen om ens effekter, når de sande effekter er ens.
- **Ingen datalækage i prædiktionen:** vej-effekt, naboniveau, prisindeks og tidligere salg beregnes kun på salg før vurderingsdatoen, og husets egne salg indgår aldrig i dets vej-effekt eller naboniveau (enhedstestet i `praediktion_geo.py`). Vurderingen bruges aldrig som forklarende variabel i den uafhængige model.
- **Egen implementering af faste effekter:** kystmodellerne i `geografi.py` genskaber hovedmodellens koefficienter fra `pyfixest` med en afvigelse på højst 0,00005.
- **Indbyggede kontroller:** `praediktion_geo.py` genfinder MdAPE for trin 1–3 og vurderingen fra `praediktion_tidstro.py` (variant b: 17,6 %, 15,0 %, 13,2 % og 12,5 %).
- **Faste seeds:** test- og kalibreringshandler trækkes med fast seed, så alle resultater kan genskabes.

---

## Pakkeversioner
| Pakke | Version |
|---|---|
| Python | 3.13 |
| pandas | [udfyld] |
| numpy | [udfyld] |
| matplotlib | [udfyld] |
| geopandas | [udfyld] |
| pyfixest | [udfyld] |
| statsmodels | [udfyld] |
| scipy | [udfyld] |
| scikit-learn | [udfyld] |

Versionerne findes med:
```bash
python3 --version
pip3 show pandas numpy matplotlib geopandas pyfixest statsmodels scipy scikit-learn | grep -E "Name|Version"
```

---

## Brug af kunstig intelligens
AI-assistenten Claude (Anthropic) er brugt som hjælp til programmeringen af scripts. Alle analysevalg er truffet af forfatteren, og alle tal i opgaven er kontrolleret mod scriptenes output. Se afsnit 3.6 i opgaven.
