"""
robusthed_benyttelse.py – Robusthedstjek: kun handler med vurderingens benyttelseskode 1 (beboelse)

Vurderingens benyttelseskode følger Danmarks Statistiks klassifikation (v1:1992): 01 beboelse,
02 beboelse og forretning, 05 landbrug, 08 sommerhus, 09 ubebygget areal osv. Er koden fx 09,
er vurderingen af en UBEBYGGET grund, mens salgsprisen er for grund + (nyt) hus. Sammenligningen
med vurderingen er da ikke fair. Her gentages de vigtigste sammenligninger kun for kode 1:

  A) Vurderingsgabet: markedets vs. vurderingens prissætning af energimærket (som tabel 14)
  B) Tidstro præcision (variant c, niveau pr. kvartal): model vs. vurdering (som tabel 22)
  C) Kombination af vurdering og model (som K1 i tabel 23)

Output i figurer_og_tabeller: tabel_26_benyttelseskode.csv

Kræver: forudsigelser_tidstro.csv (fra praediktion_tidstro.py) og data_renset.csv.
faelles.py i samme mappe. Stier findes automatisk: scripts i bachelorprojekt/scripts, data i bachelorprojekt/data. Tager 1–2 minutter.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from faelles import dk, forbered_data, overskrift

# =============================================================================
# INDSTILLINGER
# =============================================================================
MAPPE = Path(__file__).resolve().parent.parent / "data"   # bachelorprojekt/data (scripts ligger i bachelorprojekt/scripts)
UD = MAPPE.parent / "figurer_og_tabeller"                     # bachelorprojekt/figurer_og_tabeller
UD.mkdir(exist_ok=True)
SEED = 42
B = 1000
KODE_NAVNE = {1: "Beboelse", 2: "Beboelse og forretning", 3: "Forretning", 5: "Landbrug, bebygget",
              8: "Sommerhus", 9: "Ubebygget areal", 11: "Kommunal beboelse/forretning",
              12: "Anden kommunal", 13: "Anden vurdering", 33: "Private institutioner",
              34: "Visse erhvervsejendomme"}
EM = ["A", "B", "C", "E", "F", "G", "Intet"]


def abs_pct(pred, pris):
    return np.abs(100 * (pred - pris) / pris)


def bootstrap_diff(f1, f2, B=B, seed=SEED, blok=100):
    """Bootstrap af MdAPE(f1) − MdAPE(f2) på de samme gentrukne handler."""
    rng = np.random.default_rng(seed)
    n, ud = len(f1), []
    for start in range(0, B, blok):
        idx = rng.integers(0, n, size=(min(blok, B - start), n), dtype=np.int32)
        ud.append(np.median(f1[idx], axis=1) - np.median(f2[idx], axis=1))
    d = np.concatenate(ud)
    return np.quantile(d, 0.025), np.quantile(d, 0.975)


# =============================================================================
# DATA
# =============================================================================
fil = MAPPE / "forudsigelser_tidstro.csv"
if not fil.exists():
    raise SystemExit("forudsigelser_tidstro.csv mangler – kør praediktion_tidstro.py først.")
pred = pd.read_csv(fil, sep=";", decimal=",", encoding="utf-8-sig")
df, bolig = forbered_data(MAPPE)
if "vur_benyttelsekode" not in df.columns:
    raise SystemExit("data_renset.csv mangler kolonnen vur_benyttelsekode.")
samml = df[df["salgsaar"].isin([2020, 2022]) & (df["vurd_matcher_salgsaar"] == 1)
           & (df["ejendomvaerdibeloeb"] > 0)].copy()
samml["ln_vurd"] = np.log(samml["ejendomvaerdibeloeb"])
samml["ln_gab"] = samml["ln_pris"] - samml["ln_vurd"]
samml["kode"] = samml["vur_benyttelsekode"]

# =============================================================================
# FORDELING PÅ BENYTTELSESKODE
# =============================================================================
overskrift("Sammenlignelige handler (2020/2022) pr. benyttelseskode")
fordeling = (samml.groupby("kode", dropna=False)
             .agg(Handler=("pris", "size"), Median_pris=("pris", "median"),
                  Median_salgspris_vurdering=("vurderingsgab", "median"))
             .sort_values("Handler", ascending=False))
fordeling.insert(0, "Betydning", [KODE_NAVNE.get(int(k), "Ukendt") if pd.notna(k) else "Mangler"
                                  for k in fordeling.index])
fordeling["Andel"] = fordeling["Handler"] / fordeling["Handler"].sum()
vis = fordeling.copy().astype(object)
vis["Handler"] = fordeling["Handler"].map(lambda x: dk(x))
vis["Median_pris"] = fordeling["Median_pris"].map(lambda x: dk(x))
vis["Median_salgspris_vurdering"] = fordeling["Median_salgspris_vurdering"].map(lambda x: dk(x, 3))
vis["Andel"] = fordeling["Andel"].map(lambda x: f"{dk(100 * x, 2)} %")
print(vis.to_string())

kode1 = samml[samml["kode"] == 1]
print(f"\nHandler med kode 1: {dk(len(kode1))} af {dk(len(samml))} "
      f"({100 * len(kode1) / len(samml):.1f} %)")

# =============================================================================
# A) VURDERINGSGABET – ENERGIMÆRKET (SOM TABEL 14)
# =============================================================================
overskrift("A) Dekomposition: marked vs. vurdering for energimærket – alle vs. kun kode 1")
rk = {}
for navn, data in [("Alle", samml), ("Kun kode 1", kode1)]:
    mp = pf.feols(f"ln_pris ~ {bolig} | kvartal + postnr", data=data, vcov={"CRV1": "postnr"}).coef()
    mv = pf.feols(f"ln_vurd ~ {bolig} | kvartal + postnr", data=data, vcov={"CRV1": "postnr"}).coef()
    for e in EM:
        v = f"energimaerke::{e}"
        rk.setdefault(f"Energimærke {e}", {})[f"Marked – {navn}"] = 100 * (np.exp(mp[v]) - 1)
        rk[f"Energimærke {e}"][f"Vurdering – {navn}"] = 100 * (np.exp(mv[v]) - 1)
        rk[f"Energimærke {e}"][f"Gab – {navn}"] = 100 * (np.exp(mp[v] - mv[v]) - 1)
tA = pd.DataFrame(rk).T
print(tA.round(1).to_string())

# =============================================================================
# B) TIDSTRO PRÆCISION OG C) KOMBINATION
# =============================================================================
overskrift("B–C) Tidstro præcision (niveau pr. kvartal) – alle vs. kun kode 1")
d = samml.merge(pred[["behandlingsid", "test", "p_gb"]], on="behandlingsid", how="inner")
d = d.dropna(subset=["p_gb"])
d["ln_gb"] = np.log(d["p_gb"])
rk = []
for navn, sub in [("Alle", d), ("Kun kode 1", d[d["kode"] == 1])]:
    kal, test = sub[~sub["test"]].copy(), sub[sub["test"]].copy()
    for kol, pkol in [("vurd", "ejendomvaerdibeloeb"), ("gb", "p_gb")]:
        f = (kal["pris"] / kal[pkol]).groupby(kal["kvartal"]).median()
        test[f"p_{kol}_c"] = test[pkol] * test["kvartal"].map(f)
    k1 = pf.feols("ln_pris ~ ln_vurd + ln_gb | kvartal", data=kal, vcov="hetero")
    test["p_k1"] = np.exp(k1.predict(newdata=test))
    test = test.dropna(subset=["p_vurd_c", "p_gb_c", "p_k1"])
    fv = abs_pct(test["p_vurd_c"], test["pris"]).to_numpy()
    fg = abs_pct(test["p_gb_c"], test["pris"]).to_numpy()
    fk = abs_pct(test["p_k1"], test["pris"]).to_numpy()
    lo1, hi1 = bootstrap_diff(fg, fv)
    lo2, hi2 = bootstrap_diff(fk, fv)
    rk.append(pd.Series({
        "Testhandler": len(test),
        "MdAPE vurdering": np.median(fv), "MdAPE model": np.median(fg), "MdAPE kombination": np.median(fk),
        "Model − vurdering, pp": np.median(fg) - np.median(fv), "KI nedre (model)": lo1, "KI øvre (model)": hi1,
        "Kombination − vurdering, pp": np.median(fk) - np.median(fv), "KI nedre (komb.)": lo2,
        "KI øvre (komb.)": hi2,
        "Vægt vurdering": k1.coef()["ln_vurd"], "Vægt model": k1.coef()["ln_gb"]}, name=navn))
tB = pd.DataFrame(rk)
vis = tB.round(2).astype(object)
vis["Testhandler"] = tB["Testhandler"].map(lambda x: dk(x))
print(vis.T.to_string())
print("\nNegativ forskel = mere præcis end vurderingen. Intervaller er 95 %-bootstrap-KI.")

ud = pd.concat({"A_energimaerke": tA, "B_praecision": tB.T}, axis=0)
ud.to_csv(UD / "tabel_26_benyttelseskode.csv", sep=";", decimal=",", encoding="utf-8-sig")
print("  gemt: tabel_26_benyttelseskode.csv")
