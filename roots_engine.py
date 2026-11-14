"""
Roots of Tomorrow - site profile + crop-rotation engine (thin slice, v1)

Farmer answers (dict, keys as in the input form specification) -> up to 4 ranked
3-season rotation plans with benefits, trade-offs, assumptions and a confidence level.

Data used (folders as in dt_sets.zip):
  fao_gaez/GAEZ_v5_12_Crops_Global_Sample_2001_2020.csv   suitability classes 1-8, 12 crops, 4 management levels
  isric_soilgrids/SoilGrids_GAEZ_85857_Locations_3_Depths.csv   soil at the same 85,857 cells
  nasa_gpm/NASA_GPM_50_Sites_Monthly_2015_2025.csv        monthly rainfall at 50 sites (rainfall variability, fallback)
  ml_inputs/power_climate_monthly.csv (from Notebook A)    NASA POWER monthly temperature and rainfall 2001-2020 at every cell
Knowledge base (folder kb/): crop_catalog.csv, rotation_rules.csv, sources.json

Everything here is decision support. The rotation points are ranking heuristics, not yield forecasts.
"""
from __future__ import annotations
import itertools, json, math, os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

KB_VERSION = "starter-0.2 (unreviewed by agronomists)"
EARTH_KM = 6371.0088
MGMT = ["HRLM", "HILM", "LRLM", "LILM"]          # rainfed-high, irrigated-high, rainfed-low, irrigated-low
CLASS_SCORE = {1: 100, 2: 85, 3: 70, 4: 55, 5: 40, 6: 25, 7: 10, 8: 0}   # ordinal mapping, not calibrated
FALLOW = "fallow"
FALLOW_NAME = "Fallow (rest the land or grow a cover crop)"
FARMER_EVIDENCE_FLOOR = 60.0      # crops the farmer already grows are not scored below this (rule R11)
DIVERSITY_PENALTY = 3.0           # composite points when a plan repeats a crop (rule R13)
GOALS = ["water_saving", "soil_health", "risk_reduction", "income", "fallow_use", "food_security"]
GOAL_LABEL = {
    "water_saving": "Most water-efficient", "soil_health": "Best for soil health",
    "risk_reduction": "Lowest production risk", "income": "Highest indicative income",
    "fallow_use": "Makes the most of your land", "food_security": "Strongest for household food supply"}
PRIORITY_W = {"not_important": 0, "slightly_important": 1, "important": 2, "very_important": 3}
WATER_VALUES = {"rainfall_only", "limited_irrigation", "reliable_irrigation", "unknown"}
LOGGING_VALUES = {"weeks", "days", "none", "unknown"}
INPUT_VALUES = {"minimal", "moderate", "high", "unknown"}
TEXTURE_VALUES = {"clay", "loam", "sandy", "silt", "unknown"}
NONCROP = {"fallow", "other", "unknown"}
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ALIASES = {"rice": "wetland_rice", "paddy": "wetland_rice", "bean": "phaseolus_bean", "beans": "phaseolus_bean",
           "common_bean": "phaseolus_bean", "millet": "pearl_millet", "peanut": "groundnut", "pigeon_pea": "pigeonpea",
           "sweetpotato": "sweet_potato", "corn": "maize"}
DISCLAIMER = ("These recommendations are decision-support guidance based on regional data. "
              "They do not replace advice from a local agricultural extension officer.")


# ----------------------------------------------------------------------------------------------
# Knowledge base
# ----------------------------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, kb_dir):
        self.catalog = pd.read_csv(os.path.join(kb_dir, "crop_catalog.csv")).set_index("crop_id")
        for c in ("texture_good", "texture_poor"):
            self.catalog[c] = self.catalog[c].fillna("").astype(str)
        self.rules = pd.read_csv(os.path.join(kb_dir, "rotation_rules.csv")).set_index("rule_id")
        self.sources = json.load(open(os.path.join(kb_dir, "sources.json")))
        self.crops = list(self.catalog.index)

    def pts(self, rule_id):
        return float(self.rules.loc[rule_id, "points"])

    def src(self, ids):
        return [{"id": i, "cite": self.sources[i]["cite"], "url": self.sources[i]["url"]} for i in ids]


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
class RootsData:
    def __init__(self, data_dir, kb: KnowledgeBase, power_dir=None):
        g = pd.read_csv(os.path.join(data_dir, "fao_gaez", "GAEZ_v5_12_Crops_Global_Sample_2001_2020.csv"))
        cols = [f"{c}_class" for c in kb.crops]
        missing = [c for c in cols if c not in g.columns]
        if missing:
            raise ValueError(f"GAEZ file lacks columns for catalog crops: {missing}")
        cells = g["cell_id"].drop_duplicates().tolist()
        pos = pd.Series(np.arange(len(cells)), index=cells)
        self.classes = np.full((len(cells), 4, len(cols)), 8, dtype=np.int8)
        mi = g["management_code"].map({m: i for i, m in enumerate(MGMT)}).values
        self.classes[pos[g["cell_id"]].values, mi, :] = g[cols].values
        meta = g.drop_duplicates("cell_id").set_index("cell_id").loc[cells]
        self.cell_ids = np.array(cells)
        self.lat = meta["latitude"].values
        self.lon = meta["longitude"].values
        self.origin = meta["sample_origin"].values
        self.tree = BallTree(np.radians(np.c_[self.lat, self.lon]), metric="haversine")

        s = pd.read_csv(os.path.join(data_dir, "isric_soilgrids", "SoilGrids_GAEZ_85857_Locations_3_Depths.csv"),
                        low_memory=False).set_index("cell_id").loc[cells]
        self.soil_status = s["soil_status"].values
        w = np.array([5.0, 10.0, 15.0])

        def wmean(prefix, unit):
            m = np.c_[[s[f"{prefix}_{d}{unit}"].values for d in ("0_5cm", "5_15cm", "15_30cm")]].T
            ok = ~np.isnan(m)
            num = np.where(ok, m, 0) @ w
            den = ok @ w
            out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            out[ok.sum(1) < 2] = np.nan            # require at least two of the three depths
            return out
        self.soil = pd.DataFrame({
            "ph": wmean("ph", ""), "clay": wmean("clay_pct", ""), "sand": wmean("sand_pct", ""),
            "silt": wmean("silt_pct", ""), "oc_g_kg": self._oc(s, w),
            "n_g_kg": self._n(s, w)}, index=cells)
        self.soil_ok = (~self.soil[["ph", "clay", "sand", "silt"]].isna().any(axis=1)).values
        ok_idx = np.where(self.soil_ok)[0]
        self._soil_ok_idx = ok_idx
        self._soil_tree = BallTree(np.radians(np.c_[self.lat[ok_idx], self.lon[ok_idx]]), metric="haversine") \
            if len(ok_idx) else None

        gp = pd.read_csv(os.path.join(data_dir, "nasa_gpm", "NASA_GPM_50_Sites_Monthly_2015_2025.csv"))
        gp = gp[gp["data_status"] == "available"]
        self.gpm_sites = gp.drop_duplicates("site_id").set_index("site_id")[["country", "latitude", "longitude"]]
        mm = gp.groupby(["site_id", "month_number"])["precipitation_mm"]
        self.gpm_mean = mm.mean().unstack().reindex(columns=range(1, 13))
        self.gpm_sd = mm.std().unstack().reindex(columns=range(1, 13))
        self.gpm_nyears = gp.groupby("site_id")["year"].nunique()
        self.gpm_tree = BallTree(np.radians(self.gpm_sites[["latitude", "longitude"]].values), metric="haversine")

        # NASA POWER monthly climatology at every cell (output of Notebook A); optional
        self.has_power = False
        cands = [power_dir] if power_dir else []
        cands += [os.path.join(data_dir, "ml_inputs"), os.path.join(os.path.dirname(os.path.abspath(data_dir)), "ml_inputs")]
        for d in cands:
            f = os.path.join(d, "power_climate_monthly.csv") if d else None
            if f and os.path.exists(f):
                pw = pd.read_csv(f).set_index("cell_id").reindex(cells)
                self.pow_T = pw[[f"T2M_{k:02d}" for k in range(1, 13)]].to_numpy(float)
                self.pow_P = pw[[f"PRECIP_MM_{k:02d}" for k in range(1, 13)]].to_numpy(float)
                self.has_power = True
                self.power_source = f
                break

    @staticmethod
    def _oc(s, w):
        m = np.c_[[s[f"organic_carbon_g_kg_{d}"].values for d in ("0_5cm", "5_15cm", "15_30cm")]].T
        ok = ~np.isnan(m)
        den = ok @ w
        out = np.where(den > 0, (np.where(ok, m, 0) @ w) / np.where(den > 0, den, 1), np.nan)
        out[ok.sum(1) < 2] = np.nan
        return out

    @staticmethod
    def _n(s, w):
        m = np.c_[[s[f"total_nitrogen_g_kg_{d}"].values for d in ("0_5cm", "5_15cm", "15_30cm")]].T
        ok = ~np.isnan(m)
        den = ok @ w
        out = np.where(den > 0, (np.where(ok, m, 0) @ w) / np.where(den > 0, den, 1), np.nan)
        out[ok.sum(1) < 2] = np.nan
        return out

    def nearest_cells(self, lat, lon, k=1):
        d, i = self.tree.query(np.radians([[lat, lon]]), k=k)
        return d[0] * EARTH_KM, i[0]

    def cells_within(self, lat, lon, km):
        i, d = self.tree.query_radius(np.radians([[lat, lon]]), r=km / EARTH_KM, return_distance=True)
        return i[0], d[0] * EARTH_KM

    def nearest_gpm(self, lat, lon):
        d, i = self.gpm_tree.query(np.radians([[lat, lon]]), k=1)
        return self.gpm_sites.index[i[0][0]], float(d[0][0] * EARTH_KM)


# ----------------------------------------------------------------------------------------------
# Input normalisation
# ----------------------------------------------------------------------------------------------
def _choice(v, allowed, default="unknown"):
    if v is None:
        return default
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in allowed else default


def _month(v):
    try:
        m = int(v)
        return m if 1 <= m <= 12 else None
    except (TypeError, ValueError):
        return None


def _crop(v, kb):
    if v is None:
        return "unknown"
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    s = ALIASES.get(s, s)
    if s in kb.crops or s in NONCROP:
        return s
    return "other"


def normalise_input(f, kb):
    n = {}
    loc = f.get("location") or {}
    n["lat"] = float(loc.get("lat", loc.get("latitude")))
    n["lon"] = float(loc.get("lon", loc.get("longitude")))
    if not (-90 <= n["lat"] <= 90 and -180 <= n["lon"] <= 180):
        raise ValueError("location out of range")
    n["current_crop"] = _crop(f.get("current_crop"), kb)
    n["previous_crop"] = _crop(f.get("previous_crop"), kb)
    n["previous_crop_2"] = _crop(f.get("previous_crop_2"), kb)
    hm = f.get("harvest_month")
    n["harvested"] = isinstance(hm, str) and hm.strip().lower() in ("already_harvested", "harvested")
    n["harvest_month"] = _month(hm)
    n["next_planting_month"] = _month(f.get("next_planting_month"))
    sp = f.get("seasons_per_year")
    n["seasons_per_year"] = int(sp) if str(sp) in ("1", "2", "3") else None
    n["water_access"] = _choice(f.get("water_access"), WATER_VALUES)
    n["waterlogging"] = _choice(f.get("waterlogging"), LOGGING_VALUES)
    n["input_level"] = _choice(f.get("input_level"), INPUT_VALUES)
    n["soil_texture"] = _choice(f.get("soil_texture"), TEXTURE_VALUES)
    try:
        ph = float(f.get("soil_ph")) if f.get("soil_ph") not in (None, "") else None
        n["soil_ph"] = ph if ph is not None and 3.0 <= ph <= 10.0 else None
    except (TypeError, ValueError):
        n["soil_ph"] = None
    ex = f.get("excluded_crops") or []
    n["excluded_crops"] = sorted({_crop(c, kb) for c in ex} & set(kb.crops))
    pr = f.get("priorities") or {}
    n["priorities"] = {g: _choice(pr.get(g), set(PRIORITY_W), "important") for g in GOALS}
    return n


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def texture_class(sand, clay, silt):
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), (sand, clay, silt))):
        return "unknown"
    if clay >= 35:
        return "clay"
    if sand >= 60 or (sand >= 50 and clay < 15):
        return "sandy"
    if silt >= 50 and clay < 27:
        return "silt"
    return "loam"


def rain_in_window(clim, start_month, dur_days, start_day=15):
    total, m, day, left = 0.0, start_month, start_day, dur_days
    while left > 0:
        md = DAYS[(m - 1) % 12]
        take = min(md - day + 1, left)
        total += clim[(m - 1) % 12] * take / md
        left -= take
        m = m % 12 + 1
        day = 1
    return total


def temp_in_window(temps, start_month, dur_days, start_day=15):
    """Day-weighted mean of monthly mean temperature over the growing window."""
    tot, w, m, day, left = 0.0, 0.0, start_month, start_day, dur_days
    while left > 0:
        md = DAYS[(m - 1) % 12]
        take = min(md - day + 1, left)
        tot += temps[(m - 1) % 12] * take
        w += take
        left -= take
        m = m % 12 + 1
        day = 1
    return tot / w


def rainy_blocks(clim, thr=100.0):
    wet = [x >= thr for x in clim]
    if all(wet):
        return [(0, 12)]
    if not any(wet):
        return []
    blocks = []
    for s0 in range(12):
        if wet[s0] and not wet[s0 - 1]:
            length = 0
            while wet[(s0 + length) % 12]:
                length += 1
            blocks.append((s0, length))
    return blocks


_LUT = np.array([CLASS_SCORE.get(i, 0) for i in range(9)], dtype=float)


def _bilinear(cls_arr, a, b):
    """cls_arr: (4, ncrops) class codes; a = irrigation weight, b = input weight -> scores (ncrops,)"""
    sc = _LUT[np.asarray(cls_arr, dtype=int)]
    rh, ih, rl, il = sc[0], sc[1], sc[2], sc[3]
    return (1 - a) * (1 - b) * rl + (1 - a) * b * rh + a * (1 - b) * il + a * b * ih


def _scenario_weights(water, inp):
    a = {"rainfall_only": (0, 0, 0), "limited_irrigation": (0.5, 0, 1), "reliable_irrigation": (1, 1, 1),
         "unknown": (0, 0, 1)}[water]
    b = {"minimal": (0, 0, 0), "moderate": (0.5, 0, 1), "high": (1, 1, 1), "unknown": (0.5, 0, 1)}[inp]
    return a, b


# ----------------------------------------------------------------------------------------------
# Site profile
# ----------------------------------------------------------------------------------------------
def build_site_profile(n, data: RootsData, kb: KnowledgeBase, as_of_month):
    warnings, notes = [], []
    d, idx = data.nearest_cells(n["lat"], n["lon"], k=1)
    i0, dist = int(idx[0]), float(d[0])
    if dist > 90:
        raise ValueError(f"No suitability data within 90 km of this location (nearest cell {dist:.0f} km away).")
    if dist > 40:
        warnings.append(f"The nearest suitability cell is {dist:.0f} km from your pin (cells are about 55 km apart); "
                        f"suitability may differ locally.")
    site = {"gaez_cell_id": str(data.cell_ids[i0]), "gaez_cell_lat": float(data.lat[i0]),
            "gaez_cell_lon": float(data.lon[i0]), "distance_to_cell_km": round(dist, 1),
            "gaez_classes_by_management": {}}
    cls = data.classes[i0].astype(int)                       # (4, ncrops)
    for mi, m in enumerate(MGMT):
        site["gaez_classes_by_management"][m] = {c: int(cls[mi, j]) for j, c in enumerate(kb.crops)}

    # soil
    soil = data.soil.iloc[i0]
    soil_src = "SoilGrids 5 km, 0-30 cm depth-weighted, at the suitability cell"
    soil_dist = dist
    if not data.soil_ok[i0] and data._soil_tree is not None:
        dd, ii = data._soil_tree.query(np.radians([[data.lat[i0], data.lon[i0]]]), k=1)
        if dd[0][0] * EARTH_KM <= 60:
            j = data._soil_ok_idx[ii[0][0]]
            soil = data.soil.iloc[j]
            soil_dist = float(dd[0][0] * EARTH_KM)
            soil_src = f"SoilGrids 5 km borrowed from a neighbouring cell {soil_dist:.0f} km away (own cell has no complete soil data)"
            notes.append(soil_src)
        else:
            soil_src = "no complete SoilGrids data within 60 km"
    sat_tex = texture_class(soil.get("sand"), soil.get("clay"), soil.get("silt")) if data.soil_ok.any() else "unknown"
    site["soil"] = {"ph": None if pd.isna(soil["ph"]) else round(float(soil["ph"]), 2),
                    "clay_pct": None if pd.isna(soil["clay"]) else round(float(soil["clay"]), 1),
                    "sand_pct": None if pd.isna(soil["sand"]) else round(float(soil["sand"]), 1),
                    "silt_pct": None if pd.isna(soil["silt"]) else round(float(soil["silt"]), 1),
                    "organic_carbon_g_kg": None if pd.isna(soil["oc_g_kg"]) else round(float(soil["oc_g_kg"]), 1),
                    "total_nitrogen_g_kg": None if pd.isna(soil["n_g_kg"]) else round(float(soil["n_g_kg"]), 2),
                    "texture_class": sat_tex, "source": soil_src}
    site["_soil_ok"] = bool(not pd.isna(soil["ph"]))

    # climate: NASA POWER at the cell (everywhere); GPM for month-to-month variability where a site is near
    gs, gd = data.nearest_gpm(n["lat"], n["lon"])
    clim, temps = None, None
    cv = [None] * 12
    if gd <= 100:
        gm = data.gpm_mean.loc[gs].values.astype(float)
        gsd = data.gpm_sd.loc[gs].values.astype(float)
        cv = [round(float(s_ / m_), 2) if m_ > 1 else None for m_, s_ in zip(gm, gsd)]
    if data.has_power and np.isfinite(data.pow_P[i0]).all():
        clim = data.pow_P[i0].astype(float)
        temps = data.pow_T[i0].astype(float)
        src = "NASA POWER (MERRA-2) monthly climatology 2001-2020 at the suitability cell"
        if gd <= 100:
            src += f"; variability from GPM IMERG site {gs} ({gd:.0f} km away)"
    elif gd <= 100:
        clim = gm
        src = f"GPM IMERG monthly {data.gpm_nyears[gs]} years at site {gs} ({gd:.0f} km away)"
    if clim is not None:
        site["rainfall"] = {"source": src, "monthly_mean_mm": [round(float(x), 1) for x in clim], "monthly_cv": cv,
                            "annual_mm": round(float(clim.sum()), 0), "wet_months_ge_100mm": int((clim >= 100).sum())}
    else:
        warnings.append("No rainfall data for this location, so rainy-season fit is not checked.")
        site["rainfall"] = None
    if temps is not None:
        site["temperature"] = {"source": "NASA POWER (MERRA-2) 2 m air temperature, monthly climatology 2001-2020",
                               "monthly_mean_c": [round(float(x), 1) for x in temps],
                               "annual_mean_c": round(float(temps.mean()), 1)}
    else:
        site["temperature"] = None
        warnings.append("No temperature data loaded (ml_inputs.zip missing), so crop temperature limits are not checked.")
    site["_clim"] = clim
    site["_temp"] = temps

    # neighbourhood heterogeneity (how much suitability changes between sample cells nearby)
    ii, dd = data.cells_within(n["lat"], n["lon"], 85)
    nb = [int(j) for j in ii if int(j) != i0]
    het = None
    if nb:
        a, b = _scenario_weights(n["water_access"], n["input_level"])
        s0 = _bilinear(cls, a[0], b[0])
        diffs = []
        for j in nb:
            sj = _bilinear(data.classes[j].astype(int), a[0], b[0])
            act = (s0 > 0) | (sj > 0)
            if act.any():
                diffs.append(np.abs(s0[act] - sj[act]).mean() / 15.0)
        het = float(np.median(diffs)) if diffs else 0.0
    site["neighbourhood_variation_classes"] = None if het is None else round(het, 2)
    if het is not None and het >= 1.5:
        warnings.append("Suitability changes a lot between the surrounding sample cells, so this area is heterogeneous; "
                        "treat the scores as regional.")
    return site, warnings, notes, cls, sat_tex, i0


# ----------------------------------------------------------------------------------------------
# Season set-up and per-season, per-crop evaluation
# ----------------------------------------------------------------------------------------------
def plan_seasons(n, site, as_of_month, assumptions):
    clim = site["_clim"]
    irrig = n["water_access"] in ("reliable_irrigation",)
    spy = n["seasons_per_year"]
    if spy is None:
        if irrig:
            spy = 2
        elif clim is not None:
            blocks = rainy_blocks(clim)
            spy = int(np.clip(sum(max(1, int(b[1] // 3.5)) for b in blocks), 1, 3)) if blocks else 1
        else:
            spy = 1
        assumptions.append(f"Seasons per year not given; assumed {spy} "
                           f"({'reliable irrigation' if irrig else 'from the local rainfall pattern' if clim is not None else 'no rainfall data, default'}).")
    start = n["next_planting_month"]
    if start is None:
        est_basis = "the month after today"
        if clim is not None and not irrig:
            wet = [x >= 100 for x in clim]
            for k in range(1, 13):
                m = (as_of_month - 1 + k) % 12
                if wet[m] and not wet[m - 1]:
                    start = m + 1
                    est_basis = "the start of the next rainy period"
                    break
        if start is None:
            start = as_of_month % 12 + 1
        assumptions.append(f"Next planting month not given; estimated as {MONTH_NAME[start - 1]} ({est_basis}).")
    gap = 12 / spy
    seasons = []
    for k in range(3):
        m = (start - 1 + int(round(k * gap))) % 12 + 1
        seasons.append({"index": k + 1, "plant_month": m, "length_days": gap * 30.4})
    return spy, seasons


def evaluate(n, site, cls, sat_tex, kb, seasons, minscore):
    """Return table[k][crop] = dict(valid, reason, score, low, high, adj, rain_ratio, rain_status, ...)"""
    a, b = _scenario_weights(n["water_access"], n["input_level"])
    s_c = _bilinear(cls, a[0], b[0])
    s_lo = np.minimum(_bilinear(cls, a[1], b[1]), s_c)
    s_hi = np.maximum(_bilinear(cls, a[2], b[2]), s_c)
    clim = site["_clim"]
    temps = site.get("_temp")
    hist = {n["current_crop"], n["previous_crop"], n["previous_crop_2"]} & set(kb.crops)
    sat_ph = site["soil"]["ph"]
    farmer_ph = n["soil_ph"]
    farmer_tex = n["soil_texture"]
    water = n["water_access"]
    table = []
    for ss in seasons:
        row = {}
        for j, c in enumerate(kb.crops):
            cat = kb.catalog.loc[c]
            dmid = (cat.duration_min_d + cat.duration_max_d) / 2
            room = ss["length_days"] - 14
            if dmid > room >= cat.duration_min_d:
                dmid = room
                dshort = True
            else:
                dshort = False
            e = {"crop": c, "score": float(s_c[j]), "low": float(s_lo[j]), "high": float(s_hi[j]),
                 "adj": 0.0, "valid": True, "reason": [], "notes": [], "rain_ratio": None, "rain_status": "not checked",
                 "duration_days": float(dmid), "nir_mm": None}
            # duration fits season
            if cat.duration_min_d > room:
                e["valid"] = False
                e["reason"].append(f"too long for a {int(round(ss['length_days'] / 30.4))}-month season")
            elif dshort:
                e["notes"].append(f"needs an early-maturing variety (about {int(dmid)} days) to fit the season")
            # waterlogging
            wl = n["waterlogging"]
            tol = cat.waterlog_tol
            if wl == "weeks" and tol == "low":
                e["valid"] = False
                e["reason"].append("standing water for weeks; this crop does not tolerate waterlogging")
            elif wl == "weeks" and tol == "medium":
                e["adj"] -= 12
                e["notes"].append("only moderately tolerant of long standing water")
            elif wl == "days" and tol == "low":
                e["adj"] -= 8
                e["notes"].append("sensitive to a few days of standing water")
            if c == "wetland_rice":
                if wl == "none" and water == "rainfall_only":
                    e["valid"] = False
                    e["reason"].append("wetland rice needs standing water, but the field drains well and has no irrigation")
                elif wl == "none" and water == "limited_irrigation":
                    e["adj"] -= 15
                    e["notes"].append("needs standing water; limited irrigation and good drainage make this hard")
                elif wl == "unknown" and water in ("rainfall_only", "unknown"):
                    e["adj"] -= 8
                    e["notes"].append("needs standing water; not confirmed for this field")
            # rainfall fit
            if clim is not None:
                wmid = (cat.water_need_mm_low + cat.water_need_mm_high) / 2
                p = 0.75 * rain_in_window(clim, ss["plant_month"], dmid)
                ratio = p / wmid
                e["rain_ratio"] = round(float(ratio), 2)
                e["nir_mm"] = round(float(max(0.0, wmid - p)), 0)
                e["rain_status"] = "good" if ratio >= 0.8 else "marginal" if ratio >= 0.5 else "poor"
                rice_flooded = c == "wetland_rice" and n["waterlogging"] == "weeks"
                if not rice_flooded and water != "reliable_irrigation":
                    if e["rain_status"] == "marginal":
                        e["adj"] -= {"rainfall_only": 12, "unknown": 12, "limited_irrigation": 5}[water]
                        e["notes"].append(f"rainfall in the growing window covers about {ratio:.0%} of the crop's water need")
                    elif e["rain_status"] == "poor":
                        if water == "rainfall_only":
                            e["valid"] = False
                            e["reason"].append(f"too dry in the growing window (rain covers about {ratio:.0%} of need)")
                        else:
                            e["adj"] -= 25 if water == "unknown" else 15
                            e["notes"].append(f"rain covers only about {ratio:.0%} of need; depends on irrigation")
            # temperature limits (monthly mean over the growing window; irrigation does not change this)
            e["temp_mean_c"] = None
            if temps is not None:
                tw = temp_in_window(temps, ss["plant_month"], dmid)
                e["temp_mean_c"] = round(float(tw), 1)
                if tw < cat.temp_abs_min:
                    e["valid"] = False
                    e["reason"].append(f"too cold in the growing window (about {tw:.0f} °C average)")
                elif tw > cat.temp_abs_max:
                    e["valid"] = False
                    e["reason"].append(f"too hot in the growing window (about {tw:.0f} °C average)")
                elif tw < cat.temp_opt_min:
                    e["adj"] -= min(20.0, 4.0 * (cat.temp_opt_min - tw))
                    e["notes"].append(f"cooler than ideal in the growing window (about {tw:.1f} °C vs {cat.temp_opt_min:.0f}-{cat.temp_opt_max:.0f} °C)")
                elif tw > cat.temp_opt_max:
                    e["adj"] -= min(20.0, 4.0 * (tw - cat.temp_opt_max))
                    e["notes"].append(f"warmer than ideal in the growing window (about {tw:.1f} °C vs {cat.temp_opt_min:.0f}-{cat.temp_opt_max:.0f} °C)")
            # farmer soil overrides (only the difference from the satellite value counts)
            if farmer_ph is not None:
                def penph(x):
                    if x is None:
                        return 0.0
                    dist = max(cat.ph_opt_min - x, x - cat.ph_opt_max, 0.0)
                    return min(30.0, 12.0 * dist)
                delta = penph(farmer_ph) - (penph(sat_ph) if sat_ph is not None else 0.5 * penph(farmer_ph))
                if abs(delta) >= 0.5:
                    e["adj"] -= delta
                    e["notes"].append(f"your soil pH ({farmer_ph}) vs this crop's preferred {cat.ph_opt_min}-{cat.ph_opt_max}")
            if farmer_tex != "unknown":
                def pentex(t):
                    return 8.0 if t in str(cat.texture_poor).split(";") else 0.0
                delta = pentex(farmer_tex) - (pentex(sat_tex) if sat_tex != "unknown" else 0.5 * pentex(farmer_tex))
                if abs(delta) >= 0.5:
                    e["adj"] -= delta
                    if delta > 0:
                        e["notes"].append(f"{farmer_tex} soil is a poorer match for this crop")
            e["s_adj"] = float(np.clip(e["score"] + e["adj"], 0, 100))
            e["farmer_evidence"] = False
            if c in hist and e["valid"] and e["s_adj"] < FARMER_EVIDENCE_FLOOR:
                e["s_adj"] = FARMER_EVIDENCE_FLOOR
                e["high"] = max(e["high"], FARMER_EVIDENCE_FLOOR)
                e["farmer_evidence"] = True
                e["notes"].append("score raised because you already grow this crop on this field; "
                                  "the regional data may under-rate your area")
            if e["s_adj"] < minscore:
                e["valid"] = False
                e["reason"].append("rated not suitable or too marginal here by the regional model"
                                   + (" (after your soil answers)" if e["adj"] < -0.5 and e["score"] >= minscore else ""))
            row[c] = e
        table.append(row)
    return table


# ----------------------------------------------------------------------------------------------
# Sequence scoring
# ----------------------------------------------------------------------------------------------
def _legume(kb, c):
    return c in kb.catalog.index and int(kb.catalog.loc[c, "n_fixing"]) == 1


def _family(kb, c):
    return kb.catalog.loc[c, "family"] if c in kb.catalog.index else None


def score_sequence(seq, hist, table, n, kb, seasons, minscore, check=True, strict=True):
    """seq: 3 crop ids; hist: [c_minus2, c_minus1, c0] (may be noncrop). Returns dict or None when a hard rule fails."""
    full = list(hist) + list(seq)
    irrigated_rice_ok = n["water_access"] == "reliable_irrigation" or n["waterlogging"] in ("weeks", "days")
    pen_total, credit_total, events = 0.0, 0.0, []
    for k, c in enumerate(seq):
        pos = 3 + k
        prev = full[pos - 1]
        prev2 = full[pos - 2]
        # same crop twice
        if prev == c and c in kb.catalog.index:
            if c == "wetland_rice" and irrigated_rice_ok:
                pen_total += 8
                events.append(("same_crop", k, c, -8))
            elif check:
                return None
            else:
                pen_total += {"soybean": 10, "maize": 5}.get(c, 8)
                events.append(("same_crop", k, c, -{"soybean": 10, "maize": 5}.get(c, 8)))
        elif c in kb.catalog.index and prev in kb.catalog.index:
            if _family(kb, c) == _family(kb, prev):
                pen_total += 5
                events.append(("same_family", k, c, -5))
            if _legume(kb, c) and _legume(kb, prev):
                if check and strict:
                    return None
                pen_total += 6
                events.append(("legume_after_legume", k, c, -6))
        if c in kb.catalog.index and prev2 in kb.catalog.index and prev != c and _family(kb, c) == _family(kb, prev2) \
                and _family(kb, prev) != _family(kb, c):
            pen_total += 2
            events.append(("same_family_2back", k, c, -2))
        # legume credit
        if prev in kb.catalog.index and c in kb.catalog.index and _legume(kb, prev) and not _legume(kb, c):
            mult_in = {"minimal": 1.2, "moderate": 0.8, "high": 0.6, "unknown": 1.0}[n["input_level"]]
            resp = kb.catalog.loc[c, "as_cereal_resp_mult"]
            pre = kb.catalog.loc[prev, "as_legume_pre_mult"]
            pts = kb.pts("R01") * pre * resp * mult_in
            credit_total += pts
            events.append(("legume_credit", k, f"{prev}->{c}", round(pts, 1)))
    real = [(k, c) for k, c in enumerate(seq) if c != FALLOW]
    if len(real) < 2:
        return None                                              # a plan needs at least two cropped seasons
    if len({c for _, c in real}) < 2 and not ({c for _, c in real} == {"wetland_rice"} and irrigated_rice_ok):
        return None                                              # ... and at least two different crops
    rc = [c for _, c in real]
    s_adj = [table[k][c]["s_adj"] for k, c in real]
    agro = float(np.clip(np.mean(s_adj) - pen_total / 3 + credit_total / 3, 0, 100))
    low = float(np.mean([table[k][c]["low"] for k, c in real]))
    share = len(real) / 3.0                                       # fallow seasons produce nothing
    # goals
    goals = {}
    ws = []
    for k, c in enumerate(seq):
        if c == FALLOW:
            ws.append(100.0)
            continue
        cat = kb.catalog.loc[c]
        wmid = (cat.water_need_mm_low + cat.water_need_mm_high) / 2
        if c == "wetland_rice":
            ws.append(10.0)
        elif table[k][c]["nir_mm"] is not None and n["water_access"] != "rainfall_only":
            ws.append(float(np.clip(100 * (1 - table[k][c]["nir_mm"] / 600), 0, 100)))
        else:
            ws.append(float(np.clip(100 * (1 - (wmid - 300) / 600), 0, 100)))
    goals["water_saving"] = float(np.mean(ws))
    legumes = [c for c in rc if _legume(kb, c)]
    credited = sum(1 for e in events if e[0] == "legume_credit")
    fam = {_family(kb, c) for c in rc}
    fam_adj = sum(1 for e in events if e[0] in ("same_family", "same_crop"))
    prev_leg = hist[2] in kb.catalog.index and _legume(kb, hist[2]) and seq[0] != FALLOW and not _legume(kb, seq[0])
    sh = 40 * (1 if (legumes or prev_leg) else 0) + 30 * (1 if credited else 0) \
        + 30 * (len(fam) - 1) / 2.0 - 15 * fam_adj
    goals["soil_health"] = float(np.clip(sh, 0, 100))
    drought = {"high": 100, "medium": 60, "low": 25}
    div = len(set(rc)) / 3.0
    goals["risk_reduction"] = float(np.clip(0.5 * min(s_adj)
                                            + 0.25 * np.mean([drought[kb.catalog.loc[c, "drought_tol"]] for c in rc])
                                            + 0.25 * 100 * div, 0, 100))
    mk = {"high": 100, "medium": 65, "low": 35}
    goals["income"] = float(share * np.mean([mk[kb.catalog.loc[c, "market_class"]] * table[k][c]["s_adj"] / 100
                                             for k, c in real]))
    occ = [0.0 if c == FALLOW else min(1.0, table[k][c]["duration_days"] / seasons[k]["length_days"])
           for k, c in enumerate(seq)]
    goals["fallow_use"] = float(100 * np.mean(occ))
    st = float(share * np.mean([kb.catalog.loc[c, "staple_share"] for c in rc]))
    groups = {kb.catalog.loc[c, "group"] for c in rc}
    goals["food_security"] = float(np.clip(50 * st + 25 * (1 if ("cereal" in groups and "legume" in groups) else 0)
                                           + 25 * share * np.mean(s_adj) / 100, 0, 100))
    w = {g: PRIORITY_W[n["priorities"][g]] for g in GOALS}
    tw = sum(w.values())
    goal_w = sum(w[g] * goals[g] for g in GOALS) / tw if tw > 0 else float(np.mean(list(goals.values())))
    comp = 0.5 * agro + 0.5 * goal_w
    if len(set(rc)) < len(rc):
        comp -= DIVERSITY_PENALTY
        events.append(("repeat_in_plan", 2, "plan repeats a crop", -DIVERSITY_PENALTY))
    return {"seq": list(seq), "agro": agro, "goals": goals, "goal_weighted": goal_w,
            "composite": comp, "events": events, "penalties": pen_total,
            "credit": credit_total, "mean_low": low}


# ----------------------------------------------------------------------------------------------
# Benefits and trade-offs text (every statement carries its evidence level)
# ----------------------------------------------------------------------------------------------
def region_is_ssa(lat, lon):
    return -35 <= lat <= 18 and -18 <= lon <= 52


def describe_plan(sc, table, n, kb, seasons, hist, base, p75):
    seq_all = sc["seq"]
    seq = [c for c in seq_all if c != FALLOW]                    # cropped seasons only
    name = lambda c: FALLOW_NAME if c == FALLOW else kb.catalog.loc[c, "name"]
    ben, trade = [], []
    credits = [e for e in sc["events"] if e[0] == "legume_credit"]
    if credits or (hist[2] in kb.catalog.index and _legume(kb, hist[2]) and seq_all[0] != FALLOW
                    and not _legume(kb, seq_all[0])):
        pairs = ", ".join(e[2].replace("->", " then ").replace("_", " ") for e in credits) or \
            f"{name(hist[2])} (current) then {name(seq_all[0])}"
        text = (f"Legume-to-next-crop sequences in this plan ({pairs}) are the pattern that rotation studies link to higher "
                f"yield in the following crop: 23% on average for legume pre-crops versus 16% for non-legumes in 3,663 field "
                f"comparisons worldwide.")
        cond = ["Average of many trials, not a forecast for your field; roughly one trial in four showed no gain or a loss.",
                "The gain shrinks where nitrogen fertiliser use is high"
                + (" - you reported high input use, so expect less." if n["input_level"] == "high" else ".")]
        srcs = ["S1", "S8"]
        if region_is_ssa(n["lat"], n["lon"]):
            text += " In sub-Saharan Africa, effects were larger after soybean and groundnut than after cowpea, and larger for maize than for millet or sorghum."
            srcs += ["S2", "S3"]
        ben.append({"goal": "soil_health", "title": "Nitrogen and yield carry-over from legumes", "text": text,
                    "evidence_level": "meta-analysis (peer-reviewed)", "conditions": cond, "sources": kb.src(srcs)})
    known = [c for c in seq if c in {n["current_crop"], n["previous_crop"], n["previous_crop_2"]}]
    if known:
        ben.append({"goal": "risk_reduction", "title": "Includes a crop you already grow",
                    "text": (f"{', '.join(sorted({name(c) for c in known}))}: already grown on your field, so seed, skills and "
                             f"market links are in place."),
                    "evidence_level": "farmer experience",
                    "conditions": ["Based on your answers about recent crops."], "sources": []})
    fams = {kb.catalog.loc[c, "family"] for c in seq}
    if len(set(seq)) == 3 and len(seq) == 3:
        ben.append({"goal": "risk_reduction", "title": "Three different crops over three seasons",
                    "text": ("Rotating instead of repeating one crop was linked in a global analysis to 14-27% more yield, "
                             "energy, protein and revenue over whole sequences, and to less year-to-year yield variation."),
                    "evidence_level": "meta-analysis (peer-reviewed)",
                    "conditions": ["Averages from experiments; local results depend on soil, rainfall and management."],
                    "sources": kb.src(["S1"])})
    wl = sc["goals"]["water_saving"]
    mids = {c: (kb.catalog.loc[c, "water_need_mm_low"] + kb.catalog.loc[c, "water_need_mm_high"]) / 2 for c in seq}
    lowest = min(mids, key=mids.get)
    if wl >= max(55, p75["water_saving"]):
        ben.append({"goal": "water_saving", "title": "Lower seasonal water demand",
                    "text": (f"This plan is in the top quarter of the options for your field on water use; "
                             f"{name(lowest)} is its least water-demanding crop "
                             f"(about {int(kb.catalog.loc[lowest, 'water_need_mm_low'])}-{int(kb.catalog.loc[lowest, 'water_need_mm_high'])} mm per season, "
                             f"FAO 1986 ranges)."),
                    "evidence_level": "FAO guideline values (the score itself is a heuristic)",
                    "conditions": ["Seasonal need varies with climate; values are approximate ranges."],
                    "sources": kb.src(["S4"])})
    if sc["goals"]["fallow_use"] >= max(60, p75["fallow_use"]):
        ben.append({"goal": "fallow_use", "title": "Land in use for most of each season",
                    "text": f"Crop durations fill about {sc['goals']['fallow_use']:.0f}% of the available season length.",
                    "evidence_level": "calculation from typical crop durations (starter values)",
                    "conditions": ["Duration values are general starter figures; local varieties differ."], "sources": []})
    groups = {kb.catalog.loc[c, "group"] for c in seq}
    if "cereal" in groups and "legume" in groups:
        ben.append({"goal": "food_security", "title": "Staple grain plus protein-rich legume",
                    "text": ("Combining cereals and legumes supports both energy and protein from your own fields; "
                             "rotation systems in the global analysis produced about 24% more dietary energy and 14% more protein."),
                    "evidence_level": "meta-analysis (peer-reviewed)",
                    "conditions": ["Experimental averages; depends on how much of the harvest your household eats."],
                    "sources": kb.src(["S1"])})
    if sc["goals"]["income"] >= max(45, p75["income"]):
        ben.append({"goal": "income", "title": "Higher-value crops matched to good suitability",
                    "text": "Income is indicative only: it combines how suitable the land is with a rough market class (high, medium, low) for each crop. No local prices are included.",
                    "evidence_level": "heuristic (no price data)", "conditions": ["Check local prices before deciding."], "sources": []})
    # trade-offs
    for k, c in enumerate(seq_all):
        if c == FALLOW:
            trade.append(f"Season {k + 1}: left fallow (no harvest). Rain or temperature in this season suits few crops "
                         f"under your water answer; a short cover crop or green manure could protect the soil.")
            continue
        e = table[k][c]
        if e["rain_status"] == "marginal" and n["water_access"] != "reliable_irrigation":
            tail = "so some dry-spell risk." if n["water_access"] == "rainfall_only" else "so irrigation would need to fill the gap."
            trade.append(f"Season {k + 1} ({name(c)}): rainfall in the growing window covers about {e['rain_ratio']:.0%} of the crop's water need, {tail}")
        if e["s_adj"] < 50:
            trade.append(f"Season {k + 1} ({name(c)}): only moderate suitability here (score {e['s_adj']:.0f}/100).")
        for nt in e["notes"]:
            if any(k in nt for k in ("poorer match", "pH", "waterlogging", "standing water", "tolerant", "than ideal", "early-maturing")):
                trade.append(f"Season {k + 1} ({name(c)}): {nt}.")
    for ev in sc["events"]:
        if ev[0] == "same_family":
            trade.append(f"Season {ev[1] + 1}: {name(ev[2])} follows a crop from the same family, which can carry over pests and diseases (uncited principle).")
        if ev[0] == "legume_after_legume":
            trade.append(f"Season {ev[1] + 1}: a legume follows another legume, so the nitrogen benefit is lost and shared diseases are possible.")
        if ev[0] == "same_crop":
            trade.append(f"Season {ev[1] + 1}: repeats {name(ev[2])} (irrigated paddy rice allows this); yield penalties for repeated crops have been reported.")
    if "cereal" not in groups:
        trade.append("No cereal in this plan; check it fits your household's staple food needs.")
    if "legume" not in groups and not any(e[0] == "legume_credit" for e in sc["events"]):
        trade.append("No legume in this plan, so no nitrogen credit.")
    vs = None
    if base is not None:
        vs = {"current_practice_sequence": base["seq"],
              "composite_change": round(sc["composite"] - base["composite"], 1),
              "goal_changes": {g: round(sc["goals"][g] - base["goals"][g], 1) for g in GOALS},
              "note": "Compared with repeating your current crop for three seasons (scored with the same method)."}
    return ben, trade, vs


# ----------------------------------------------------------------------------------------------
# Confidence
# ----------------------------------------------------------------------------------------------
def confidence(n, site, cls, table, kb, seasons, hist, warnings):
    pts, reasons = 100, []

    def dock(p, why):
        nonlocal pts
        pts -= p
        reasons.append(why)
    if n["water_access"] == "unknown":
        dock(15, "Water source not known: results shown as a range between rain-fed and irrigated.")
    if n["input_level"] == "unknown":
        dock(10, "Input level not known: results shown as a range between low and high input.")
    elif n["input_level"] == "moderate":
        dock(4, "Moderate input use has no matching layer; results sit between the low-input and high-input layers.")
    if n["waterlogging"] == "unknown":
        dock(6, "Standing-water history not known.")
    if n["soil_texture"] == "unknown":
        dock(3, "Soil type not given; satellite soil estimate used.")
    if n["next_planting_month"] is None:
        dock(8, "Planting month not given; estimated from rainfall.")
    if n["seasons_per_year"] is None:
        dock(4, "Number of seasons per year not given; estimated.")
    for k in ("previous_crop", "previous_crop_2"):
        if n[k] in ("unknown",):
            dock(4, f"{k.replace('_', ' ')} not remembered: rotation rules cannot use it.")
    if n["current_crop"] == "unknown":
        dock(6, "Current crop not given.")
    if site["distance_to_cell_km"] > 40:
        dock(10, f"Nearest suitability cell is {site['distance_to_cell_km']:.0f} km away.")
    elif site["distance_to_cell_km"] > 25:
        dock(4, f"Nearest suitability cell is {site['distance_to_cell_km']:.0f} km away (cells are about 55 km apart).")
    if site["neighbourhood_variation_classes"] is not None and site["neighbourhood_variation_classes"] >= 1.5:
        dock(8, "Suitability varies a lot between nearby sample cells.")
    if site["rainfall"] is None:
        dock(15, "No rainfall record nearby: rainy-season fit not checked.")
    if not site["_soil_ok"]:
        dock(6, "No complete satellite soil data for this location.")
    # reality check: does the model rate the farmer's own crops as marginal?
    a, b = _scenario_weights(n["water_access"], n["input_level"])
    sc0 = _bilinear(cls, a[0], b[0])
    own = [c for c in {n["current_crop"], n["previous_crop"], n["previous_crop_2"]} if c in kb.crops]
    low_own = [c for c in own if sc0[kb.crops.index(c)] <= 40]
    if low_own:
        dock(15, "The regional model rates your own recent crop(s) (" + ", ".join(kb.catalog.loc[c, "name"] for c in low_own) +
             ") as marginal or worse for your area. If they do well for you, the model may be under-rating your area, "
             "so their scores were raised to at least 60 and other scores should be treated as conservative.")
        warnings.append("Reality check: " + reasons[-1])
    pts = min(pts, 95)
    if low_own:
        pts = min(pts, 45)             # the data contradict the farmer's own experience: never better than Low
    level = "High" if pts >= 75 else "Medium" if pts >= 50 else "Low"
    return {"level": level, "score": max(pts, 0), "reasons": reasons,
            "note": "Heuristic: points are deducted for each unknown or weak input. Not a statistical probability."}


# ----------------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------------
def recommend(farmer, data: RootsData, kb: KnowledgeBase, as_of_month=None, n_plans=3, max_plans=4):
    if as_of_month is None:
        import datetime
        as_of_month = datetime.date.today().month
    n = normalise_input(farmer, kb)
    assumptions = []
    site, warnings, notes, cls, sat_tex, i0 = build_site_profile(n, data, kb, as_of_month)
    spy, seasons = plan_seasons(n, site, as_of_month, assumptions)
    site["seasons_per_year_used"] = spy
    site["season_plan"] = [{"season": s["index"], "plant_month": MONTH_NAME[s["plant_month"] - 1]} for s in seasons]
    hist = [n["previous_crop_2"], n["previous_crop"], n["current_crop"]]
    if n["water_access"] == "unknown":
        assumptions.append("Water source unknown: the central estimate assumes rain-fed farming; irrigated results are the upper end of the range.")
    if n["input_level"] in ("unknown", "moderate"):
        assumptions.append("Input level treated as halfway between the low-input and high-input suitability layers.")
    if n["soil_texture"] == "unknown":
        assumptions.append(f"Soil type taken from satellite soil ({sat_tex}).")
    if n["input_level"] in ("unknown", "moderate") or n["water_access"] in ("unknown", "limited_irrigation"):
        assumptions.append("Suitability classes are not strictly comparable between management levels (in this dataset the "
                           "high-input class is worse than the low-input class in about 14% of rain-fed crop-cell pairs), "
                           "so blended scenarios are approximate.")
    if n["harvested"] is False and n["harvest_month"] is not None and n["next_planting_month"] is not None:
        gap = (n["next_planting_month"] - n["harvest_month"]) % 12
        if gap > 6:
            warnings.append("Your next planting month is well before the expected harvest; check for relay planting.")

    # candidate crops: relax the suitability floor only if suitability (not field conditions) is what limits the list
    def eligible(t):
        return [{c for c, e in row.items() if e["valid"] and c not in n["excluded_crops"]} for row in t]
    def select(seasons):
        e0 = eligible(evaluate(n, site, cls, sat_tex, kb, seasons, 0))
        fk = [k for k, x in enumerate(e0) if len(x) < 2]                # seasons where almost nothing can grow
        act = [k for k in range(3) if k not in fk] or [0, 1, 2]
        lim = any(len(e0[k]) < 3 for k in act)
        for floor in ((25, 10) if lim else (40, 25, 10)):
            t = evaluate(n, site, cls, sat_tex, kb, seasons, floor)
            okf = eligible(t)
            if all(len(okf[k]) >= 3 for k in act) or (lim and all(len(okf[k]) >= 1 for k in act)):
                return t, okf, floor, fk, lim
        t = evaluate(n, site, cls, sat_tex, kb, seasons, 10)
        return t, eligible(t), 10, fk, lim

    table, ok, chosen_floor, fallow_k, cond_limited = select(seasons)
    if fallow_k and len(fallow_k) < 3 and spy > 1 and 3 - len(fallow_k) < 2:
        # only one croppable season a year: plan the main season over the next three years instead
        good = next(k for k in range(3) if k not in fallow_k)
        m0 = seasons[good]["plant_month"]
        skipped = sorted({MONTH_NAME[seasons[k]["plant_month"] - 1] for k in fallow_k})
        seasons = [{"index": k + 1, "plant_month": m0, "length_days": 12 * 30.4} for k in range(3)]
        spy = 1
        site["seasons_per_year_used"] = 1
        site["season_plan"] = [{"season": k + 1, "plant_month": MONTH_NAME[m0 - 1]} for k in range(3)]
        warnings.append(f"The season planted in {', '.join(skipped)} is too dry, cold or short for most crops under your "
                        f"answers, so the plans cover your main season ({MONTH_NAME[m0 - 1]}) over the next three years. "
                        f"With irrigation, a second season may be possible.")
        table, ok, chosen_floor, fallow_k, cond_limited = select(seasons)
    if cond_limited:
        warnings.append("Only a few crops can be grown under the conditions you described (for example long standing "
                        "water, no irrigation or a short season), so fewer clearly different plans are possible.")
    elif chosen_floor < 40:
        warnings.append(f"Few crops are well suited here, so the minimum suitability was lowered "
                        f"({'marginal' if chosen_floor == 25 else 'very marginal'} land); treat options with caution.")
    if fallow_k and len(fallow_k) < 3:
        for k in fallow_k:
            ok[k] = set(ok[k]) | {FALLOW}
        warnings.append("Season(s) " + ", ".join(str(k + 1) for k in fallow_k) + " are too dry, cold or short for most crops "
                        "under your answers, so leaving the land fallow is included as an option there.")
    for k, row in enumerate(table):
        for c in n["excluded_crops"]:
            if c in row:
                row[c]["valid"] = False
                row[c]["reason"] = ["excluded by the farmer"]

    cands = []
    for strict in (True, False):
        for seq in itertools.product(*[sorted(x) for x in ok]):
            sc = score_sequence(seq, hist, table, n, kb, seasons, chosen_floor, strict=strict)
            if sc is not None:
                cands.append(sc)
        if cands:
            break
    if cands and not strict:
        warnings.append("A legume had to follow another legume in some plans because few other crops are possible here.")
    result = {"knowledge_base_version": KB_VERSION, "farmer_input_normalised": n, "assumptions": assumptions,
              "warnings": warnings, "site_profile": {k: v for k, v in site.items() if not k.startswith("_")}}
    result["site_profile"]["minimum_suitability_score_used"] = chosen_floor
    crops_considered = []
    for c in kb.crops:
        reasons = sorted({r for row in table for r in row[c]["reason"]})
        crops_considered.append({"crop": c, "eligible_in_seasons": [k + 1 for k, row in enumerate(table) if row[c]["valid"]],
                                 "suitability_score": round(table[0][c]["score"], 0),
                                 "why_not": reasons if not any(row[c]["valid"] for row in table) else []})
    result["crops_considered"] = crops_considered
    result["confidence"] = confidence(n, site, cls, table, kb, seasons, hist, warnings)
    result["status"] = "provisional" if result["confidence"]["level"] == "Low" else "ok"
    result["status_note"] = ("Low confidence: treat these as starting points to discuss with a local extension officer, not as firm advice."
                             if result["status"] == "provisional" else "")
    if not cands:
        union = set().union(*ok) - {FALLOW}
        result["plans"] = []
        if not union:
            top = {}
            for c in crops_considered:
                for r in c["why_not"]:
                    top[r] = top.get(r, 0) + 1
            why = sorted(top, key=lambda r: -top[r])[:2]
            result["status"] = "no_suitable_crop"
            result["status_note"] = ("None of the 12 crops in the current list is workable for this field under the answers given"
                                     + (f" (most common reasons: {'; '.join(why)})." if why else ".")
                                     + " Check the water, drainage and exclusion answers, or ask a local extension officer.")
        elif len(union) == 1:
            only = kb.catalog.loc[next(iter(union)), "name"]
            result["status"] = "single_crop_only"
            result["status_note"] = (f"Only {only} is workable under the answers given, and a rotation needs at least two crops. "
                                     f"Relaxing an excluded crop, or confirming irrigation or drainage, may open more options.")
        else:
            result["status"] = "no_valid_rotation"
            result["status_note"] = "The workable crops cannot be combined into a rotation under the rules (no repeats, season fit)."
        # suggest planting months that would open more options (temperature, rain and season length only)
        counts = {}
        for m in range(1, 13):
            trial = [{"index": k + 1, "plant_month": (m - 1 + int(round(k * 12 / spy))) % 12 + 1,
                      "length_days": 12 / spy * 30.4} for k in range(3)]
            tt = evaluate(n, site, cls, sat_tex, kb, trial, 10)
            counts[m] = len({c for c, e in tt[0].items() if e["valid"] and c not in n["excluded_crops"]})
        best_m = max(counts, key=lambda m: (counts[m], -((m - seasons[0]["plant_month"]) % 12)))
        if counts[best_m] >= 2 and best_m != seasons[0]["plant_month"]:
            result["status_note"] += (f" Planting in {MONTH_NAME[best_m - 1]} would allow {counts[best_m]} crops here, "
                                      f"compared with {counts[seasons[0]['plant_month']]} for {MONTH_NAME[seasons[0]['plant_month'] - 1]}.")
            result["suggested_planting_month"] = best_m
        result["warnings"].append("No valid 3-season plan could be built. " + result["status_note"])
        result["disclaimer"] = DISCLAIMER
        return result
    cands.sort(key=lambda s: -s["composite"])
    best = cands[0]
    base = None
    if n["current_crop"] in kb.crops:
        base = score_sequence([n["current_crop"]] * 3, hist, table, n, kb, seasons, chosen_floor, check=False)

    def distinct(sc, picked, max_shared):
        for p in picked:
            if sum(1 for x, y in zip(sc["seq"], p["seq"]) if x == y) >= 2:
                return False
            if len(set(sc["seq"]) & set(p["seq"])) > max_shared:
                return False
        return True
    picked = [dict(best, label="Best match to your priorities", focus="overall")]
    w = {g: PRIORITY_W[n["priorities"][g]] for g in GOALS}
    order = sorted(GOALS, key=lambda g: (-w[g], GOALS.index(g)))
    order = [g for g in order if w[g] > 0] or GOALS
    for max_shared, tol in ((0, 12), (1, 12), (0, 20), (1, 20), (0, 35), (1, 35), (2, 35)):
        if len(picked) >= n_plans and (max_shared > 0 or len(picked) >= max_plans):
            break
        for g in order:
            if len(picked) >= max_plans:
                break
            if any(p["focus"] == g for p in picked):
                continue
            pool = [x for x in cands if x["composite"] >= best["composite"] - tol and distinct(x, picked, max_shared)]
            if not pool:
                continue
            top = max(pool, key=lambda x: 0.5 * x["goals"][g] + 0.5 * x["composite"])
            med = float(np.median([x["goals"][g] for x in cands]))
            lab = GOAL_LABEL[g] if top["goals"][g] >= med else "Alternative option"
            picked.append(dict(top, label=lab, focus=g))
    picked = picked[:1] + sorted(picked[1:], key=lambda x: -x["composite"])
    if len(picked) < n_plans:
        result["warnings"].append(f"Only {len(picked)} clearly different plan(s) could be built under your constraints.")
    p75 = {g: float(np.percentile([x["goals"][g] for x in cands], 75)) for g in GOALS}
    plans = []
    for r, p in enumerate(picked, 1):
        ben, trade, vs = describe_plan(p, table, n, kb, seasons, hist, base, p75)
        seqd = []
        for k, c in enumerate(p["seq"]):
            if c == FALLOW:
                seqd.append({"season": k + 1, "plant_month": MONTH_NAME[seasons[k]["plant_month"] - 1], "crop": FALLOW,
                             "crop_name": FALLOW_NAME, "suitability_score": None, "suitability_range": None,
                             "approx_duration_days": 0, "rain_fit": {"ratio": None, "status": "not checked"},
                             "temp_mean_c": None, "farmer_evidence": False, "gaez_classes_at_site": None,
                             "notes": ["no crop is planned this season"]})
                continue
            e = table[k][c]
            seqd.append({"season": k + 1, "plant_month": MONTH_NAME[seasons[k]["plant_month"] - 1], "crop": c,
                         "crop_name": kb.catalog.loc[c, "name"],
                         "suitability_score": round(e["s_adj"], 0),
                         "suitability_range": [round(e["low"], 0), round(e["high"], 0)],
                         "approx_duration_days": int(e["duration_days"]),
                         "rain_fit": {"ratio": e["rain_ratio"], "status": e["rain_status"]},
                         "temp_mean_c": e.get("temp_mean_c"),
                         "farmer_evidence": e.get("farmer_evidence", False),
                         "gaez_classes_at_site": {m: site["gaez_classes_by_management"][m][c] for m in MGMT},
                         "notes": e["notes"]})
        plans.append({"option": r, "label": p["label"], "sequence": seqd,
                      "scores": {"agronomic_fit": round(p["agro"], 1), "composite": round(p["composite"], 1),
                                 "goals": {g: round(v, 1) for g, v in p["goals"].items()}},
                      "benefits": ben, "trade_offs": trade, "vs_current_practice": vs,
                      "rule_events": [{"rule": e[0], "season": e[1] + 1, "detail": e[2], "points": e[3]} for e in p["events"]]})
    result["plans"] = plans
    result["current_practice_reference"] = None if base is None else {
        "sequence": base["seq"], "composite": round(base["composite"], 1),
        "goals": {g: round(v, 1) for g, v in base["goals"].items()}}
    result["disclaimer"] = DISCLAIMER
    result["sources"] = {k: {"cite": v["cite"], "url": v["url"]} for k, v in kb.sources.items()}
    return result


def format_recommendation(rec):
    """Plain-text summary for notebooks and logs."""
    L = []
    sp = rec["site_profile"]
    L.append(f"Confidence: {rec['confidence']['level']} ({rec['confidence']['score']}/100)")
    L.append(f"Site: cell {sp['gaez_cell_id']} {sp['distance_to_cell_km']} km from pin; soil texture {sp['soil']['texture_class']}, "
             f"pH {sp['soil']['ph']}; rainfall {sp['rainfall']['annual_mm'] if sp['rainfall'] else 'n/a'} mm/yr; "
             f"{sp['seasons_per_year_used']} season(s)/yr, plant "
             + ", ".join(s['plant_month'] for s in sp['season_plan']))
    for p in rec["plans"]:
        seq = " -> ".join(f"{s['crop_name']} ({s['plant_month']})" for s in p["sequence"])
        L.append(f"\nOption {p['option']} - {p['label']}: {seq}   [composite {p['scores']['composite']}]")
        for b in p["benefits"]:
            L.append(f"   + {b['title']} [{b['evidence_level']}]")
        for t in p["trade_offs"][:3]:
            L.append(f"   - {t}")
    for w in rec["warnings"]:
        L.append("WARNING: " + w)
    return "\n".join(L)
