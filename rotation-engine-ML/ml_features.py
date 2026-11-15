"""
ml_features.py - builds the feature row the Roots of Tomorrow ML model (Notebook B v2) needs for any location,
so the rotation engine can call ml_adjust.SoilAdjuster.ph_delta().

Inputs (same files the engine already loads):
  dt_dir   : unzipped dt_sets.zip   (fao_gaez/..., isric_soilgrids/...)
  mli_dir  : unzipped ml_inputs.zip (power_climate_features.csv, gaez_yxx_yield_wide.csv)
  meta     : model_metadata_v2.json (p99 yields used to turn yield into the 0-100 index)

Usage:
    from ml_features import FeatureBuilder
    from ml_adjust import SoilAdjuster
    fb = FeatureBuilder(dt_dir, mli_dir, 'model_metadata_v2.json')
    adj = SoilAdjuster('roots_suitability_model_v2.joblib', 'model_metadata_v2.json')
    feats, sat_ph, info = fb.features(lat, lon)
    delta = adj.ph_delta(feats_for(crop...)...)   # see adjust_for_farmer below
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
W = np.array([5.0, 10.0, 15.0])


class FeatureBuilder:
    def __init__(self, dt_dir, mli_dir, meta_path, k=5):
        self.meta = json.load(open(meta_path))
        self.k = k
        g = pd.read_csv(os.path.join(dt_dir, 'fao_gaez', 'GAEZ_v5_12_Crops_Global_Sample_2001_2020.csv'),
                        usecols=['cell_id', 'latitude', 'longitude']).drop_duplicates('cell_id').set_index('cell_id')
        s = pd.read_csv(os.path.join(dt_dir, 'isric_soilgrids', 'SoilGrids_GAEZ_85857_Locations_3_Depths.csv'),
                        low_memory=False).set_index('cell_id').reindex(g.index)

        def dm(prefix):
            m = s[[f'{prefix}_{d}' for d in ('0_5cm', '5_15cm', '15_30cm')]].to_numpy(float)
            ok = ~np.isnan(m)
            out = np.where(ok.sum(1) >= 2, (np.where(ok, m, 0) @ W) / np.maximum(ok @ W, 1e-9), np.nan)
            return out
        soil = pd.DataFrame({'soil_ph': dm('ph'), 'soil_clay_pct': dm('clay_pct'), 'soil_sand_pct': dm('sand_pct'),
                             'soil_silt_pct': dm('silt_pct'), 'soil_oc_g_kg': dm('organic_carbon_g_kg'),
                             'soil_n_g_kg': dm('total_nitrogen_g_kg')}, index=g.index)
        clim = pd.read_csv(os.path.join(mli_dir, 'power_climate_features.csv')).set_index('cell_id').reindex(g.index)
        self.cells = g.join(soil).join(clim)
        y = pd.read_csv(os.path.join(mli_dir, 'gaez_yxx_yield_wide.csv')).set_index('cell_id').reindex(g.index)
        self.cols = [(c, m) for c in self.meta['crops'] for m in self.meta['managements']]
        p99 = np.array([self.meta['p99_kg_ha'][f'{c}|{m}'] for c, m in self.cols])
        self.idx = np.clip(100 * y[[f'{c}|{m}' for c, m in self.cols]].to_numpy(float) / p99, 0, 100)
        self.tree = BallTree(np.radians(g[['latitude', 'longitude']].to_numpy()), metric='haversine')

    def features(self, lat, lon, crop, mgmt):
        """Feature dict for one crop x management at a location, plus the satellite pH and some info."""
        d, i = self.tree.query(np.radians([[lat, lon]]), k=self.k + 1)
        d, i = d[0] * EARTH_KM, i[0]
        own = i[0]
        row = self.cells.iloc[own]
        f = {k: (float(v) if pd.notna(v) else np.nan) for k, v in row.items() if k not in ('latitude', 'longitude')}
        f['abs_lat'] = abs(lat)
        j = self.cols.index((crop, mgmt))
        nb_i, nb_d = i[1:], d[1:]                         # neighbours other than the location's own cell
        vals = self.idx[nb_i, j]
        w = np.where(np.isnan(vals), 0, 1 / (nb_d + 10))
        f['nb1_index'] = float(vals[0]) if not np.isnan(vals[0]) else np.nan
        f['nb1_dist_km'] = float(nb_d[0])
        f['nbk_index'] = float(np.nansum(np.nan_to_num(vals) * w) / w.sum()) if w.sum() > 0 else np.nan
        f['nbk_dist_km'] = float(nb_d.mean())
        sat_ph = f.get('soil_ph')
        return f, (None if sat_ph is None or np.isnan(sat_ph) else sat_ph), {'cell_distance_km': float(d[0])}


def adjust_for_farmer(fb, adjuster, lat, lon, crops, mgmt, farmer_ph):
    """Change in the 0-100 suitability index per crop caused by the farmer's soil pH (0 if no pH given)."""
    out = {}
    for c in crops:
        f, sat_ph, _ = fb.features(lat, lon, c, mgmt)
        out[c] = adjuster.ph_delta(f, [c], mgmt, sat_ph, farmer_ph)[c]
    return out
