# src/model.py
import os
import json
import math
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

# Use the same feature builders as training/live
from scraper import get_probable_pitchers, build_all_features
from pybaseball import cache as pb_cache
pb_cache.enable()  # leverage local cache

#Paths
ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
RAW_DIR = os.path.join(ROOT, "data", "raw")
HIST_DIR = os.path.join(ROOT, "data", "historical")
PRED_DIR = os.path.join(ROOT, "predictions")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_CSV = os.path.join(RAW_DIR, "today_pitcher_features.csv")
PROBABLES_CSV = os.path.join(RAW_DIR, "today_probables.csv")
MODEL_PKL = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_JSON = os.path.join(MODELS_DIR, "feature_list.json")

# Parsing & math utils
def _safe_float(x) -> Optional[float]:
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x.strip("%")) / 100.0
        return float(x)
    except Exception:
        return None

def _parse_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Parse % strings
    for col in ["K%", "BB%"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float)

    # Numeric coercions
    for c in [
        "recent_csw", "recent_swstr", "recent_pitches", "recent_games",
        "K%", "BB%", "FIP", "xFIP", "SIERA",
        "opp_k_vs_hand_14d", "opp_k_vs_hand_ytd"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categorical → numeric helpers
    if "home_away" in df.columns:
        df["home_away"] = df["home_away"].fillna("away").str.lower().map({"home": 1, "away": 0}).astype(int)
    if "throws" in df.columns:
        df["throws"] = df["throws"].fillna("").replace("", np.nan)
        df["throws_R"] = (df["throws"] == "R").astype(int)
        df["throws_L"] = (df["throws"] == "L").astype(int)

    # Fill sensible defaults
    df["recent_games"] = df["recent_games"].fillna(0)
    df["recent_pitches"] = df["recent_pitches"].fillna(0)
    if "recent_csw" in df:
        df["recent_csw"] = df["recent_csw"].fillna(df["recent_csw"].median(skipna=True))
    if "recent_swstr" in df:
        df["recent_swstr"] = df["recent_swstr"].fillna(df["recent_swstr"].median(skipna=True))
    for c in ["K%", "FIP", "xFIP", "SIERA"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median(skipna=True))

    for c in ["opp_k_vs_hand_14d", "opp_k_vs_hand_ytd"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median(skipna=True) if df[c].notna().any() else 0.23)

    return df

def _poisson_tail_prob(lam: float, k: int) -> float:
    """P(X >= k) for Poisson(lam)."""
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    p = math.exp(-lam)
    cdf = 0.0 if k == 0 else p
    for i in range(1, k):
        p = p * lam / i
        cdf += p
    return 1.0 - cdf

def _poisson_quantile(lam: float, q: float) -> int:
    if lam <= 0:
        return 0
    cum = 0.0
    p = math.exp(-lam)
    k = 0
    while cum + p < q and k < 100:
        cum += p
        k += 1
        p = p * lam / k
    return k

#Baseline (fallback)
def _estimate_ip(recent_pitches: float, recent_games: float) -> float:
    avg_pitches = (recent_pitches / max(recent_games, 1)) if recent_games and recent_games > 0 else (recent_pitches if recent_pitches > 0 else 85.0)
    ip = avg_pitches / 15.0  # ~15 pitches/IP
    return float(np.clip(ip, 3.5, 7.5))

def _baseline_lambda(row: pd.Series) -> float:
    season_k = row.get("K%", np.nan)
    recent_csw = row.get("recent_csw", np.nan)
    recent_sw = row.get("recent_swstr", np.nan)
    opp14 = row.get("opp_k_vs_hand_14d", np.nan)
    oppytd = row.get("opp_k_vs_hand_ytd", np.nan)
    ip = _estimate_ip(row.get("recent_pitches", 0.0), row.get("recent_games", 0.0))

    season_k = 0.24 if pd.isna(season_k) else float(season_k)
    recent_csw = 0.29 if pd.isna(recent_csw) else float(recent_csw)
    recent_sw = 0.12 if pd.isna(recent_sw) else float(recent_sw)
    opp14 = 0.23 if pd.isna(opp14) else float(opp14)
    oppytd = 0.23 if pd.isna(oppytd) else float(oppytd)

    opp_k = 0.6 * opp14 + 0.4 * oppytd
    csw_boost = 1.0 + (recent_csw - 0.29) * 2.2
    sw_boost  = 1.0 + (recent_sw  - 0.12) * 2.5
    base_k_rate = season_k * (0.85 + 0.6 * (opp_k / 0.23)) * csw_boost * sw_boost

    home_away = row.get("home_away", 0)
    ip_adj = ip * (1.03 if home_away == 1 else 1.0)

    lam = max(0.1, base_k_rate) * ip_adj * 3.0
    return float(np.clip(lam, 0.3, 12.0))

#Trained model loader
def _load_trained() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PKL):
        return None
    try:
        import joblib
        pipe = joblib.load(MODEL_PKL)
        feat_list = None
        if os.path.exists(FEATURES_JSON):
            with open(FEATURES_JSON, "r") as f:
                feat_list = json.load(f)
        return {"pipe": pipe, "features": feat_list}
    except Exception as e:
        print(f"Could not load trained model, using baseline. Reason: {e}")
        return None

#Main
def main():
    # Date to run (today UTC by default so it matches scraper defaults)
    date_str = datetime.now(timezone.utc).date().isoformat()
    feat_date = datetime.fromisoformat(date_str)

    # Build today's features directly (same pipeline + caches as training)
    prob_public, prob_internal = get_probable_pitchers(date_str)
    prob_public.to_csv(PROBABLES_CSV, index=False)

    # You can tune lookback here; 50 starts is a good default for stability
    features, _pmix = build_all_features(
        prob_int=prob_internal,
        lookback_days=30,
        feature_date_utc=feat_date,
        lookback_games=50
    )
    features.to_csv(FEATURES_CSV, index=False)

    feats = _parse_columns(features.copy())

    # Determine slate date
    if "game_date" in feats.columns and feats["game_date"].notna().any():
        slate_date = pd.to_datetime(feats["game_date"].iloc[0]).date().isoformat()
    else:
        slate_date = date_str

    # Save a dated snapshot for backtesting
    feat_snapshot_path = os.path.join(HIST_DIR, f"features_{slate_date}.csv")
    features.to_csv(feat_snapshot_path, index=False)
    print(f"Saved feature snapshot → {feat_snapshot_path}")

    # Pick predictor (trained or baseline)
    trained = _load_trained()
    if trained is not None:
        pipe = trained["pipe"]
        feature_list = trained["features"]
        if feature_list is None:
            non_feats = {"pitcher_name","team","opponent","throws","home_away","game_date","pitcher_id"}
            feature_list = [c for c in feats.columns if c not in non_feats]
        X = feats.copy()
        X = X[feature_list] if all(c in X.columns for c in feature_list) else X
        pred_mean = pipe.predict(X)
        lam = np.clip(pred_mean, 0.3, 12.0).astype(float)
    else:
        lam = feats.apply(_baseline_lambda, axis=1).values

    # Distribution outputs
    p_over_5p5 = np.array([_poisson_tail_prob(l, 6) for l in lam])
    p_over_6p5 = np.array([_poisson_tail_prob(l, 7) for l in lam])
    p10 = np.array([_poisson_quantile(l, 0.10) for l in lam])
    p50 = np.array([_poisson_quantile(l, 0.50) for l in lam])
    p90 = np.array([_poisson_quantile(l, 0.90) for l in lam])

    # Build output (include pitcher_id for backtesting)
    out = pd.DataFrame({
        "pitcher_name": features.get("pitcher_name"),
        "team": features.get("team"),
        "opponent": features.get("opponent"),
        "home_away": features.get("home_away"),
        "game_date": features.get("game_date"),
        "throws": features.get("throws"),
        "pitcher_id": features.get("pitcher_id"),
        "pred_mean_K": lam,
        "p_over_5p5": p_over_5p5,
        "p_over_6p5": p_over_6p5,
        "P10": p10,
        "P50": p50,
        "P90": p90,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    })

    out_path = os.path.join(PRED_DIR, f"preds_{slate_date}.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")

if __name__ == "__main__":
    main()
