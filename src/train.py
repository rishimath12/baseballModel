# src/train.py
import os, glob, argparse, json
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
from pybaseball import cache as pb_cache

# speed up repeated calls
pb_cache.enable()

# our pipeline pieces
from scraper import (
    get_starters_for_date,   # starters via MLB PBP (no reliever mixups)
    get_season_statcast,     # fast season-wide Statcast parquet cache
    build_features_for_ids,  # builds features using the cached season file
)

ROOT       = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
HIST_DIR   = os.path.join(ROOT, "data", "historical")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PA_SO = {"strikeout", "strikeout_double_play"}

# helpers

def _daterange(start: datetime, end: datetime) -> List[datetime]:
    d, out = start, []
    while d <= end:
        out.append(d); d += timedelta(days=1)
    return out

def _ks_map_from_cache(date_str: str) -> dict:
    """Count Ks per pitcher for a date using the season parquet (fully local)."""
    year = int(date_str[:4])
    season = get_season_statcast(year)           # cached call
    day = season[season["game_date"] == date_str]
    if day.empty or "pitcher" not in day.columns:
        return {}
    so = day[day["events"].isin(PA_SO)].dropna(subset=["pitcher"])
    if so.empty:
        return {}
    so["pitcher"] = so["pitcher"].astype(int)
    return so.groupby("pitcher").size().to_dict()  # {pitcher_id: Ks}

#  build + label

def build_historical(start: str, end: str, lookback_games: int = 5, out: Optional[str] = None) -> str:
    """
    Builds a labeled pitcher-start dataset for [start, end] (inclusive).
    Uses:
      - Starters from MLB PBP
      - Features from cached season Statcast (fast)
      - Labels (ks_actual) from cached day Statcast
    """
    start_dt, end_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
    out_path = out or os.path.join(HIST_DIR, f"hist_pitcher_games_{start}_to_{end}.csv")
    header_written = os.path.exists(out_path)
    total = 0

    for day in _daterange(start_dt, end_dt):
        d_str = day.date().isoformat()
        print(f"[build] {d_str} — starters …")
        starters = get_starters_for_date(d_str)
        if starters.empty:
            print(f"[build]   no games")
            continue

        feats, _ = build_features_for_ids(
            starters_df=starters,
            lookback_days=30,
            feature_date_utc=day,
            lookback_games=lookback_games,
        )
        if feats.empty:
            print(f"[build]   empty features")
            continue

        kmap = _ks_map_from_cache(d_str)
        feats = feats.copy()
        feats["ks_actual"] = feats["pitcher_id"].map(kmap)

        feats.to_csv(out_path, mode=("a" if header_written else "w"),
                     index=False, header=not header_written)
        header_written = True
        total += len(feats)
        print(f"[build]   +{len(feats)} rows → {out_path}")

    print(f"[build] done → {out_path}  ({total} rows)")
    return out_path

#  fit + save

def fit_model(hist_glob: str, model_out: str, feats_out: str):
    """
    Trains LightGBM Poisson on all historical CSVs matching hist_glob,
    saves model + feature list.
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_error
        import joblib
    except Exception as e:
        print(f"[fit] Missing training deps: {e}")
        print("      Try: pip install lightgbm joblib scikit-learn")
        return

    files = sorted(glob.glob(hist_glob))
    if not files:
        print(f"[fit] No historical files matched {hist_glob}")
        return

    df_list = []
    for fp in files:
        try:
            df_list.append(pd.read_csv(fp))
            print(f"[fit] loaded {fp}")
        except Exception as e:
            print(f"[fit] skip {fp}: {e}")
    if not df_list:
        print("[fit] No readable historical files.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df[df["ks_actual"].notna()].copy()
    if df.empty:
        print("[fit] No labeled rows (ks_actual); abort.")
        return

    # feature selection: drop IDs and non-numeric targets
    non_feats = {"pitcher_name","team","opponent","throws","home_away","game_date","pitcher_id","ks_actual"}
    feature_cols = [c for c in df.columns if c not in non_feats]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["ks_actual"].astype(float)

    # chronological split
    order = df.sort_values("game_date").index
    X = X.loc[order]; y = y.loc[order]
    cut = int(len(X) * 0.8) if len(X) >= 10 else max(1, len(X) - 1)
    Xtr, Xva = X.iloc[:cut], X.iloc[cut:]
    ytr, yva = y.iloc[:cut], y.iloc[cut:]

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)

    params = dict(
        objective="poisson",
        metric="l2",
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=20,
        verbosity=-1,
    )

    print("[fit] Training LightGBM…")
    # Use LightGBM ≥4.0 callback API (no early_stopping_rounds kwarg)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=1200,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    if len(Xva) > 0:
        pred = model.predict(Xva, num_iteration=model.best_iteration)
        mae = mean_absolute_error(yva, pred)
        print(f"[fit] Validation MAE: {mae:.3f}")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    import joblib
    joblib.dump(model, model_out)
    with open(feats_out, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[fit] saved → {model_out}")
    print(f"[fit] saved → {feats_out}")

# CLI

def main():
    ap = argparse.ArgumentParser(description="Build historical data and auto‑fit the model.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True,   help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--lookback_games", type=int, default=5, help="N most recent starts for features")
    ap.add_argument("--out", default=None, help="Optional output CSV path")
    ap.add_argument("--hist_glob", default=os.path.join(HIST_DIR, "hist_pitcher_games_*.csv"),
                    help="Glob of historical CSVs to use for training")
    ap.add_argument("--model_out", default=os.path.join(MODELS_DIR, "model.pkl"))
    ap.add_argument("--features_out", default=os.path.join(MODELS_DIR, "feature_list.json"))
    ap.add_argument("--no_fit", action="store_true", help="Build only; skip training")
    args = ap.parse_args()

    out_csv = build_historical(args.start, args.end, lookback_games=args.lookback_games, out=args.out)
    if not args.no_fit:
        fit_model(args.hist_glob, args.model_out, args.features_out)

if __name__ == "__main__":
    main()
