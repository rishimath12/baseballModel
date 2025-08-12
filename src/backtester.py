# src/backtester.py
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import pandas as pd
from pybaseball import statcast_pitcher, cache

# speed up repeat calls
cache.enable()

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

ROOT = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
PRED_DIR = os.path.join(ROOT, "predictions")
BACKTEST_DIR = os.path.join(ROOT, "data", "backtests")
HIST_DIR = os.path.join(ROOT, "data", "historical")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(BACKTEST_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

_SO_EVENTS = {"strikeout", "strikeout_double_play"}

def _yesterday_chi() -> str:
    """Return yesterday's date in America/Chicago."""
    if ZoneInfo is None:
        return (datetime.now() - timedelta(days=1)).date().isoformat()
    now_chi = datetime.now(ZoneInfo("America/Chicago"))
    return (now_chi - timedelta(days=1)).date().isoformat()

def _load_preds(date_str: str) -> pd.DataFrame:
    path = os.path.join(PRED_DIR, f"preds_{date_str}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    return pd.read_csv(path)

def _load_feature_snapshot(date_str: str) -> Optional[pd.DataFrame]:
    fpath = os.path.join(HIST_DIR, f"features_{date_str}.csv")
    return pd.read_csv(fpath) if os.path.exists(fpath) else None

def _actual_ks_for_date(pid: int, date_str: str) -> Optional[int]:
    """Count strikeouts for a pitcher on the given date via Statcast."""
    try:
        df = statcast_pitcher(date_str, date_str, int(pid))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return int(df["events"].isin(_SO_EVENTS).sum())

def _eval_metrics(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["ks_actual"])
    if df.empty:
        return {"n": 0}
    e = df["ks_actual"] - df["pred_mean_K"]
    mae = float(e.abs().mean())
    rmse = float((e**2).mean() ** 0.5)
    bias = float(e.mean())
    met = {
        "n": int(len(df)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias
    }
    if "p_over_5p5" in df and "p_over_6p5" in df:
        met["mean_p_over_5p5"] = float(df["p_over_5p5"].mean())
        met["emp_over_5p5"] = float((df["ks_actual"] >= 6).mean())
        met["mean_p_over_6p5"] = float(df["p_over_6p5"].mean())
        met["emp_over_6p5"] = float((df["ks_actual"] >= 7).mean())
    return met

def _append_labeled_rows(date_str: str, preds: pd.DataFrame, feats_snap: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Merge predictions/actuals into the frozen features snapshot and
    append labeled rows to a cumulative historical file.
    """
    if feats_snap is None:
        print(f"[append] No feature snapshot for {date_str}; skipping labeled append.")
        return None

    # Merge on pitcher_id primarily; keep metadata columns
    keep_cols = list(feats_snap.columns)
    merged = feats_snap.merge(
        preds[["pitcher_id", "ks_actual"]], on="pitcher_id", how="left"
    )
    merged = merged.dropna(subset=["ks_actual"])

    if merged.empty:
        print("[append] No finished games yet (no ks_actual).")
        return None

    out_hist = os.path.join(HIST_DIR, "hist_pitcher_games_labeled.csv")
    header = not os.path.exists(out_hist)
    merged.to_csv(out_hist, mode="a", index=False, header=header)
    return out_hist

def _retrain_if_requested(hist_csv: str, model_out: str, features_out: str):
    try:
        import lightgbm as lgb
        import joblib, json
        from sklearn.metrics import mean_absolute_error
    except Exception as e:
        print(f"[retrain] Skipping retrain (missing deps): {e}")
        return

    if not os.path.exists(hist_csv):
        print(f"[retrain] Labeled file not found: {hist_csv}")
        return

    df = pd.read_csv(hist_csv)
    df = df[df["ks_actual"].notna()].copy()
    if df.empty:
        print("[retrain] No labeled rows; abort.")
        return

    # Drop identifiers for training
    non_feats = {"pitcher_name","team","opponent","throws","home_away","game_date","pitcher_id","ks_actual"}
    feature_cols = [c for c in df.columns if c not in non_feats]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["ks_actual"].astype(float)

    # Chronological split
    order = df.sort_values("game_date").index
    X = X.loc[order]; y = y.loc[order]
    split = int(len(X)*0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y.iloc[:split], y.iloc[split:]

    train_set = lgb.Dataset(Xtr, label=ytr)
    valid_set = lgb.Dataset(Xva, label=yva, reference=train_set)
    params = {
        "objective": "poisson",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    print("[retrain] Training LightGBM Poisson on labeled set…")
    model = lgb.train(
        params, train_set,
        num_boost_round=1500,
        valid_sets=[train_set, valid_set],
        valid_names=["train","valid"],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    mae = mean_absolute_error(yva, pred)
    print(f"[retrain] Validation MAE: {mae:.3f}")

    joblib.dump(model, model_out)
    with open(features_out, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[retrain] Saved model → {model_out}")
    print(f"[retrain] Saved feature list → {features_out}")

def main():
    ap = argparse.ArgumentParser(description="Backtest yesterday's predictions, log metrics, and append labels.")
    ap.add_argument("--date", help="Slate date YYYY-MM-DD (defaults to yesterday in America/Chicago)", default=None)
    ap.add_argument("--append_historical", action="store_true", help="Append labeled rows to historical dataset")
    ap.add_argument("--retrain", action="store_true", help="After appending, retrain model on labeled set")
    args = ap.parse_args()

    date_str = args.date or _yesterday_chi()
    print(f"[backtest] Evaluating slate {date_str}")

    preds = _load_preds(date_str)
    feats_snap = _load_feature_snapshot(date_str)

    # Ensure we have pitcher_id (needed for statcast lookup & merging)
    if "pitcher_id" not in preds.columns:
        if feats_snap is not None and "pitcher_id" in feats_snap.columns:
            preds = preds.merge(feats_snap[["pitcher_name","team","pitcher_id"]].drop_duplicates(),
                                on=["pitcher_name","team"], how="left")
        else:
            raise ValueError("pitcher_id missing from predictions and no feature snapshot to recover it.")

    # Fetch actual Ks
    actuals = []
    for _, r in preds.iterrows():
        pid = r.get("pitcher_id")
        if pd.isna(pid):
            actuals.append(None)
            continue
        ks = _actual_ks_for_date(int(pid), date_str)
        actuals.append(ks)

    preds = preds.copy()
    preds["ks_actual"] = actuals
    preds["abs_err"] = (preds["ks_actual"] - preds["pred_mean_K"]).abs()
    preds["sq_err"]  = (preds["ks_actual"] - preds["pred_mean_K"])**2

    # Save per-day eval
    out_day = os.path.join(BACKTEST_DIR, f"eval_{date_str}.csv")
    preds.to_csv(out_day, index=False)
    print(f"[backtest] Saved day eval → {out_day}")

    # Append to rolling log
    log_path = os.path.join(BACKTEST_DIR, "eval_log.csv")
    header = not os.path.exists(log_path)
    preds.assign(eval_date=date_str).to_csv(log_path, mode="a", index=False, header=header)
    print(f"[backtest] Updated log → {log_path}")

    # Metrics
    metrics = _eval_metrics(preds)
    if metrics["n"] == 0:
        print("[backtest] No finalized games found yet.")
    else:
        print(f"[backtest] n={metrics['n']}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  bias={metrics['bias']:.3f}")
        if "mean_p_over_5p5" in metrics:
            print(f"           p5.5 mean={metrics['mean_p_over_5p5']:.3f} vs emp>=6={metrics['emp_over_5p5']:.3f}")
            print(f"           p6.5 mean={metrics['mean_p_over_6p5']:.3f} vs emp>=7={metrics['emp_over_6p5']:.3f}")

    # Append labeled rows to historical dataset
    labeled_path = None
    if args.append_historical:
        labeled_path = _append_labeled_rows(date_str, preds, feats_snap)
        if labeled_path:
            print(f"[append] Labeled rows appended → {labeled_path}")

    # Optional retrain
    if args.retrain and labeled_path:
        model_out = os.path.join(MODELS_DIR, "model.pkl")
        features_out = os.path.join(MODELS_DIR, "feature_list.json")
        _retrain_if_requested(labeled_path, model_out, features_out)

if __name__ == "__main__":
    sys.exit(main())
