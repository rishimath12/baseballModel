import os, time, argparse, requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict

import pandas as pd
from pybaseball import statcast, pitching_stats, cache as pb_cache

pb_cache.enable()

MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PBP      = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
MLB_PEOPLE   = "https://statsapi.mlb.com/api/v1/people"

ROOT      = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
RAW_DIR   = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PA_SO = {"strikeout", "strikeout_double_play"}

# -------------------- helpers --------------------

def _to_iso_date_col(df: pd.DataFrame, col: str = "game_date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date.astype("string")
    return df

def _season_bounds(y: int) -> Tuple[str, str]:
    return f"{y}-03-01", f"{y}-11-30"

def _abbr_from_team(team_obj: dict) -> str:
    if not team_obj: return ""
    return (
        team_obj.get("abbreviation")
        or team_obj.get("fileCode")
        or team_obj.get("teamCode")
        or team_obj.get("name")
        or ""
    )

def _fetch_pitcher_hand(pid: int) -> Optional[str]:
    try:
        r = requests.get(f"{MLB_PEOPLE}/{pid}", timeout=10)
        r.raise_for_status()
        people = r.json().get("people", [])
        if people:
            code = (people[0].get("pitchHand") or {}).get("code")
            if code in ("R","L"): return code
    except Exception:
        pass
    return None

# -------------------- BULK season cache --------------------

def get_season_statcast(year: int) -> pd.DataFrame:
    p = os.path.join(CACHE_DIR, f"season_{year}.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
        return _to_iso_date_col(df, "game_date")

    start, end = _season_bounds(year)
    print(f"[cache] fetching season Statcast {year} …")
    df = statcast(start, end)
    if df is None:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame(columns=[
            "game_date","game_pk","home_team","away_team","inning_topbot",
            "pitcher","p_throws","pitcher_throws","pitch_type","description",
            "events","pitch_number","at_bat_number"
        ])

    keep = [c for c in df.columns if c in {
        "game_date","game_pk","home_team","away_team","inning_topbot",
        "pitcher","p_throws","pitcher_throws","pitch_type","description",
        "events","pitch_number","at_bat_number"
    }]
    df = df[keep].copy()
    df = _to_iso_date_col(df, "game_date")
    df.to_parquet(p, index=False)
    return df


# -------------------- Starters (authoritative PBP) --------------------

def get_starters_for_date(date_str: str) -> pd.DataFrame:
    """First pitcher in TOP = home starter; first in BOTTOM = away starter."""
    params = {"sportId": 1, "date": date_str}
    sched = requests.get(MLB_SCHEDULE, params=params, timeout=30).json()
    rows = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            game_pk   = g.get("gamePk")
            game_date = d.get("date")
            home_team = _abbr_from_team(g.get("teams", {}).get("home", {}).get("team"))
            away_team = _abbr_from_team(g.get("teams", {}).get("away", {}).get("team"))
            try:
                plays = requests.get(MLB_PBP.format(game_pk=game_pk), timeout=30).json().get("allPlays", [])
            except Exception:
                plays = []

            def first_pitcher(half: str):
                for p in plays:
                    if (p.get("about") or {}).get("halfInning") == half:
                        pit = (p.get("matchup") or {}).get("pitcher", {}).get("id")
                        return int(pit) if pit else None
                return None

            home_pid = first_pitcher("top")
            away_pid = first_pitcher("bottom")
            if home_pid:
                rows.append({"pitcher_id": home_pid,"team": home_team,"opponent": away_team,
                             "home_away": "home","game_date": game_date,
                             "throws": _fetch_pitcher_hand(home_pid),"pitcher_name": None})
            if away_pid:
                rows.append({"pitcher_id": away_pid,"team": away_team,"opponent": home_team,
                             "home_away": "away","game_date": game_date,
                             "throws": _fetch_pitcher_hand(away_pid),"pitcher_name": None})
    return pd.DataFrame(rows).drop_duplicates(subset=["pitcher_id","game_date"])

# -------------------- Feature builders (season cache; FAST) --------------------

def _last_n_starts(season_df: pd.DataFrame, pid: int, n: int, end_date: str) -> pd.DataFrame:
    dfp = season_df[(season_df["pitcher"] == pid) & (season_df["game_date"] <= end_date)]
    if dfp.empty: return dfp
    dates = dfp["game_date"].dropna().drop_duplicates().sort_values().to_list()
    keep = set(dates[-n:])
    return dfp[dfp["game_date"].isin(keep)]

def _recent_form_from_season(season_df: pd.DataFrame, pid: int, end_date: str, lookback_games: int) -> dict:
    df = _last_n_starts(season_df, pid, lookback_games, end_date)
    if df.empty:
        return {"recent_csw": None, "recent_swstr": None, "recent_pitches": 0, "recent_games": 0}
    df = df.copy()
    df["whiff"]  = df["description"].isin(["swinging_strike","swinging_strike_blocked"]).astype(int)
    df["called"] = (df["description"] == "called_strike").astype(int)
    g = df.groupby("game_date").agg(pitches=("pitch_number","count"),
                                    whiffs=("whiff","sum"),
                                    called=("called","sum")).reset_index()
    g["csw"] = (g["whiffs"] + g["called"]) / g["pitches"].clip(lower=1)
    recent_csw    = float(g["csw"].mean())
    recent_swstr  = float(g["whiffs"].sum() / g["pitches"].sum())
    recent_pitches= int(g["pitches"].sum())
    return {"recent_csw": recent_csw, "recent_swstr": recent_swstr,
            "recent_pitches": recent_pitches, "recent_games": int(len(g))}

def build_features_for_ids(starters_df: pd.DataFrame,
                           lookback_days: int,
                           feature_date_utc: datetime,
                           lookback_games: Optional[int] = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Training wrapper used by train.py (pitch mix long returned as empty for now)."""
    if starters_df is None or starters_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    end_str = feature_date_utc.date().isoformat()
    year = feature_date_utc.year
    season_df = get_season_statcast(year)

    rows = []
    for _, r in starters_df.iterrows():
        pid = int(r["pitcher_id"])
        recent = _recent_form_from_season(season_df, pid, end_str, lookback_games or 5)
        rows.append({
            "pitcher_id": pid,
            "pitcher_name": r.get("pitcher_name"),
            "team": r["team"],
            "opponent": r["opponent"],
            "home_away": r["home_away"],
            "game_date": r["game_date"],
            "throws": r["throws"],
            **recent
        })
    return pd.DataFrame(rows), pd.DataFrame(columns=["pitcher_id","pitcher_name","pitch_type","usage_pct","whiff_rate"])

# ---------- Live path for model.py (probables + same fast features) ----------

def get_probable_pitchers(date_str: str):
    params = {"sportId": 1, "date": date_str,
              "hydrate": "probablePitcher(note,person,stats),game(content(summary))"}
    data = requests.get(MLB_SCHEDULE, params=params, timeout=30).json()
    pub, internal = [], []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game_date = d.get("date")
            home_team = _abbr_from_team(g.get("teams", {}).get("home", {}).get("team"))
            away_team = _abbr_from_team(g.get("teams", {}).get("away", {}).get("team"))
            for side, opp_side in (("home","away"),("away","home")):
                info = g["teams"][side]; opp = g["teams"][opp_side]
                p = info.get("probablePitcher") or {}
                pid = p.get("id")
                if not pid: continue
                team = _abbr_from_team(info.get("team")); opp_t = _abbr_from_team(opp.get("team"))
                throws = (p.get("pitchHand") or {}).get("code") or _fetch_pitcher_hand(int(pid))
                row = {"pitcher_name": p.get("fullName"),
                       "team": team, "opponent": opp_t,
                       "home_away": "home" if side=="home" else "away",
                       "game_date": game_date, "throws": throws}
                pub.append(row); internal.append({**row, "pitcher_id": int(pid)})
    return pd.DataFrame(pub).drop_duplicates(), pd.DataFrame(internal).drop_duplicates(subset=["pitcher_id"])

def build_all_features(prob_int: pd.DataFrame,
                       lookback_days: int,
                       feature_date_utc: datetime,
                       lookback_games: Optional[int] = 5):
    """Live features for model.py using season cache (mirrors training)."""
    return build_features_for_ids(prob_int, lookback_days, feature_date_utc, lookback_games)

# ---------- CLI (optional quick run) ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--lookback_games", type=int, default=5)
    args = ap.parse_args()
    feat_date = datetime.fromisoformat(args.date)
    pub, intr = get_probable_pitchers(args.date)
    feats, pmix = build_all_features(intr, lookback_days=30, feature_date_utc=feat_date, lookback_games=args.lookback_games)
    feats.to_csv(os.path.join(RAW_DIR, f"features_{args.date}.csv"), index=False)
    print(f"Saved features for {args.date} → data/raw/features_{args.date}.csv")

if __name__ == "__main__":
    main()
