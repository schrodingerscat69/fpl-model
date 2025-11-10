#!/usr/bin/env python3
"""
Compare predicted_points in predictions/gw*_predictions.csv against actual
points for that gameweek from data_processed/training_table_with_fixture.csv.

Output (per GW only):
  predictions/error_analysis/gw{N}_pred_vs_actual.csv

Merging strategy (in order):
1) Try by ('gameweek','player_name','team_short') if those exist in training table.
2) Else, resolve player 'id' from (player_name, team_short) using season metadata
   (players.csv + teams.csv) and merge by ('gameweek','id').

Predictions must have columns:
  ['gameweek','player_name','team_short','position','predicted_points'].
"""

from __future__ import annotations
from pathlib import Path
import sys, re, unicodedata
import pandas as pd
import numpy as np

# ---------- repo paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
DATA_DIR = ROOT / "data_processed"
PRED_DIR = ROOT / "predictions"
TRAIN_FILE = DATA_DIR / "training_base_raw.csv"

EA_DIR = DATA_DIR / "error_analysis"
EA_DIR.mkdir(parents=True, exist_ok=True)

# Season metadata (used only if we must resolve IDs)
DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")

# ---------- canonical cols ----------
PLAYER_ID = "id"
GW_COL = "gameweek"
PRED_COL = "predicted_points"
TARGET_COL = "event_points"

# ---------- helpers ----------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _norm(series: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x): return x
        x = str(x).strip().lower()
        x = unicodedata.normalize("NFKD", x)
        return "".join(ch for ch in x if not unicodedata.combining(ch))
    return series.map(f)

def _pick(df: pd.DataFrame, cands: list[str], required=True) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {cands} found in columns: {list(df.columns)}")
    return None

def _gw_from_name(p: Path) -> int:
    m = re.search(r"gw(\d+)_predictions\.csv$", p.name, re.I)
    if not m:
        raise ValueError(f"Cannot infer GW from: {p.name}")
    return int(m.group(1))

def _season_lookup(season: str) -> pd.DataFrame:
    """(player_name, team_short) → player_id for a season (accents/Case insensitive)."""
    season_dir = DATA_REPO_ROOT / season
    players = pd.read_csv(season_dir / "players.csv")
    teams   = pd.read_csv(season_dir / "teams.csv")
    tlu = teams.rename(columns={"code":"team_code","short_name":"team_short"})[["team_code","team_short"]]
    pe = players.merge(tlu, on="team_code", how="left")
    lu = pd.DataFrame({
        "player_id": _num(pe["player_id"]),
        "player_name": pe["web_name"].astype(str),
        "team_short": pe["team_short"].astype(str),
    })
    lu["player_name_norm"] = _norm(lu["player_name"])
    lu["team_short_norm"]  = _norm(lu["team_short"])
    return lu

def _resolve_ids_from_name_team(df_pred: pd.DataFrame, season: str) -> pd.DataFrame:
    """Attach 'id' to predictions using (player_name, team_short)."""
    look = _season_lookup(season)
    df = df_pred.copy()
    df["player_name_norm"] = _norm(df["player_name"])
    df["team_short_norm"]  = _norm(df["team_short"])
    df = df.merge(
        look[["player_id","player_name_norm","team_short_norm"]],
        on=["player_name_norm","team_short_norm"],
        how="left"
    ).drop(columns=["player_name_norm","team_short_norm"])
    df = df.rename(columns={"player_id": PLAYER_ID})
    unresolved = df[df[PLAYER_ID].isna()]
    if not unresolved.empty:
        print(f"  [!] {len(unresolved)} rows could not be resolved to an ID; dropping a few examples:")
        print(unresolved[["player_name","team_short"]].head(10))
        df = df.dropna(subset=[PLAYER_ID])
    df[PLAYER_ID] = _num(df[PLAYER_ID])
    return df

# ---------- main ----------
def main():
    # Load training table (actuals source)
    if not TRAIN_FILE.exists():
        print(f"ERROR: missing {TRAIN_FILE}", file=sys.stderr)
        return 1

    df_all = pd.read_csv(TRAIN_FILE)
    if GW_COL not in df_all.columns or "season" not in df_all.columns:
        raise KeyError("training_table_with_fixture.csv must include 'season' and 'gameweek'.")

    # normalize
    df_all["season"] = df_all["season"].astype(str)
    df_all[GW_COL] = _num(df_all[GW_COL])

    # find id col & target col in training table
    if PLAYER_ID not in df_all.columns:
        alt_id = _pick(df_all, ["player_id","element","playerId"])
        df_all = df_all.rename(columns={alt_id: PLAYER_ID})
    if TARGET_COL not in df_all.columns:
        alt_t = _pick(df_all, ["total_points","gw_points","points"])
        df_all = df_all.rename(columns={alt_t: TARGET_COL})

    df_all[PLAYER_ID]  = _num(df_all[PLAYER_ID])
    df_all[TARGET_COL] = _num(df_all[TARGET_COL])

    # choose current season (latest)
    season_current = sorted(df_all["season"].unique())[-1]

    # Get all prediction files
    pred_files = sorted([p for p in PRED_DIR.glob("gw*_predictions.csv") if p.is_file()],
                        key=lambda p: _gw_from_name(p))
    if not pred_files:
        print(f"No gw*_predictions.csv files in {PRED_DIR}")
        return 0

    for pred_path in pred_files:
        gw = _gw_from_name(pred_path)
        print(f"\nGW {gw}: {pred_path.name}")

        # Load predictions; enforce the 5 expected columns
        dfp = pd.read_csv(pred_path)
        for needed in [GW_COL, "player_name", "team_short", "position", PRED_COL]:
            if needed not in dfp.columns:
                raise KeyError(f"{pred_path.name} missing required column: '{needed}'")

        # normalize types/strings
        dfp[GW_COL] = _num(dfp[GW_COL])
        dfp[PRED_COL] = _num(dfp[PRED_COL])

        # -------- attempt A: direct merge on (gameweek, player_name, team_short)
        direct_merge_possible = {"player_name","team_short"}.issubset(df_all.columns)
        if direct_merge_possible:
            # Normalized keys to avoid accent/case mismatches
            dfp["_name_norm"] = _norm(dfp["player_name"])
            dfp["_team_norm"] = _norm(dfp["team_short"])

            dfa = df_all[df_all["season"] == season_current].copy()
            dfa["_name_norm"] = _norm(dfa["player_name"]) if "player_name" in dfa.columns else np.nan
            dfa["_team_norm"] = _norm(dfa["team_short"]) if "team_short" in dfa.columns else np.nan

            has_keys = {"_name_norm","_team_norm"}.issubset(dfa.columns)
            if has_keys:
                merged = dfp.merge(
                    dfa[[GW_COL, TARGET_COL, "_name_norm", "_team_norm"]],
                    left_on=[GW_COL, "_name_norm", "_team_norm"],
                    right_on=[GW_COL, "_name_norm", "_team_norm"],
                    how="inner"
                )
            else:
                merged = pd.DataFrame()  # fallback below
        else:
            merged = pd.DataFrame()

        # -------- attempt B: if no/low match, resolve IDs and merge by (gameweek,id)
        if merged.empty or len(merged) < max(1, int(0.7*len(dfp))):
            dfp_id = _resolve_ids_from_name_team(dfp, season_current)
            dfa = df_all[df_all["season"] == season_current][[GW_COL, PLAYER_ID, TARGET_COL]].copy()
            merged = dfp_id.merge(dfa, on=[GW_COL, PLAYER_ID], how="inner")

        if merged.empty:
            print("  [!] No rows matched. Check name/team conventions or metadata path.")
            continue

        # compute errors & order columns
        merged["error"] = merged[PRED_COL] - merged[TARGET_COL]
        merged["abs_error"] = merged["error"].abs()

        ordered = [
            "season", GW_COL, PLAYER_ID,
            "player_name", "team_short", "position",
            PRED_COL, TARGET_COL, "error", "abs_error"
        ]
        keep = [c for c in ordered if c in merged.columns]
        out = merged[keep].sort_values(["team_short","abs_error"], ascending=[True, False]).reset_index(drop=True)

        # write per-GW CSV only
        out_csv = EA_DIR / f"gw{gw}_pred_vs_actual.csv"
        out.to_csv(out_csv, index=False)
        print(f"  -> wrote {out_csv.relative_to(ROOT)} (rows={len(out)})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
