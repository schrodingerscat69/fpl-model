#!/usr/bin/env python3
"""
Produce per-GW player-level residual tables, sorted by team then by error.

Inputs  : data_processed/model_residuals_gw*.csv
Outputs : data_processed/error_analysis/gw{N}_players_by_team_error.csv

Sorting 1) team_short (A→Z)
        2) abs_error (desc) within each team
"""

from pathlib import Path
import re
import argparse
import pandas as pd


# ---------- paths ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data_processed"
OUT_DIR = DATA_DIR / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def list_residual_files() -> list[Path]:
    files = list(DATA_DIR.glob("model_residuals_gw*.csv"))
    def gw_key(p: Path):
        m = re.search(r"model_residuals_gw(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=gw_key)

def file_for_gw(gw: int) -> Path | None:
    p = DATA_DIR / f"model_residuals_gw{gw}.csv"
    return p if p.exists() else None

def ensure_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee presence of error and abs_error columns."""
    if "error" not in df.columns:
        if {"predicted_points", "actual_points"}.issubset(df.columns):
            df = df.copy()
            df["error"] = df["predicted_points"] - df["actual_points"]
        else:
            raise ValueError("Residual file missing predicted_points/actual_points.")
    if "abs_error" not in df.columns:
        df = df.copy()
        df["abs_error"] = df["error"].abs()
    return df

def sort_players(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by team then by abs error (desc)."""
    # Stable sort: first by abs_error desc, then by team asc (so team groups stay)
    # But we want primary grouping by team → sort by team asc, then abs_error desc.
    return df.sort_values(by=["team_short", "abs_error"], ascending=[True, False])

def slim_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "season", "gameweek", "id",
        "player_name", "team_short", "position",
        "predicted_points", "actual_points", "error", "abs_error",
    ]
    # keep any that exist; residual files already have these from your pipeline
    return df[[c for c in cols if c in df.columns]].copy()

def write_players_csv(gw: int, df_sorted: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"gw{gw}_players_by_team_error.csv"
    df_sorted.to_csv(out_path, index=False)
    return out_path


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Export player residuals sorted by team then error.")
    ap.add_argument("--gw", type=int, default=None, help="Specific gameweek to export (e.g., 9).")
    args = ap.parse_args()

    targets: list[tuple[int, Path]] = []

    if args.gw is not None:
        p = file_for_gw(args.gw)
        if p is None:
            print(f"[!] No file for GW{args.gw}: {DATA_DIR}/model_residuals_gw{args.gw}.csv")
            return 1
        targets = [(args.gw, p)]
    else:
        files = list_residual_files()
        if not files:
            print(f"[!] No residuals found in {DATA_DIR} (model_residuals_gw*.csv).")
            return 1
        for f in files:
            m = re.search(r"model_residuals_gw(\d+)\.csv$", f.name)
            if m:
                targets.append((int(m.group(1)), f))

    for gw, path in targets:
        df = pd.read_csv(path)
        df = ensure_errors(df)
        df_out = slim_columns(df)
        df_out = sort_players(df_out)

        out = write_players_csv(gw, df_out)
        print(f"[GW{gw}] Wrote {out.relative_to(PROJECT_ROOT)} "
              f"({len(df_out):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

