#!/usr/bin/env python3
"""
Scan predictions/error_analysis/gw*_pred_vs_actual.csv and, for each GW and each team,
produce an Actual-vs-Predicted scatter plot labeled with player names.

Outputs:
  data_processed/plots/gw{N}/gw{N}_{TEAM}_actual_vs_pred.png
"""

from __future__ import annotations
from pathlib import Path
import re
import sys
import math
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- repo paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
PRED_ERR_DIR = ROOT / "data_processed" / "error_analysis"
OUT_BASE = ROOT / "data_processed" / "plots"

# ---------- helpers ----------
def _gw_from_name(p: Path) -> int:
    m = re.search(r"gw(\d+)_pred_vs_actual\.csv$", p.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer GW from filename: {p.name}")
    return int(m.group(1))

def _safe_tag(s: str) -> str:
    """Sanitize team code for filenames (remove accents, keep ascii)."""
    s_norm = unicodedata.normalize("NFKD", s)
    s_ascii = "".join(ch for ch in s_norm if not unicodedata.combining(ch))
    s_ascii = re.sub(r"[^A-Za-z0-9._-]+", "_", s_ascii)
    return s_ascii.strip("_") or "UNK"

def _compute_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.5) -> tuple[float, float]:
    lo = float(min(np.min(x) if x.size else 0, np.min(y) if y.size else 0))
    hi = float(max(np.max(x) if x.size else 1, np.max(y) if y.size else 1))
    # Expand a bit so labels/points don’t sit on the border
    return (math.floor(lo - pad), math.ceil(hi + pad))

def _plot_team(df_team: pd.DataFrame, gw: int, team: str, out_dir: Path):
    x = df_team["predicted_points"].to_numpy(dtype=float)
    y = df_team["event_points"].to_numpy(dtype=float)
    names = df_team["player_name"].astype(str).tolist()

    lo, hi = _compute_limits(x, y, pad=0.5)

    plt.figure(figsize=(6.4, 6.4))
    plt.scatter(x, y, alpha=0.7)

    # y = x reference line
    plt.plot([lo, hi], [lo, hi])

    # Annotate each point with player name (a small offset to reduce overlap)
    for xi, yi, name in zip(x, y, names):
        plt.annotate(
            name,
            (xi, yi),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=8,
        )

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Predicted points")
    plt.ylabel("Actual points")
    plt.title(f"GW{gw} — {team}: Actual vs Predicted")
    plt.tight_layout()

    out_path = out_dir / f"gw{gw}_{_safe_tag(team)}_actual_vs_pred.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path

# ---------- main ----------
def main():
    if not PRED_ERR_DIR.exists():
        print(f"ERROR: missing directory {PRED_ERR_DIR}", file=sys.stderr)
        return 1

    files = sorted(
        [p for p in PRED_ERR_DIR.glob("gw*_pred_vs_actual.csv") if p.is_file()],
        key=lambda p: _gw_from_name(p),
    )
    if not files:
        print(f"No files like gw*_pred_vs_actual.csv found in {PRED_ERR_DIR}")
        return 0

    for fpath in files:
        gw = _gw_from_name(fpath)
        df = pd.read_csv(fpath)

        # Basic sanity columns (as produced by your earlier script)
        required = {"gameweek", "team_short", "player_name", "predicted_points", "event_points"}
        missing = required - set(df.columns)
        if missing:
            print(f"[GW{gw}] Skipping {fpath.name}: missing columns {sorted(missing)}")
            continue

        out_dir = OUT_BASE / f"gw{gw}"
        out_dir.mkdir(parents=True, exist_ok=True)

        teams = sorted(df["team_short"].dropna().unique())
        if not teams:
            print(f"[GW{gw}] No teams found in {fpath.name}; skipping.")
            continue

        print(f"[GW{gw}] {fpath.name} → {out_dir.relative_to(ROOT)} (teams: {len(teams)})")
        for team in teams:
            df_team = df[df["team_short"] == team].copy()
            if df_team.empty:
                continue
            out_path = _plot_team(df_team, gw, team, out_dir)
            print(f"  - {team}: {out_path.relative_to(ROOT)}  (n={len(df_team)})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

