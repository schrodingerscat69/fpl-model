#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths (adjusted to your layout) ---
PROJECT_ROOT = Path("/home/mann-gandhi/Documents/GitHub-Repositories/fpl-model")
DATA_DIR     = PROJECT_ROOT / "data_processed"            # where model_residuals_gw#.csv lives
PLOTS_DIR    = DATA_DIR / "plots"                         # where we’ll save the figure(s)

def load_residuals(csv_path: Path | None, gw: int | None) -> tuple[pd.DataFrame, Path]:
    if csv_path is None:
        if gw is None:
            raise ValueError("Provide --gw or --csv")
        csv_path = DATA_DIR / f"model_residuals_gw{gw}.csv"
    df = pd.read_csv(csv_path)
    needed = {"team_short", "predicted_points", "actual_points"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df, csv_path

def make_plot(df: pd.DataFrame, title: str, outfile: Path):
    # sort by team (then position if present) to create blocks
    order_cols = ["team_short", "position"] if "position" in df.columns else ["team_short"]
    df_sorted = df.sort_values(order_cols, kind="mergesort").reset_index(drop=True)
    df_sorted["x"] = np.arange(len(df_sorted), dtype=float)

    x_pred = df_sorted["x"] - 0.15
    x_act  = df_sorted["x"] + 0.15
    y_pred = df_sorted["predicted_points"].astype(float)
    y_act  = df_sorted["actual_points"].astype(float)

    # team tick positions
    team_groups   = df_sorted.groupby("team_short", sort=False)
    tick_positions = [(g["x"].iloc[0] + g["x"].iloc[-1]) / 2.0 for _, g in team_groups]
    tick_labels    = [team for team, _ in team_groups]

    # vertical separators at team boundaries
    boundaries = team_groups["x"].min().iloc[1:].to_list()

    plt.figure(figsize=(14, 6))
    # small segment connecting actual ↔ predicted per player (nice visual)
    for xi, ya, yp in zip(df_sorted["x"], y_act, y_pred):
        plt.plot([xi - 0.05, xi + 0.05], [ya, yp], linewidth=0.7, alpha=0.4)

    plt.scatter(x_pred, y_pred, s=20, label="Predicted")
    plt.scatter(x_act,  y_act,  s=20, label="Actual", marker="x")

    for b in boundaries:
        plt.axvline(b - 0.5, linewidth=0.6, alpha=0.3)

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Teams (players grouped by team)")
    plt.ylabel("FPL points")
    plt.title(title)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    print(f"Saved plot → {outfile}")

def main():
    ap = argparse.ArgumentParser(description="Plot predicted vs actual points by team.")
    ap.add_argument("--gw", type=int, default=None,
                    help="Gameweek, reads data_processed/model_residuals_gw{GW}.csv")
    ap.add_argument("--csv", type=str, default=None,
                    help="Explicit residuals CSV path (overrides --gw)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    df, used_csv = load_residuals(csv_path, args.gw)

    # work out GW for title/filename
    gw = args.gw
    if gw is None:
        try:
            gw = int("".join(ch for ch in used_csv.stem if ch.isdigit()))
        except Exception:
            gw = -1

    title = f"Predicted vs Actual Points by Team (GW{gw})" if gw != -1 else "Predicted vs Actual Points by Team"
    outfile = PLOTS_DIR / (f"gw{gw}_pred_vs_actual_by_team.png" if gw != -1 else "pred_vs_actual_by_team.png")
    make_plot(df, title, outfile)

if __name__ == "__main__":
    main()

