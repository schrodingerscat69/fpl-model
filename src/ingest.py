from pathlib import Path
import pandas as pd


# === paths ===
# This file lives in:    fpl-model/src/ingest.py
# The data lives in:     ../FPL-Elo-Insights-data/data/
# The output goes to:    ../data_processed/training_base_raw.csv

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent          # fpl-model/
DATA_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")
OUT_DIR = PROJECT_ROOT / "data_processed"
OUT_FILE = OUT_DIR / "training_base_raw.csv"


def load_all_player_gameweek_stats():
    """
    Walk all seasons under DATA_ROOT, e.g.:
        data/2024-2025/By Gameweek/GW1/player_gameweek_stats.csv
        data/2025-2026/By Gameweek/GW2/player_gameweek_stats.csv
    For every CSV found:
      - read it
      - add 'season' and 'gameweek' columns
    Return one big DataFrame.
    """

    frames = []

    # loop over seasons like "2024-2025", "2025-2026", etc.
    for season_dir in DATA_ROOT.iterdir():
        if not season_dir.is_dir():
            continue

        season_name = season_dir.name  # we'll keep this as string label

        by_gw_dir = season_dir / "By Gameweek"
        if not by_gw_dir.is_dir():
            # if a season folder doesn't have "By Gameweek", skip it
            continue

        # inside "By Gameweek" we expect folders like "GW1", "GW2", ...
        for gw_dir in by_gw_dir.iterdir():
            if not gw_dir.is_dir():
                continue
            if not gw_dir.name.startswith("GW"):
                continue

            # parse the integer gameweek number from folder name
            # e.g. "GW12" -> 12
            try:
                gw_num = int(gw_dir.name[2:])
            except ValueError:
                continue

            csv_path = gw_dir / "player_gameweek_stats.csv"
            if not csv_path.is_file():
                # some gameweeks might be missing this file
                continue

            df = pd.read_csv(csv_path)

            # add context columns
            df["season"] = season_name
            df["gameweek"] = gw_num

            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No player_gameweek_stats.csv files found under {DATA_ROOT}. "
            "Check that DATA_ROOT is correct."
        )

    # combine them all
    big_df = pd.concat(frames, ignore_index=True)

    return big_df


def main():
    # load + merge everything
    df = load_all_player_gameweek_stats()

    # quick sanity prints
    print("Combined shape:", df.shape)
    print("Seasons found:", sorted(df["season"].unique()))
    # show first few unique gameweeks per season
    preview = (
        df[["season", "gameweek"]]
        .drop_duplicates()
        .sort_values(["season", "gameweek"])
        .groupby("season")
        .head(5)
    )
    print("Sample season/gameweek combos:\n", preview)

    # make sure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # save the raw merged data
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
