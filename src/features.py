from pathlib import Path
import pandas as pd
import numpy as np

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
RAW_FILE = PROJECT_ROOT / "data_processed" / "training_base_raw.csv"
OUT_FILE = PROJECT_ROOT / "data_processed" / "training_table.csv"

# This is the column in player_gameweek_stats that identifies the player.
# In most FPL-style data it's literally called "id".
PLAYER_ID_COL = "id"

def add_rolling_features(df):
    """
    For each player, sorted by season then gameweek,
    compute lag/rolling stats based ONLY on earlier gameweeks.
    """

    # Sort so rolling windows make sense
    df = df.sort_values(by=["season", PLAYER_ID_COL, "gameweek"]).copy()

    # We'll group by player
    g = df.groupby([ "season", PLAYER_ID_COL ], group_keys=False)

    # 1. Last GW points (lag 1)
    df["pts_prev_gw"] = g["event_points"].shift(1)

    # 2. Average points over last 3 GWs (excluding current GW)
    df["pts_avg_last3"] = (
        g["event_points"]
        .shift(1)                # shift so current GW isn't included
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # 3. Average minutes over last 3 GWs
    df["mins_avg_last3"] = (
        g["minutes"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # 4. How often they actually played (>=60 min) in last 3 GWs
    played60 = (df["minutes"] >= 60).astype(int)
    df["played60_rate_last3"] = (
        g[played60.name]  # use same groupby
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    return df


def build_training_table(df):
    """
    Take the raw stacked GW data and:
      - keep useful columns,
      - add rolling features,
      - drop rows that don't have enough history,
      - output final ML table.
    """

    # --- sanity: check what columns we have ---
    print("\nAvailable columns in raw df:")
    print(df.columns.tolist())
    print()

    # Add rolling / lag features
    df = add_rolling_features(df)

    # Target we're trying to predict
    target_col = "event_points"

    # Candidate feature columns.
    # We'll include:
    #   - rolling stuff we just built
    #   - current snapshot stuff that should be known before the GW starts
    # Adjust these names if your dataset uses slightly different naming.
    base_feature_cols = [
        "pts_prev_gw",
        "pts_avg_last3",
        "mins_avg_last3",
        "played60_rate_last3",
        "now_cost",
        "selected_by_percent",
        "form",
        "minutes",  # Careful: minutes THIS gw is not allowed as input for prediction.
                    # We'll drop it before training, but we keep it here
                    # for debugging / inspection.
    ]

    # Keep only columns that actually exist in the df to avoid KeyErrors
    feature_cols = [c for c in base_feature_cols if c in df.columns]

    # We'll also keep some ID/time info so we know what row is what
    id_cols = ["season", "gameweek", PLAYER_ID_COL]

    # Build the smaller table
    model_df = df[id_cols + feature_cols + [target_col]].copy()

    # We cannot train on rows where we don't have any history for that player,
    # e.g. GW1 has no pts_prev_gw etc.
    # Easiest rule: drop rows where pts_prev_gw is NaN.
    if "pts_prev_gw" in model_df.columns:
        model_df = model_df[model_df["pts_prev_gw"].notna()].copy()

    # OPTIONAL: also drop rows where the target is missing
    model_df = model_df[model_df[target_col].notna()].copy()

    return model_df


def main():
    # load the raw combined file from previous step
    df_raw = pd.read_csv(RAW_FILE)

    # build training table
    training_df = build_training_table(df_raw)

    # save result
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(OUT_FILE, index=False)

    print("Training table shape:", training_df.shape)
    print("Wrote", OUT_FILE)


if __name__ == "__main__":
    main()
