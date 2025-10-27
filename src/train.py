from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- Paths ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent                              # fpl-model/
DATA_FILE = PROJECT_ROOT / "data_processed" / "training_table.csv"  # built in features.py

# cloned data-only repo path:
DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")


# ---------- Loading & splitting ----------
def load_data() -> pd.DataFrame:
    """Load the processed training table and coerce numeric snapshots."""
    df = pd.read_csv(DATA_FILE)

    # Make sure snapshot-like columns are numeric
    for col in ["selected_by_percent", "form", "now_cost"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Loaded training_table.csv with shape:", df.shape)
    print("Columns:", df.columns.tolist(), "\n")
    return df


def make_train_test_split(df: pd.DataFrame):
    """
    Train on earlier gameweeks, test on the latest GW in the most recent season.
    """
    seasons = sorted(df["season"].unique())
    if not seasons:
        raise RuntimeError("No seasons found in training_table.csv")
    season_current = seasons[-1]

    df_season = df[df["season"] == season_current].copy()
    gws = sorted(df_season["gameweek"].unique())
    if len(gws) < 2:
        raise RuntimeError("Need at least 2 distinct gameweeks for a holdout test set.")

    cutoff = max(gws) - 1  # hold out the final GW
    print("Gameweeks in this season:", gws)
    print(f"Using cutoff GW={cutoff}:\n  train = GW <= {cutoff}\n  test  = GW >  {cutoff}\n")

    df_train = df_season[df_season["gameweek"] <= cutoff].copy()
    df_test  = df_season[df_season["gameweek"] >  cutoff].copy()
    return df_train, df_test, season_current


# ---------- Features ----------
def get_features_and_target(df: pd.DataFrame):
    """
    Return X, y, and the feature column list.
    We exclude identifiers and any 'current GW' info we wouldn't know pre-deadline.
    """
    target_col = "event_points"
    drop_cols = {
        "event_points",      # target
        "season",
        "gameweek",
        "id",                # player id
        "minutes",           # current GW minutes (unknown before kickoff)
    }

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


# ---------- Player/Team lookup ----------
def load_player_lookup(season_name: str) -> pd.DataFrame:
    """
    Build a lookup so we can map:
        model 'id'  -> player_name, team_short, position
    using:
        data/<season_name>/players.csv   (player_id, web_name, team_code, position, ...)
        data/<season_name>/teams.csv     (code, short_name, ...)
    """
    season_folder = DATA_REPO_ROOT / season_name
    players_csv = season_folder / "players.csv"
    teams_csv   = season_folder / "teams.csv"

    if not players_csv.is_file():
        raise RuntimeError(f"Missing {players_csv}")
    if not teams_csv.is_file():
        raise RuntimeError(f"Missing {teams_csv}")

    players = pd.read_csv(players_csv)
    teams   = pd.read_csv(teams_csv)

    # teams.csv:
    # code,id,name,short_name,strength,...,elo,...
    # players.csv:
    # player_code,player_id,first_name,second_name,web_name,team_code,position

    # Map team_code -> short club label
    team_lookup = teams.rename(
        columns={"code": "team_code", "short_name": "team_short"}
    )[["team_code", "team_short"]]

    players_enriched = players.merge(team_lookup, on="team_code", how="left")

    # Build final lookup
    # player_id should match df['id'] in training_table.csv
    lookup = pd.DataFrame(
        {
            "player_id": pd.to_numeric(players_enriched["player_id"], errors="coerce"),
            "player_name": players_enriched["web_name"].astype(str),
            "team_short": players_enriched["team_short"].astype(str),
            "position": players_enriched["position"].astype(str),
        }
    )

    return lookup


def attach_player_metadata(results_df: pd.DataFrame, player_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge model results (which have 'id') with lookup (which has player_id, etc.),
    and arrange nice readable columns.
    """
    merged = results_df.merge(
        player_lookup,
        left_on="id",
        right_on="player_id",
        how="left",
    )

    nice_cols = [
        "season",
        "gameweek",
        "player_name",
        "team_short",
        "position",
        "predicted_points",
        "actual_points",
        "id",  # keep id at the end
    ]
    nice_cols = [c for c in nice_cols if c in merged.columns]

    merged = merged[nice_cols].sort_values(by="predicted_points", ascending=False)
    return merged


# ---------- Pretty printing by position ----------
def print_top_by_position(results_named: pd.DataFrame, top_n: int = 10):
    """
    Print top N predicted players for each position separately.
    Assumes 'position' exists and looks like 'Goalkeeper', 'Defender', 'Midfielder', 'Forward'.
    """

    # We'll define an order to keep output consistent
    pos_order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

    for pos in pos_order:
        subset = results_named[results_named["position"] == pos].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(by="predicted_points", ascending=False).head(top_n)

        print(f"\n=== Top {top_n} {pos}s ===")
        # Show relevant columns
        cols_to_show = [
            "player_name",
            "team_short",
            "predicted_points",
            "actual_points",
        ]
        # filter in case one of them is missing
        cols_to_show = [c for c in cols_to_show if c in subset.columns]

        print(subset[cols_to_show].reset_index(drop=True))


# ---------- Train & evaluate ----------
def train_and_eval(df_train: pd.DataFrame, df_test: pd.DataFrame, season_current: str):
    # Build X/y
    X_train, y_train, feature_cols = get_features_and_target(df_train)
    X_test_full, y_test, _ = get_features_and_target(df_test)

    # Keep columns aligned
    X_test = X_test_full[feature_cols]

    print("Feature columns being used:")
    for c in feature_cols:
        print("  ", c)
    print()

    # Train baseline model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Evaluate
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test  = mean_absolute_error(y_test,  y_pred_test)
    print(f"Train MAE: {mae_train:.4f}")
    print(f"Test  MAE: {mae_test:.4f}\n")

    # Build scored test rows
    results = df_test.copy()
    results["predicted_points"] = y_pred_test
    results["actual_points"]    = y_test.values

    # Sort by predicted points overall
    results_sorted = results.sort_values(
        by="predicted_points", ascending=False
    )[["season", "gameweek", "id", "predicted_points", "actual_points"]]

    # Add player_name / team / position
    player_lookup = load_player_lookup(season_current)
    results_named = attach_player_metadata(results_sorted, player_lookup)

    # Print summary
    print("Top predicted players in the test set (overall best):")
    print(results_named.head(15))
    print(f"\n[{len(results_named)} rows total]\n")

    # Print per-position top 10
    print_top_by_position(results_named, top_n=10)

    return model, results_named


def main():
    df = load_data()
    df_train, df_test, season_current = make_train_test_split(df)
    _model, _named = train_and_eval(df_train, df_test, season_current)


if __name__ == "__main__":
    main()
