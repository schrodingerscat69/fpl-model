from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# Config
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # repo root
DATA_FILE = PROJECT_ROOT / "data_processed" / "training_table_with_fixture.csv"

PLAYER_ID_COL = "id"
TARGET_COL = "event_points"

# Gameweek you want to inspect for residuals / error analysis.
# Change this each time you want to look at a different finished GW.
EVAL_GW = 10  # <-- set this manually (int). Example: 9, 10, etc.


# ============================================================
# Feature sets
# ============================================================

FEATURES_BASELINE = [
    "pts_prev_gw",
    "pts_avg_last3",
    "mins_avg_last3",
    "played60_rate_last3",
    "now_cost",
    "selected_by_percent",
    "form",
    "is_home",
    "team_elo",
    "opp_elo",
    "opp_def_strength",
]

FEATURES_EXTENDED_BASIC = FEATURES_BASELINE + [
    "team_code",
    "opp_code",
]

FEATURES_EXTENDED_INTERACT = FEATURES_EXTENDED_BASIC + [
    "elo_diff",
    "form_x_home",
]


# ============================================================
# Helpers
# ============================================================

def clean_snapshot_columns(df: pd.DataFrame):
    """
    Convert string-ish snapshot cols like 'selected_by_percent' ('12.3%')
    into numeric. In-place.
    """
    for col in ["selected_by_percent", "form", "now_cost"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra engineered features used in FEATURES_EXTENDED_INTERACT.
    Safe if some source cols are missing.
    """
    # elo_diff = team_elo - opp_elo
    if "team_elo" in df.columns and "opp_elo" in df.columns:
        df["elo_diff"] = df["team_elo"] - df["opp_elo"]
    else:
        df["elo_diff"] = np.nan

    # form_x_home = form * is_home
    if "form" in df.columns and "is_home" in df.columns:
        df["form_x_home"] = (
            pd.to_numeric(df["form"], errors="coerce")
            * pd.to_numeric(df["is_home"], errors="coerce")
        )
    else:
        df["form_x_home"] = np.nan

    return df


def make_train_test_split(df_season: pd.DataFrame, cutoff_gw: int):
    """
    - train = all rows with gameweek <= cutoff_gw
    - test  = all rows with gameweek >  cutoff_gw
    """
    df_train = df_season[df_season["gameweek"] <= cutoff_gw].copy()
    df_test  = df_season[df_season["gameweek"] >  cutoff_gw].copy()
    return df_train, df_test


def build_xy(df_subset: pd.DataFrame, feature_cols):
    """
    Build train/test matrices for a given feature set.
    Return X, y, and df_model (aligned rows).
    """
    cols_needed = feature_cols + [TARGET_COL]
    df_model = df_subset.dropna(subset=cols_needed).copy()

    X = df_model[feature_cols].copy()
    y = df_model[TARGET_COL].copy()

    return X, y, df_model


def train_and_eval_model(
    model,
    X_train, y_train,
    X_test, y_test,
    model_name: str,
    feature_set_name: str,
):
    """
    Fit model, compute train/test MAE, return dict of scores plus fitted model.
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test  = mean_absolute_error(y_test,  y_pred_test)

    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features_used": X_train.shape[1],
        "train_mae": mae_train,
        "test_mae": mae_test,
        "fitted_model": model,
    }


def attach_player_print_info(df_rows: pd.DataFrame, season_current: str):
    """
    Attach readable info: player name, team short, position.
    """
    DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")
    season_dir = DATA_REPO_ROOT / season_current

    players_df = pd.read_csv(season_dir / "players.csv")
    teams_df   = pd.read_csv(season_dir / "teams.csv")

    team_lookup = teams_df.rename(
        columns={"code": "team_code", "short_name": "team_short"}
    )[["team_code", "team_short"]]

    players_enriched = players_df.merge(team_lookup, on="team_code", how="left")

    lookup = pd.DataFrame({
        "player_id": pd.to_numeric(players_enriched["player_id"], errors="coerce"),
        "player_name": players_enriched["web_name"].astype(str),
        "team_short": players_enriched["team_short"].astype(str),
        "position": players_enriched["position"].astype(str),
    })

    out = df_rows.merge(
        lookup,
        left_on=PLAYER_ID_COL,
        right_on="player_id",
        how="left"
    )
    return out


def print_top_players(df_test_preds_named, header, top_n=15):
    """
    Show top predicted scorers in test split (sanity check).
    """
    cols_to_show = [
        "season",
        "gameweek",
        "player_name",
        "team_short",
        "position",
        "predicted_points",
        "actual_points",
        PLAYER_ID_COL,
    ]

    preview = (
        df_test_preds_named
        .sort_values(by="predicted_points", ascending=False)
        .head(top_n)[cols_to_show]
    )

    print(f"\n{header}")
    print(preview.reset_index(drop=True))


def run_feature_set(
    df_train,
    df_test,
    feature_cols,
    feature_set_name: str,
):
    """
    Train & evaluate Ridge / RF / HGBDT on this feature set.
    Return list of result dicts (with fitted_model + aligned data).
    """
    X_train, y_train, df_train_used = build_xy(df_train, feature_cols)
    X_test,  y_test,  df_test_used  = build_xy(df_test,  feature_cols)

    if len(X_train) == 0 or len(X_test) == 0:
        return []

    # Ridge
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_scores = train_and_eval_model(
        ridge_model,
        X_train, y_train, X_test, y_test,
        model_name="Ridge",
        feature_set_name=feature_set_name,
    )

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_scores = train_and_eval_model(
        rf_model,
        X_train, y_train, X_test, y_test,
        model_name="RandomForest",
        feature_set_name=feature_set_name,
    )

    # HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        random_state=42,
    )
    hgb_scores = train_and_eval_model(
        hgb_model,
        X_train, y_train, X_test, y_test,
        model_name="HistGBDT",
        feature_set_name=feature_set_name,
    )

    # attach context we’ll need downstream
    for scores in [ridge_scores, rf_scores, hgb_scores]:
        scores["df_train_used"] = df_train_used
        scores["df_test_used"]  = df_test_used
        scores["feature_cols"]  = feature_cols

    return [ridge_scores, rf_scores, hgb_scores]


# ============================================================
# main
# ============================================================

def main():
    # Load processed table (already merged with fixture info)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {DATA_FILE.name} with shape: {df.shape}")
    print("Columns:", list(df.columns))

    # Clean + engineer
    df = clean_snapshot_columns(df)
    df = add_engineered_columns(df)

    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["season"]   = df["season"].astype(str)

    # Focus on last season in data
    seasons = sorted(df["season"].unique())
    season_current = seasons[-1]
    df_season = df[df["season"] == season_current].copy()

    # Check available GWs
    gws = sorted(df_season["gameweek"].dropna().unique())
    print("\nGameweeks in this season:", gws)

    # Train/test split cutoff.
    # train = GW <= cutoff, test = GW > cutoff.
    cutoff_gw = 9  #update to 10 after gw11
    print(f"Using cutoff GW={cutoff_gw}:")
    print("    train = gameweek <= ", cutoff_gw)
    print("    test  = gameweek >  ", cutoff_gw)

    df_train, df_test = make_train_test_split(df_season, cutoff_gw)

    # Which feature sets do we evaluate?
    feature_sets = {
        "baseline": FEATURES_BASELINE,
        "extended_basic": FEATURES_EXTENDED_BASIC,
        "extended_interact": FEATURES_EXTENDED_INTERACT,
    }

    print("\nFeature sets to evaluate:")
    for name, cols in feature_sets.items():
        print(f"\nFeature set '{name}' using {len(cols)} cols:")
        for c in cols:
            print(" ", c)

    # Train / score all combos
    all_results = []
    for fs_name, fs_cols in feature_sets.items():
        all_results.extend(
            run_feature_set(df_train, df_test, fs_cols, fs_name)
        )

    if not all_results:
        print("No models trained. Not enough rows?")
        return

    # Summarize benchmark
    results_table = pd.DataFrame([
        {
            "feature_set": r["feature_set"],
            "model": r["model"],
            "n_features_used": r["n_features_used"],
            "train_mae": r["train_mae"],
            "test_mae": r["test_mae"],
        }
        for r in all_results
    ])

    print("\n=== Model Benchmark by Feature Set (MAE lower = better) ===")
    print(
        results_table[
            ["feature_set", "model", "n_features_used", "train_mae", "test_mae"]
        ]
    )

    # Pick best combo (lowest test_mae)
    best_idx = results_table["test_mae"].idxmin()
    best_row_summary = results_table.loc[best_idx]

    best_feature_set = best_row_summary["feature_set"]
    best_model_name  = best_row_summary["model"]
    best_test_mae    = best_row_summary["test_mae"]

    print(
        f"\nBest overall by test MAE: {best_model_name} "
        f"on feature_set='{best_feature_set}' "
        f"(test MAE={best_test_mae:.4f})"
    )

    # Grab the rich record so we can get fitted model + aligned data
    best_full = None
    for r in all_results:
        if (
            r["feature_set"] == best_feature_set
            and r["model"] == best_model_name
        ):
            best_full = r
            break

    if best_full is None:
        print("Couldn't retrieve best model details.")
        return

    best_model        = best_full["fitted_model"]
    best_feature_cols = best_full["feature_cols"]
    df_test_used      = best_full["df_test_used"].copy()

    # -------------------------------------------------
    # Restrict evaluation to the single GW we care about
    # (EVAL_GW from the config at the top).
    # -------------------------------------------------
    df_eval = df_test_used[df_test_used["gameweek"] == EVAL_GW].copy()
    if df_eval.empty:
        print(f"\nWARNING: no test rows for gameweek {EVAL_GW}. "
              f"Available in test: {sorted(df_test_used['gameweek'].unique())}")
        # we'll still fall back to using all test rows so script doesn't die
        df_eval = df_test_used.copy()
        print("Falling back to all test gameweeks in residual output.\n")

    # Predict on just those rows
    df_eval["predicted_points"] = best_model.predict(df_eval[best_feature_cols])
    df_eval["actual_points"] = df_eval[TARGET_COL]

    # Human friendly columns
    df_named = attach_player_print_info(df_eval, season_current)

    # Print sanity leaderboard (top N predicted in that GW)
    print_top_players(
        df_named,
        header=f"Top predicted players in GW{EVAL_GW} (best combo):",
        top_n=15
    )

    # Residual analysis (per-player error this GW)
    df_named["error"] = df_named["predicted_points"] - df_named["actual_points"]
    df_named["abs_error"] = df_named["error"].abs()

    df_residuals_sorted = df_named.sort_values(
        by="abs_error", ascending=False
    ).copy()

    export_cols = [
        "season",
        "gameweek",
        PLAYER_ID_COL,
        "player_name",
        "team_short",
        "position",
        "predicted_points",
        "actual_points",
        "error",
        "abs_error",
    ]

    df_export = df_residuals_sorted[export_cols].reset_index(drop=True)

    print(
        f"\n=== Worst prediction errors in GW{EVAL_GW} "
        f"(sorted by absolute error) ==="
    )
    print(df_export.head(20))

    out_csv = PROJECT_ROOT / "data_processed" / "residuals" / f"model_residuals_gw{EVAL_GW}.csv"
    df_export.to_csv(out_csv, index=False)
    print(f"\nResidual analysis for GW{EVAL_GW} saved to: {out_csv}")


if __name__ == "__main__":
    main()
