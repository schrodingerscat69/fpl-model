from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone  # <-- important fix


# ---------- paths / constants ----------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # repo root
DATA_FILE = PROJECT_ROOT / "data_processed" / "training_table_with_fixture.csv"

PLAYER_ID_COL = "id"
TARGET_COL = "event_points"

DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")


# ---------- helpers ----------

def clean_snapshot_columns(df: pd.DataFrame):
    """
    Convert snapshot cols like 'selected_by_percent' ('12.3%')
    and 'form' (string) to numeric.
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


def make_train_test_split(df_season: pd.DataFrame, cutoff_gw: int):
    """
    train = rows with gameweek <= cutoff_gw
    test  = rows with gameweek >  cutoff_gw
    """
    df_train = df_season[df_season["gameweek"] <= cutoff_gw].copy()
    df_test  = df_season[df_season["gameweek"] >  cutoff_gw].copy()
    return df_train, df_test


def build_xy(df_subset: pd.DataFrame, feature_cols):
    """
    Build X, y, and return the filtered df that lines up with them.
    We drop rows that are missing any required feature or the target.
    """
    cols_needed = feature_cols + [TARGET_COL]
    df_model = df_subset.dropna(subset=cols_needed).copy()

    X = df_model[feature_cols].copy()
    y = df_model[TARGET_COL].copy()

    return X, y, df_model


def train_and_eval_model(
    base_model,
    X_train, y_train,
    X_test, y_test,
    model_name: str,
    feature_set_name: str,
    n_features_used: int,
):
    """
    Clone model, fit it, compute MAEs. Return scores dict + fitted clone.
    """
    # clone -> brand new estimator each time
    model = clone(base_model)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features_used": n_features_used,
        "train_mae": mae_train,
        "test_mae": mae_test,
        "fitted_model": model,
    }


def attach_player_print_info(df_rows: pd.DataFrame, season_current: str):
    """
    Enrich df_rows with player_name, team_short, position.
    """
    season_dir = DATA_REPO_ROOT / season_current

    players_df = pd.read_csv(season_dir / "players.csv")
    teams_df = pd.read_csv(season_dir / "teams.csv")

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
    Show top predicted scorers in the test split.
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


# ---------- main ----------

def main():
    # 1. load processed training table (already merged with fixture info)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {DATA_FILE.name} with shape: {df.shape}")
    print("Columns:", list(df.columns))

    # 2. clean snapshot % / strings
    df = clean_snapshot_columns(df)

    # 3. numeric / helper features
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")
    df["season"] = df["season"].astype(str)

    # interaction-ish engineered features that should be knowable pre-deadline:
    # - elo_diff = team_elo - opp_elo
    # - form_x_home = form * is_home
    df["elo_diff"] = df["team_elo"] - df["opp_elo"]
    df["form_x_home"] = df["form"] * df["is_home"]

    # 4. focus on latest season only
    seasons = sorted(df["season"].unique())
    season_current = seasons[-1]
    df_season = df[df["season"] == season_current].copy()

    # 5. choose cutoff gameweek for holdout
    gws = sorted(df_season["gameweek"].dropna().unique())
    print("\nGameweeks in this season:", gws)

    cutoff_gw = 8
    print(f"Using cutoff GW={cutoff_gw}:")
    print("    train = gameweek <= ", cutoff_gw)
    print("    test  = gameweek >  ", cutoff_gw)

    df_train, df_test = make_train_test_split(df_season, cutoff_gw)

    # 6. define feature sets
    baseline_features = [
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

    extended_basic_features = baseline_features + [
        "team_code",
        "opp_code",
    ]

    extended_interact_features = extended_basic_features + [
        "elo_diff",
        "form_x_home",
    ]

    feature_sets = {
        "baseline": baseline_features,
        "extended_basic": extended_basic_features,
        "extended_interact": extended_interact_features,
    }

    # models to evaluate
    model_defs = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            random_state=42,
        ),
    }

    all_results = []
    combo_artifacts = []

    # 7. train/eval each (feature_set × model)
    for fs_name, fs_cols in feature_sets.items():
        # dedupe in case of accidental repeats
        fs_cols = list(dict.fromkeys(fs_cols))

        X_train, y_train, df_train_used = build_xy(df_train, fs_cols)
        X_test, y_test, df_test_used = build_xy(df_test, fs_cols)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"\n[WARN] Feature set '{fs_name}' has insufficient data, skipping.")
            continue

        print(f"\nFeature set '{fs_name}' using {len(fs_cols)} cols:")
        for c in fs_cols:
            print(" ", c)

        for model_name, base_model in model_defs.items():
            scores = train_and_eval_model(
                base_model=base_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                feature_set_name=fs_name,
                n_features_used=len(fs_cols),
            )

            all_results.append(scores)

            combo_artifacts.append({
                "feature_set": fs_name,
                "feature_cols": fs_cols,
                "model_name": model_name,
                "fitted_model": scores["fitted_model"],  # this is the cloned & fitted one
                "df_test_used": df_test_used.copy(),
            })

    # 8. summarize results
    results_table = pd.DataFrame(all_results)
    # sort for readability (by feature_set then model)
    results_table = results_table.sort_values(
        by=["feature_set", "model"]
    ).reset_index(drop=True)

    print("\n=== Model Benchmark by Feature Set (MAE lower = better) ===")
    print(results_table[["feature_set", "model", "n_features_used", "train_mae", "test_mae"]])

    # choose best (lowest test_mae)
    best_idx = results_table["test_mae"].astype(float).idxmin()
    best_row = results_table.loc[best_idx]

    best_feature_set = best_row["feature_set"]
    best_model_name = best_row["model"]
    best_test_mae = best_row["test_mae"]

    print(
        f"\nBest overall by test MAE: {best_model_name} on feature_set='{best_feature_set}' "
        f"(test MAE={best_test_mae:.4f})"
    )

    # 9. find the saved artifact for that combo
    best_combo = None
    for art in combo_artifacts:
        if art["feature_set"] == best_feature_set and art["model_name"] == best_model_name:
            best_combo = art
            break

    if best_combo is None:
        print("\n[WARN] couldn't retrieve best combo artifact to preview players.")
        return

    best_model = best_combo["fitted_model"]
    best_feature_cols = best_combo["feature_cols"]
    df_test_used = best_combo["df_test_used"].copy()

    # 10. generate predictions on that test split
    df_test_used["predicted_points"] = best_model.predict(df_test_used[best_feature_cols])
    df_test_used["actual_points"] = df_test_used[TARGET_COL]

    # 11. add readable player/team info, show top players
    df_named = attach_player_print_info(df_test_used, season_current)

    print_top_players(
        df_named,
        header="Top predicted players in the test set (best combo):",
        top_n=15
    )


if __name__ == "__main__":
    main()
