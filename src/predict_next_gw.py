from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")

RAW_FILE = PROJECT_ROOT / "data_processed" / "training_base_raw.csv"
TRAIN_TABLE_WITH_FIXTURE = PROJECT_ROOT / "data_processed" / "training_table_with_fixture.csv"

PLAYER_ID_COL = "id"

############################
# Helpers to load base data
############################

def load_training_history():
    """The historical training table (with fixture difficulty) we already built."""
    df_hist = pd.read_csv(TRAIN_TABLE_WITH_FIXTURE)
    df_hist["gameweek"] = df_hist["gameweek"].astype(int)
    return df_hist

def load_raw_player_rows():
    """Per-player per-finished-GW rows from ingest.py output."""
    df = pd.read_csv(RAW_FILE)
    df["gameweek"] = df["gameweek"].astype(int)
    return df

def build_player_team_map():
    """season, player_id -> team_code, position"""
    rows = []
    for season_dir in DATA_REPO_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name
        players_csv = season_dir / "players.csv"
        if not players_csv.is_file():
            continue
        players_df = pd.read_csv(players_csv)
        if "player_id" not in players_df.columns or "team_code" not in players_df.columns:
            continue
        tmp = players_df[["player_id", "team_code", "position"]].copy()
        tmp["season"] = season_name
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["team_code"] = pd.to_numeric(out["team_code"], errors="coerce")
    return out

def build_opponent_strength_lookup():
    """(season, team_code) -> opp_strength_defence_home/away from teams.csv"""
    rows = []
    for season_dir in DATA_REPO_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name
        teams_csv = season_dir / "teams.csv"
        if not teams_csv.is_file():
            continue
        teams_df = pd.read_csv(teams_csv)
        needed = ["code", "short_name", "strength_defence_home", "strength_defence_away"]
        if not all(c in teams_df.columns for c in needed):
            continue
        tmp = teams_df[needed].copy()
        tmp["season"] = season_name
        tmp = tmp.rename(columns={
            "code": "team_code",
            "short_name": "team_short"
        })
        tmp["team_code"] = pd.to_numeric(tmp["team_code"], errors="coerce")
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out["team_code"] = out["team_code"].astype(int)
    return out

def build_fixture_table():
    """
    Build long fixture table:
      season, gameweek, team_code, opp_code, is_home, team_elo, opp_elo
    Only keep Premier League fixtures for a given GW.
    Some GWs don't have 'tournament' column; fallback: check match_id contains 'prem'.
    """
    rows = []
    for season_dir in DATA_REPO_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name
        by_gw_dir = season_dir / "By Gameweek"
        if not by_gw_dir.is_dir():
            continue

        for gw_dir in by_gw_dir.iterdir():
            if not gw_dir.is_dir():
                continue
            if not gw_dir.name.startswith("GW"):
                continue
            try:
                gw_int = int(gw_dir.name[2:])
            except ValueError:
                continue

            fixtures_csv = gw_dir / "fixtures.csv"
            if not fixtures_csv.is_file():
                continue
            fx = pd.read_csv(fixtures_csv)

            # Filter to Premier League only
            if "tournament" in fx.columns:
                fx_pl = fx[fx["tournament"].astype(str).str.contains("prem", case=False, na=False)].copy()
            else:
                # fallback: use match_id string
                if "match_id" in fx.columns:
                    fx_pl = fx[fx["match_id"].astype(str).str.contains("prem", case=False, na=False)].copy()
                else:
                    fx_pl = fx.copy()

            if fx_pl.empty:
                continue

            # numeric-ify
            for col in ["home_team", "away_team", "home_team_elo", "away_team_elo"]:
                if col in fx_pl.columns:
                    fx_pl[col] = pd.to_numeric(fx_pl[col], errors="coerce")

            home_rows = pd.DataFrame({
                "season": season_name,
                "gameweek": gw_int,
                "team_code": fx_pl["home_team"],
                "opp_code": fx_pl["away_team"],
                "is_home": 1,
                "team_elo": fx_pl["home_team_elo"],
                "opp_elo": fx_pl["away_team_elo"],
            })
            away_rows = pd.DataFrame({
                "season": season_name,
                "gameweek": gw_int,
                "team_code": fx_pl["away_team"],
                "opp_code": fx_pl["home_team"],
                "is_home": 0,
                "team_elo": fx_pl["away_team_elo"],
                "opp_elo": fx_pl["home_team_elo"],
            })

            rows.append(home_rows)
            rows.append(away_rows)

    fixtures_long = pd.concat(rows, ignore_index=True)
    fixtures_long = fixtures_long.dropna(subset=["team_code", "opp_code"])
    fixtures_long["gameweek"] = fixtures_long["gameweek"].astype(int)
    fixtures_long["team_code"] = fixtures_long["team_code"].astype(int)
    fixtures_long["opp_code"] = fixtures_long["opp_code"].astype(int)
    fixtures_long["is_home"] = fixtures_long["is_home"].astype(int)
    return fixtures_long

def load_player_lookup(season_name: str):
    """For printing readable output at the end."""
    season_folder = DATA_REPO_ROOT / season_name
    players_csv = season_folder / "players.csv"
    teams_csv   = season_folder / "teams.csv"
    players = pd.read_csv(players_csv)
    teams   = pd.read_csv(teams_csv)

    team_lookup = teams.rename(
        columns={"code": "team_code", "short_name": "team_short"}
    )[["team_code", "team_short"]]

    players_enriched = players.merge(team_lookup, on="team_code", how="left")

    lookup = pd.DataFrame({
        "player_id": pd.to_numeric(players_enriched["player_id"], errors="coerce"),
        "player_name": players_enriched["web_name"].astype(str),
        "team_short": players_enriched["team_short"].astype(str),
        "position": players_enriched["position"].astype(str),
    })
    return lookup

#########################################
# Build the TRAINING dataset for the model
#########################################

def compute_rolling_features_for_history(df_raw):
    """
    Add historical rolling features up through each finished gameweek.
    This matches what we did in features_with_fixture.py.
    """
    df_raw = df_raw.sort_values(by=["season", PLAYER_ID_COL, "gameweek"]).copy()

    g = df_raw.groupby(["season", PLAYER_ID_COL], group_keys=False)

    df_raw["pts_prev_gw"] = g["event_points"].shift(1)
    df_raw["pts_avg_last3"] = g["event_points"].shift(1).rolling(window=3, min_periods=1).mean()
    df_raw["mins_avg_last3"] = g["minutes"].shift(1).rolling(window=3, min_periods=1).mean()
    played60 = (df_raw["minutes"] >= 60).astype(float)
    df_raw["played60_rate_last3"] = g[played60.name].shift(1).rolling(window=3, min_periods=1).mean()

    return df_raw

def attach_fixture_context(df_raw):
    """
    Merge team_code, then fixture info, then opponent defensive strength,
    and compute opp_def_strength.
    """
    # player -> team_code
    ptm = build_player_team_map().rename(columns={"player_id": PLAYER_ID_COL})
    df = df_raw.merge(
        ptm[["season", PLAYER_ID_COL, "team_code"]],
        on=["season", PLAYER_ID_COL],
        how="left"
    )

    fixtures_long = build_fixture_table()
    df = df.merge(
        fixtures_long,
        on=["season", "gameweek", "team_code"],
        how="left"
    )

    opp_strength = build_opponent_strength_lookup().rename(columns={
        "team_code": "opp_code",
        "strength_defence_home": "opp_strength_defence_home",
        "strength_defence_away": "opp_strength_defence_away",
    })

    df = df.merge(
        opp_strength,
        on=["season", "opp_code"],
        how="left"
    )

    def pick_def_strength(row):
        if pd.isna(row.get("is_home")):
            return np.nan
        if row["is_home"] == 1:
            return row["opp_strength_defence_away"]
        else:
            return row["opp_strength_defence_home"]

    df["opp_def_strength"] = df.apply(pick_def_strength, axis=1)

    return df

def build_train_frame_for_model(df_hist_full):
    """
    Return:
      train_df (finished GWs with targets),
      feature_cols
    """
    feature_cols = [
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
        "team_code",
        "opp_code",
    ]

    # clean snapshot text -> numeric
    for col in ["selected_by_percent", "form", "now_cost"]:
        if col in df_hist_full.columns:
            df_hist_full[col] = (
                df_hist_full[col]
                .astype(str)
                .str.replace("%", "", regex=False)
            )
            df_hist_full[col] = pd.to_numeric(df_hist_full[col], errors="coerce")

    # keep only rows where we actually have all the stuff
    train_df = df_hist_full[
        df_hist_full["pts_prev_gw"].notna()
        & df_hist_full["opp_def_strength"].notna()
        & df_hist_full["event_points"].notna()
    ].copy()

    return train_df, feature_cols

#########################################
# Build the PREDICTION dataset for the NEXT GW
#########################################

def build_next_gw_feature_rows(df_raw, season_current, last_gw, next_gw):
    """
    We synthesize "what the player looks like going into next_gw":
      - rolling stats using up to last_gw
      - latest snapshot (now_cost, form, selected_by_percent) from last_gw row
      - merge next_gw fixture info (is_home, opp_elo, etc.)
    """
    # Only rows up to last_gw for this season
    df_season_hist = df_raw[
        (df_raw["season"] == season_current) &
        (df_raw["gameweek"] <= last_gw)
    ].copy()

    # We'll compute aggregates per player
    rows = []
    for pid, g in df_season_hist.groupby(PLAYER_ID_COL):
        g_sorted = g.sort_values("gameweek")

        # last row = last_gw row for this player (latest snapshot)
        last_row = g_sorted.iloc[-1]
        last3 = g_sorted.tail(3)

        pts_prev_gw = last_row["event_points"]  # "previous GW" for next_gw
        pts_avg_last3 = last3["event_points"].mean()
        mins_avg_last3 = last3["minutes"].mean()
        played60_rate_last3 = (last3["minutes"] >= 60).mean()

        rows.append({
            "season": season_current,
            "gameweek": next_gw,
            PLAYER_ID_COL: pid,
            "pts_prev_gw": pts_prev_gw,
            "pts_avg_last3": pts_avg_last3,
            "mins_avg_last3": mins_avg_last3,
            "played60_rate_last3": played60_rate_last3,
            "now_cost": last_row.get("now_cost", np.nan),
            "selected_by_percent": last_row.get("selected_by_percent", np.nan),
            "form": last_row.get("form", np.nan),
        })

    df_next = pd.DataFrame(rows)

    # add team_code
    ptm = build_player_team_map().rename(columns={"player_id": PLAYER_ID_COL})
    df_next = df_next.merge(
        ptm[["season", PLAYER_ID_COL, "team_code"]],
        on=["season", PLAYER_ID_COL],
        how="left"
    )

    # add fixture info for next_gw
    fixtures_long = build_fixture_table()
    df_next = df_next.merge(
        fixtures_long[
            (fixtures_long["season"] == season_current)
            & (fixtures_long["gameweek"] == next_gw)
        ],
        on=["season", "gameweek", "team_code"],
        how="left",
        suffixes=("", "_fixture")
    )

    # opponent defensive strength lookup
    opp_strength = build_opponent_strength_lookup().rename(columns={
        "team_code": "opp_code",
        "strength_defence_home": "opp_strength_defence_home",
        "strength_defence_away": "opp_strength_defence_away",
    })
    df_next = df_next.merge(
        opp_strength[["season", "opp_code", "opp_strength_defence_home", "opp_strength_defence_away"]],
        on=["season", "opp_code"],
        how="left"
    )

    # compute opp_def_strength the same way
    def pick_def_strength(row):
        if pd.isna(row.get("is_home")):
            return np.nan
        if row["is_home"] == 1:
            return row["opp_strength_defence_away"]
        else:
            return row["opp_strength_defence_home"]

    df_next["opp_def_strength"] = df_next.apply(pick_def_strength, axis=1)

    # clean snapshot text -> numeric
    for col in ["selected_by_percent", "form", "now_cost"]:
        df_next[col] = (
            df_next[col]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
        df_next[col] = pd.to_numeric(df_next[col], errors="coerce")

    # keep only guys who actually HAVE a PL fixture
    df_next = df_next[df_next["is_home"].notna()].copy()

    return df_next

############################
# Pretty-print
############################

def print_top_by_position(preds_named, top_n=10):
    order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    for pos in order:
        sub = preds_named[preds_named["position"] == pos].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by="predicted_points", ascending=False).head(top_n)
        print(f"\n=== Top {top_n} {pos}s (next GW prediction) ===")
        cols = ["player_name", "team_short", "predicted_points"]
        print(sub[cols].reset_index(drop=True))

############################
# Main
############################

def main():
    # 1. load history to know last completed GW + season
    df_hist = load_training_history()
    season_current = sorted(df_hist["season"].unique())[-1]
    df_season_hist = df_hist[df_hist["season"] == season_current].copy()
    last_gw = df_season_hist["gameweek"].max() # -1 to test predictions of older weeks
    next_gw = last_gw + 1

    print(f"Season: {season_current}, last finished GW: {last_gw}, next GW to predict: {next_gw}")

    # 2. load raw rows (player stats per finished GW)
    df_raw = load_raw_player_rows()

    # 3. create a historical frame WITH fixture context + rolling, for training
    #    (only up to last_gw, because after that we don't know event_points)
    df_hist_full = df_raw.copy()
    df_hist_full = attach_fixture_context(df_hist_full)
    df_hist_full = compute_rolling_features_for_history(df_hist_full)

    train_df, feature_cols = build_train_frame_for_model(df_hist_full)

    # only keep rows up to last_gw
    train_df = train_df[
        (train_df["season"] == season_current) &
        (train_df["gameweek"] <= last_gw)
    ].copy()

    if train_df.empty:
        print("Training frame is empty. Something's wrong with historical data.")
        return

    # 4. train model on all completed GWs
    X_train = train_df[feature_cols].copy()
    y_train = train_df["event_points"].copy()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. build synthetic NEXT GW rows
    df_next = build_next_gw_feature_rows(df_raw, season_current, last_gw, next_gw)
    if df_next.empty:
        print("No next-GW prediction frame. Do we have PL fixtures for that GW?")
        return

    # same feature order
    X_next = df_next[feature_cols].copy()
    df_next["predicted_points"] = model.predict(X_next)

    # 6. attach readable names/positions
    lookup = load_player_lookup(season_current)
    preds_named = df_next.merge(
        lookup,
        left_on=PLAYER_ID_COL,
        right_on="player_id",
        how="left"
    )

    # 7. sort overall by predicted points (high to low)
    preds_named = preds_named.sort_values(by="predicted_points", ascending=False)

    print("\nTop overall predicted players for next GW (pre-deadline):")
    print(
        preds_named[
            ["player_name", "team_short", "position", "predicted_points"]
        ].head(20).reset_index(drop=True)
    )

    # per-position captain/transfer help
    print_top_by_position(preds_named, top_n=10)

    # 8. SAVE PREDICTIONS TO CSV FOR THIS GW
    # build a clean table to persist
    df_preds_all = preds_named[[
        "gameweek",
        "player_name",
        "team_short",
        "position",
        "predicted_points"
    ]].copy()

    # create predictions/ dir if not exists
    predictions_dir = PROJECT_ROOT / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    # file name = gw<next_gw>_predictions.csv
    outfile = predictions_dir / f"gw{next_gw}_predictions.csv"
    df_preds_all.to_csv(outfile, index=False)
    print(f"\nSaved predictions to {outfile}")


if __name__ == "__main__":
    main()
