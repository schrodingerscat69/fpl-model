from pathlib import Path
import pandas as pd
import numpy as np

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

# input from previous ingest step
RAW_FILE = PROJECT_ROOT / "data_processed" / "training_base_raw.csv"

# output for this new enhanced table
OUT_FILE = PROJECT_ROOT / "data_processed" / "training_table_with_fixture.csv"

# path to cloned data repo
DATA_REPO_ROOT = Path("/home/mann-gandhi/FPL-Elo-Insights-data/data")

PLAYER_ID_COL = "id"  # in training_base_raw.csv


def load_raw():
    """
    Load the concatenated per-player-per-gameweek data we built in ingest.py.
    This has columns like:
    - id (player id)
    - season
    - gameweek
    - event_points
    - now_cost, form, selected_by_percent, minutes, etc.
    """
    df = pd.read_csv(RAW_FILE)
    # make sure gameweek is int not float
    df["gameweek"] = df["gameweek"].astype(int)
    return df


def build_player_team_map():
    """
    For each season, read that season's players.csv to get:
      player_id -> team_code
    Returns a single DataFrame with columns:
      season, player_id, team_code
    """
    rows = []

    for season_dir in DATA_REPO_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name

        players_csv = season_dir / "players.csv"
        if not players_csv.is_file():
            continue

        players_df = pd.read_csv(players_csv)

        # expected columns:
        # player_id, team_code, position, web_name, ...
        if "player_id" not in players_df.columns or "team_code" not in players_df.columns:
            continue

        tmp = players_df[["player_id", "team_code"]].copy()
        tmp["season"] = season_name
        rows.append(tmp)

    if not rows:
        raise RuntimeError("No players.csv files found / parsed.")

    out = pd.concat(rows, ignore_index=True)

    # normalize types
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["team_code"] = pd.to_numeric(out["team_code"], errors="coerce")
    return out


def build_fixture_table():
    """
    Build a table of (season, gameweek, team_code) -> opponent attributes.

    For each season and GW, we'll read fixtures.csv and create 2 rows per match:
      - one row for the home team
      - one row for the away team

    We'll keep only matches where tournament contains 'prem' (Premier League),
    because that's the fixture that gives FPL points.

    Output columns:
      season
      gameweek (int)
      team_code        (this team's code)
      opp_code         (opponent's code)
      is_home          (1 if this team is home)
      team_elo         (elo for this team going into that match)
      opp_elo          (elo for opponent going into that match)
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

            gw_str = gw_dir.name[2:]
            try:
                gw_int = int(gw_str)
            except ValueError:
                continue

            fixtures_csv = gw_dir / "fixtures.csv"
            if not fixtures_csv.is_file():
                continue

            fx = pd.read_csv(fixtures_csv)

            # We expect columns like:
            # gameweek, home_team, home_team_elo, away_team, away_team_elo, tournament, ...
            # (your sample shows 'tournament' as last col)
            # Filter to matches that look like Premier League.
            if "tournament" in fx.columns:
                fx_pl = fx[fx["tournament"].astype(str).str.contains("prem", case=False, na=False)].copy()
            else:
                # worst case: if no tournament column, we just keep everything
                fx_pl = fx.copy()

            if fx_pl.empty:
                continue

            # make sure numeric
            for col in ["home_team", "away_team", "home_team_elo", "away_team_elo"]:
                if col in fx_pl.columns:
                    fx_pl[col] = pd.to_numeric(fx_pl[col], errors="coerce")

            # Create one row for home team
            home_rows = pd.DataFrame({
                "season": season_name,
                "gameweek": gw_int,
                "team_code": fx_pl["home_team"],
                "opp_code": fx_pl["away_team"],
                "is_home": 1,
                "team_elo": fx_pl["home_team_elo"],
                "opp_elo": fx_pl["away_team_elo"],
            })

            # One row for away team
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

    if not rows:
        raise RuntimeError("No fixtures.csv info collected. Check structure / tournaments names.")

    fixtures_long = pd.concat(rows, ignore_index=True)

    # Drop any rows missing team_code or opp_code (NaN from non-PL or malformed lines)
    fixtures_long = fixtures_long.dropna(subset=["team_code", "opp_code"])

    # Ensure ints where appropriate
    fixtures_long["gameweek"] = fixtures_long["gameweek"].astype(int)
    fixtures_long["team_code"] = fixtures_long["team_code"].astype(int)
    fixtures_long["opp_code"] = fixtures_long["opp_code"].astype(int)
    fixtures_long["is_home"] = fixtures_long["is_home"].astype(int)

    return fixtures_long


def build_opponent_strength_lookup():
    """
    Build a table:
      (season, team_code) -> strength_defence_home, strength_defence_away
    from teams.csv for each season.
    """
    rows = []

    for season_dir in DATA_REPO_ROOT.iterdir():
        if not season_dir.is_dir():
            continue
        season_name = season_dir.name

        teams_csv = season_dir / "teams.csv"
        if not teams_csv.is_file():
            continue

        teams_df = pd.read_csv(teams_csv)

        # teams.csv columns include:
        # code,short_name,strength_defence_home,strength_defence_away,elo,...
        needed_cols = [
            "code",
            "strength_defence_home",
            "strength_defence_away",
        ]
        if not all(c in teams_df.columns for c in needed_cols):
            continue

        tmp = teams_df[needed_cols].copy()
        tmp["season"] = season_name
        tmp = tmp.rename(columns={"code": "team_code"})

        # make sure numeric
        tmp["team_code"] = pd.to_numeric(tmp["team_code"], errors="coerce")
        tmp["strength_defence_home"] = pd.to_numeric(tmp["strength_defence_home"], errors="coerce")
        tmp["strength_defence_away"] = pd.to_numeric(tmp["strength_defence_away"], errors="coerce")

        rows.append(tmp)

    if not rows:
        raise RuntimeError("No teams.csv data found for strength_defence_*.")

    out = pd.concat(rows, ignore_index=True)

    # team_code could be float -> int
    out["team_code"] = out["team_code"].astype(int)

    return out


def add_fixture_features(df_raw):
    """
    Add team_code to each player row, merge fixture info, and compute opponent difficulty.
    Returns df_raw_enriched.
    """

    # 1. Map each player (by id) to team_code for that season
    player_team_map = build_player_team_map()
    # rename to merge with df_raw
    player_team_map = player_team_map.rename(columns={
        "player_id": PLAYER_ID_COL
    })

    df = df_raw.merge(
        player_team_map[["season", PLAYER_ID_COL, "team_code"]],
        on=["season", PLAYER_ID_COL],
        how="left"
    )

    # Make sure team_code is numeric/int
    df["team_code"] = pd.to_numeric(df["team_code"], errors="coerce").astype("Int64")

    # 2. Merge in fixture rows for Premier League games
    fixtures_long = build_fixture_table()

    df = df.merge(
        fixtures_long,
        on=["season", "gameweek", "team_code"],
        how="left"
    )

    # 3. Merge opponent defensive strength from opponent team_code
    #    We'll join opponent info (opp_code) against teams.csv lookup.
    opp_strength_lookup = build_opponent_strength_lookup()
    opp_strength_lookup = opp_strength_lookup.rename(columns={
        "team_code": "opp_code",
        "strength_defence_home": "opp_strength_defence_home",
        "strength_defence_away": "opp_strength_defence_away",
    })

    df = df.merge(
        opp_strength_lookup,
        on=["season", "opp_code"],
        how="left"
    )

    # 4. Compute an "effective opponent defence strength"
    #    Logic:
    #    - if we are home, the opponent is away, so use opponent's away defence strength
    #    - if we are away, opponent is home, so use opponent's home defence strength
    def pick_def_strength(row):
        if pd.isna(row.get("is_home")):
            return np.nan
        if row["is_home"] == 1:
            return row["opp_strength_defence_away"]
        else:
            return row["opp_strength_defence_home"]

    df["opp_def_strength"] = df.apply(pick_def_strength, axis=1)

    return df


def add_rolling_features(df):
    """
    Adds rolling/lag features for each player, based only on PAST gameweeks.
    We compute both last-3 and last-5 style features.

    New columns (examples):
      pts_avg_last5
      mins_avg_last5
      played60_rate_last5
      xgi_avg_last3, xgi_avg_last5
      shots_last3, shots_last5
      cs_rate_last3, cs_rate_last5
      gc_avg_last3, gc_avg_last5
    """

    df = df.sort_values(by=["season", PLAYER_ID_COL, "gameweek"]).copy()

    g = df.groupby(["season", PLAYER_ID_COL], group_keys=False)

    # Base lag stuff (already had)
    df["pts_prev_gw"] = g["event_points"].shift(1)

    df["pts_avg_last3"] = (
        g["event_points"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["pts_avg_last5"] = (
        g["event_points"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )

    df["mins_avg_last3"] = (
        g["minutes"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["mins_avg_last5"] = (
        g["minutes"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # Played >=60 proxy for nailedness
    played60 = (df["minutes"] >= 60).astype(float)

    df["played60_rate_last3"] = (
        g[played60.name]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["played60_rate_last5"] = (
        g[played60.name]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # xGI = xG + xA
    if "expected_goals" in df.columns and "expected_assists" in df.columns:
        df["xgi"] = df["expected_goals"].fillna(0) + df["expected_assists"].fillna(0)

        df["xgi_avg_last3"] = (
            g["xgi"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )

        df["xgi_avg_last5"] = (
            g["xgi"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    else:
        df["xgi_avg_last3"] = np.nan
        df["xgi_avg_last5"] = np.nan

    # total shots rolling
    if "goals_scored" in df.columns and "assists" in df.columns and "goals_conceded" in df.columns:
        # if you have "total_shots" use that; if not, skip this block
        if "total_shots" in df.columns:
            df["shots_last3"] = (
                g["total_shots"]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .sum()
            )

            df["shots_last5"] = (
                g["total_shots"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .sum()
            )
        else:
            df["shots_last3"] = np.nan
            df["shots_last5"] = np.nan
    else:
        df["shots_last3"] = np.nan
        df["shots_last5"] = np.nan

    # defensive trend features:
    # clean_sheets, goals_conceded
    if "clean_sheets" in df.columns:
        df["cs_rate_last3"] = (
            g["clean_sheets"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df["cs_rate_last5"] = (
            g["clean_sheets"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    else:
        df["cs_rate_last3"] = np.nan
        df["cs_rate_last5"] = np.nan

    if "goals_conceded" in df.columns:
        df["gc_avg_last3"] = (
            g["goals_conceded"]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        df["gc_avg_last5"] = (
            g["goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )
    else:
        df["gc_avg_last3"] = np.nan
        df["gc_avg_last5"] = np.nan

    return df



def build_training_table_with_fixture(df_raw_enriched):
    """
    Build final ML table with:
    - rolling features
    - snapshot features (now_cost, form, etc.)
    - fixture difficulty features

    Output columns are similar to training_table.csv, but now include:
      is_home, team_elo, opp_elo, opp_def_strength
    """

    df = add_rolling_features(df_raw_enriched)

    target_col = "event_points"

    base_feature_cols = [
        # rolling / form
        "pts_prev_gw",
        "pts_avg_last3",
        "mins_avg_last3",
        "played60_rate_last3",

        # snapshot info we know pre-deadline
        "now_cost",
        "selected_by_percent",
        "form",

        # fixture difficulty
        "is_home",
        "team_elo",
        "opp_elo",
        "opp_def_strength",

        # You *could* also include team_code and opp_code as numeric IDs,
        # which lets the model learn "playing vs BUR is good".
        # We'll include them since RF can handle ints fine.
        "team_code",
        "opp_code",

        # we carry minutes for debug only (we'll drop before training in train.py)
        "minutes",
    ]

    # keep only columns that actually exist
    feature_cols = [c for c in base_feature_cols if c in df.columns]

    id_cols = ["season", "gameweek", PLAYER_ID_COL]

    model_df = df[id_cols + feature_cols + [target_col]].copy()

    # drop rows with no past data (no lag)
    if "pts_prev_gw" in model_df.columns:
        model_df = model_df[model_df["pts_prev_gw"].notna()].copy()

    # drop rows where we didn't get fixture info (opp_def_strength missing etc.)
    # This will naturally restrict to PL rows that actually scored FPL points.
    if "opp_def_strength" in model_df.columns:
        model_df = model_df[model_df["opp_def_strength"].notna()].copy()

    # drop rows with missing target
    model_df = model_df[model_df[target_col].notna()].copy()

    return model_df


def main():
    df_raw = load_raw()
    df_enriched = add_fixture_features(df_raw)

    # Save a preview of what columns we ended up with, for debugging
    print("Columns after fixture merge:")
    print(sorted(df_enriched.columns.tolist()))
    print()

    training_df = build_training_table_with_fixture(df_enriched)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(OUT_FILE, index=False)

    print("training_table_with_fixture.csv shape:", training_df.shape)
    print("Wrote:", OUT_FILE)


if __name__ == "__main__":
    main()
