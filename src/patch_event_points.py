# src/patch_event_points_all.py
from pathlib import Path
import re
import pandas as pd

# ----- anchor paths to repo root -----
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
DATA_DIR = PROJECT_ROOT / "data_processed"
IN_TRAIN = DATA_DIR / "training_base_raw.csv"     # <- will be OVERWRITTEN atomically
POINTS_DIR = DATA_DIR / "actual-points"

GW_RE = re.compile(r"^gw(\d+)-points\.csv$", re.IGNORECASE)

def detect_current_season(df: pd.DataFrame) -> str:
    return df["season"].dropna().sort_values().iloc[-1]

def main():
    if not IN_TRAIN.is_file():
        raise SystemExit(f"Missing input table: {IN_TRAIN}")
    if not POINTS_DIR.is_dir():
        raise SystemExit(f"Missing points dir: {POINTS_DIR}")

    df = pd.read_csv(IN_TRAIN)

    # limit to current season only
    current_season = detect_current_season(df)
    df_cur_mask = (df["season"] == current_season)
    print(f"Current season detected: {current_season}")
    print(f"Rows in current season: {int(df_cur_mask.sum())}")

    if "event_points_orig" not in df.columns:
        df["event_points_orig"] = pd.NA

    total_updated = 0
    total_with_api = 0

    for f in sorted(POINTS_DIR.iterdir()):
        m = GW_RE.match(f.name)
        if not m:
            continue
        gw = int(m.group(1))

        api = pd.read_csv(f).rename(columns={"player_id": "id"})
        api = api[["id", "gameweek", "event_points"]].rename(columns={"event_points": "api_points"})

        cur = df.loc[df_cur_mask].merge(api, on=["id", "gameweek"], how="left")
        has_api = cur["api_points"].notna()
        need_update = has_api & (cur["api_points"] > 0) & (cur["event_points"] == 0)

        gw_with_api = int(has_api.sum())
        gw_updates = int(need_update.sum())

        if gw_updates:
            idx = cur.index[need_update]                         # indices within the masked frame
            target_idx = df.loc[df_cur_mask].iloc[idx].index     # map back to original df indices
            df.loc[target_idx, "event_points_orig"] = df.loc[target_idx, "event_points"]
            df.loc[target_idx, "event_points"] = cur.loc[need_update, "api_points"].astype(int).values

        total_with_api += gw_with_api
        total_updated += gw_updates
        print(f"GW{gw}: rows with API data = {gw_with_api}, updated (0 -> API) = {gw_updates}")

    # atomic overwrite
    tmp = IN_TRAIN.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(IN_TRAIN)

    print("\nSummary:")
    print(f"Total rows with API data (current season): {total_with_api}")
    print(f"Total rows updated: {total_updated}")
    print(f"Overwrote {IN_TRAIN}")

if __name__ == "__main__":
    main()

