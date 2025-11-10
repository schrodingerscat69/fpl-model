# src/fpl_api_gw_points.py
from pathlib import Path
import json
import sys
import requests
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent          # .../fpl-model
DATA_DIR = PROJECT_ROOT / "data_processed"
CACHE = DATA_DIR / "fpl_api_cache"
BOOTSTRAP = CACHE / "bootstrap-static.json"
OUTDIR = DATA_DIR / "actual-points"

def load_bootstrap():
    data = json.loads(BOOTSTRAP.read_text())
    # player id -> web_name, team_id
    players = {e["id"]: {"web_name": e["web_name"], "team": e["team"]} for e in data["elements"]}
    # team id -> short_name
    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    return players, teams

def fetch_gw_live(gw: int):
    url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def build_points_df(gw: int, live_json, players, teams):
    rows = []
    for el in live_json.get("elements", []):
        pid = el["id"]                           # FPL player (element) id
        pts = el["stats"]["total_points"]        # event points for this GW
        pmeta = players.get(pid, {})
        team_id = pmeta.get("team")
        rows.append({
            "player_id": pid,
            "player_short_name": pmeta.get("web_name"),
            "team_short_name": teams.get(team_id),
            "gameweek": gw,
            "event_points": pts
        })
    return pd.DataFrame(rows, columns=["player_id","player_short_name","team_short_name","gameweek","event_points"])

def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("Usage: python src/fpl_api_gw_points.py <GW_NUMBER>")
        sys.exit(2)
    gw = int(sys.argv[1])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    players, teams = load_bootstrap()
    live = fetch_gw_live(gw)
    df = build_points_df(gw, live, players, teams)

    out_path = OUTDIR / f"gw{gw}-points.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()

