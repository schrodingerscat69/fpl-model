# src/fpl_api_bootstrap.py
from pathlib import Path
import json, requests

CACHE = Path("data_processed/fpl_api_cache")
CACHE.mkdir(parents=True, exist_ok=True)
OUT = CACHE / "bootstrap-static.json"

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))

# tiny sanity print
print("Saved:", OUT)
print("players:", len(data.get("elements", [])), "teams:", len(data.get("teams", [])))

