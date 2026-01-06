# fpl-model

This is a local FPL (Fantasy Premier League) points prediction pipeline.

The goal: predict each player's FPL points for the *next* gameweek, then rank the best options overall and by position (GK / DEF / MID / FWD). I use this for transfers, captaincy, and bench decisions.

---

## How it works

Data comes from the public [`FPL-Elo-Insights`](https://github.com/olbauday/FPL-Elo-Insights) repo, which I have cloned locally. That dataset includes:
- per-player stats per gameweek
- team strength / Elo ratings
- fixtures by gameweek

This repo (`fpl-model`) does:
1. **`ingest.py`**  
   Combine all historical player/gameweek rows + fixtures into one table.  
   Output → `data_processed/training_base_raw.csv`

2. **`features_with_fixture.py`**  
   Add model features:
   - recent form (last GW points, avg points last 3, mins last 3, % games with 60+ mins played)
   - snapshot info (price, ownership%, FPL form)
   - fixture difficulty for that GW (home/away, team Elo, opponent Elo, opponent defence strength)  
   Output → `data_processed/training_table_with_fixture.csv`

3. **`train_with_fixture.py`**  
   Train a `RandomForestRegressor` to predict `event_points` (FPL points).  
   Report train/test MAE and show top predicted scorers for the last finished GW (sanity check).

4. **`predict_next_gw.py`**  
   Train on all completed GWs so far.  
   Build synthetic rows for the upcoming GW using latest form + next fixtures.  
   Print:
   - Top overall projected scorers
   - Top 10 GKs / DEFs / MIDs / FWDs for the upcoming GW  
   → This is what I actually use before the deadline.
5. This is not a complete and fine-tuned model yet. For example: I have not included latest data on injuries so the model doesn't know if, for eg: Isak is injured right now (as of Dec 19, 2025). In addition, the model does not account for injuries, performance of players in other tournaments, etc.
---

## 🗓 Latest Predictions
**Upcoming GW21 (model forecast)**
### Top 20 Overall 
```text
 1. Ekitiké (LIV, FWD)            15.210
 2. Lewis-Potter (BRE, DEF)       11.585
 3. Gusto (CHE, DEF)              10.410
 4. Hudson-Odoi (NFO, MID)        10.215
 5. Gvardiol (MCI, DEF)            9.960
 6. Rogers (AVL, MID)              9.630
 7. Haaland (MCI, FWD)             9.470
 8. Calvert-Lewin (LEE, FWD)       9.465
 9. Gordon (NEW, MID)              9.345
10. Chalobah (CHE, DEF)            9.195
11. Donnarumma (MCI, GK)           8.965
12. Hincapie (ARS, DEF)            8.870
13. Wilson (FUL, MID)              8.620
14. Sánchez (CHE, GK)              8.325
15. Stach (LEE, MID)               8.270
16. Fofana (CHE, DEF)              8.255
17. Foden (MCI, MID)               7.885
18. Matheus N. (MCI, DEF)          7.675
19. Woltemade (NEW, FWD)           7.515
20. Rúben (MCI, DEF)               7.445 
```
---

### Top 10 GoalKeepers
```text
 1. Donnarumma (MCI, GK)           8.965
 2. Sánchez (CHE, GK)              8.325
 3. Kelleher (BRE, GK)             7.085
 4. Roefs (SUN, GK)                5.120
 5. Leno (FUL, GK)                 4.110
 6. A.Becker (LIV, GK)             3.770
 7. Verbruggen (BHA, GK)           3.550
 8. Raya (ARS, GK)                 3.215
 9. Petrović (BOU, GK)             2.855
10. José Sá (WOL, GK)              2.635
```
---

### Top 10 Defenders
``` text
 1. Lewis-Potter (BRE, DEF)       11.585
 2. Gusto (CHE, DEF)              10.410
 3. Gvardiol (MCI, DEF)            9.960
 4. Chalobah (CHE, DEF)            9.195
 5. Hincapie (ARS, DEF)            8.870
 6. Fofana (CHE, DEF)              8.255
 7. Matheus N. (MCI, DEF)          7.675
 8. Rúben (MCI, DEF)               7.445
 9. O'Reilly (MCI, DEF)            7.240
10. James (CHE, DEF)               6.605
```
---

## Top 10 Midfielders
```text
 1. Hudson-Odoi (NFO, MID)        10.215
 2. Rogers (AVL, MID)              9.630
 3. Gordon (NEW, MID)              9.345
 4. Wilson (FUL, MID)              8.620
 5. Stach (LEE, MID)               8.270
 6. Foden (MCI, MID)               7.885
 7. Cunha (MUN, MID)               7.295
 8. Saka (ARS, MID)                7.000
 9. M.Fernandes (WHU, MID)         6.690
10. Tavernier (BOU, MID)           6.380
```
---

## Top 10 Forwards
```text
 1. Ekitiké (LIV, FWD)            15.210
 2. Haaland (MCI, FWD)             9.470
 3. Calvert-Lewin (LEE, FWD)       9.465
 4. Woltemade (NEW, FWD)           7.515
 5. Bowen (WHU, FWD)               6.520
 6. Richarlison (TOT, FWD)         5.200
 7. Nketiah (CRY, FWD)             4.220
 8. Watkins (AVL, FWD)             3.625
 9. Raúl (FUL, FWD)                3.315
10. Broja (BUR, FWD)               3.195
```
