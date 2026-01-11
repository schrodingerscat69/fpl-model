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
**Upcoming GW22 (model forecast)**
### Top 20 Overall 
```text
 1. Lewis-Potter (BRE, DEF)       10.695
 2. Ekitiké (LIV, FWD)            10.095
 3. Calvert-Lewin (LEE, FWD)       9.845
 4. Cherki (MCI, MID)              9.375
 5. Robinson (FUL, DEF)            9.300
 6. Dorgu (MUN, DEF)               9.270
 7. Rogers (AVL, MID)              9.005
 8. Haaland (MCI, FWD)             8.330
 9. James (CHE, DEF)               8.225
10. Saka (ARS, MID)                7.835
11. Gvardiol (MCI, DEF)            7.765
12. Ampadu (LEE, MID)              7.620
13. Gusto (CHE, DEF)               7.460
14. Semenyo (BOU, MID)             7.340
15. Hudson-Odoi (NFO, MID)         7.270
16. Watkins (AVL, FWD)             7.125
17. Wirtz (LIV, MID)               7.050
18. João Pedro (CHE, FWD)          6.775
19. Adingra (SUN, MID)             6.740
20. Hincapie (ARS, DEF)            6.680
```
---

### Top 10 GoalKeepers
```text
 1. Roefs (SUN, GK)                6.530
 2. Leno (FUL, GK)                 5.865
 3. Dúbravka (BUR, GK)             5.750
 4. Sánchez (CHE, GK)              5.365
 5. Lammens (MUN, GK)              4.930
 6. Donnarumma (MCI, GK)           4.890
 7. Kelleher (BRE, GK)             4.450
 8. Raya (ARS, GK)                 4.350
 9. A.Becker (LIV, GK)             3.965
10. Verbruggen (BHA, GK)           3.710
```
---

### Top 10 Defenders
``` text
 1. Lewis-Potter (BRE, DEF)       10.695
 2. Robinson (FUL, DEF)            9.300
 3. Dorgu (MUN, DEF)               9.270
 4. James (CHE, DEF)               8.225
 5. Gvardiol (MCI, DEF)            7.765
 6. Gusto (CHE, DEF)               7.460
 7. Hincapie (ARS, DEF)            6.680
 8. Dalot (MUN, DEF)               6.650
 9. Matheus N. (MCI, DEF)          6.585
10. Rúben (MCI, DEF)               6.535
```
---

## Top 10 Midfielders
```text
 1. Cherki (MCI, MID)              9.375
 2. Rogers (AVL, MID)              9.005
 3. Saka (ARS, MID)                7.835
 4. Ampadu (LEE, MID)              7.620
 5. Semenyo (BOU, MID)             7.340
 6. Hudson-Odoi (NFO, MID)         7.270
 7. Wirtz (LIV, MID)               7.050
 8. Adingra (SUN, MID)             6.740
 9. Gravenberch (LIV, MID)         6.380
10. Wilson (FUL, MID)              6.295
```
---

## Top 10 Forwards
```text
 1. Ekitiké (LIV, FWD)            10.095
 2. Calvert-Lewin (LEE, FWD)       9.845
 3. Haaland (MCI, FWD)             8.330
 4. Watkins (AVL, FWD)             7.125
 5. João Pedro (CHE, FWD)          6.775
 6. Tolu (WOL, FWD)                6.670
 7. Raúl (FUL, FWD)                5.885
 8. Woltemade (NEW, FWD)           5.235
 9. Thiago (BRE, FWD)              4.035
10. Richarlison (TOT, FWD)         3.800
```
