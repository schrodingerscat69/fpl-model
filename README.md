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
**Upcoming GW23 (model forecast)**
### Top 20 Overall 
```text
 1. Schade (BRE, MID)             10.995
 2. Garner (EVE, MID)             10.010
 3. Gravenberch (LIV, MID)         9.535
 4. Cherki (MCI, MID)              8.875
 5. Van de Ven (TOT, DEF)          8.850
 6. Watkins (AVL, FWD)             8.635
 7. Gabriel (ARS, DEF)             8.150
 8. Ødegaard (ARS, MID)            7.530
 9. Trossard (ARS, MID)            7.520
10. Laurent (BUR, MID)             7.450
11. Robinson (FUL, DEF)            7.140
12. L.Paquetá (WHU, MID)           6.915
13. Ajer (BRE, DEF)                6.855
14. Mukiele (SUN, DEF)             6.755
15. O'Brien (EVE, DEF)             6.730
16. J.Cuenca (FUL, DEF)            6.670
17. Mateta (CRY, FWD)              6.620
18. Vicario (TOT, GK)              6.605
19. Mykolenko (EVE, DEF)           6.530
20. Spence (TOT, DEF)              6.475
```
---

### Top 10 GoalKeepers
```text
 1. Vicario (TOT, GK)              6.605
 2. Perri (LEE, GK)                6.075
 3. Leno (FUL, GK)                 5.825
 4. Pickford (EVE, GK)             5.670
 5. Dúbravka (BUR, GK)             5.605
 6. Roefs (SUN, GK)                5.560
 7. Donnarumma (MCI, GK)           4.980
 8. A.Becker (LIV, GK)             4.905
 9. Lammens (MUN, GK)              4.735
10. Kelleher (BRE, GK)             4.065
```
---

### Top 10 Defenders
``` text
 1. Van de Ven (TOT, DEF)          8.850
 2. Gabriel (ARS, DEF)             8.150
 3. Robinson (FUL, DEF)            7.140
 4. Ajer (BRE, DEF)                6.855
 5. Mukiele (SUN, DEF)             6.755
 6. O'Brien (EVE, DEF)             6.730
 7. J.Cuenca (FUL, DEF)            6.670
 8. Mykolenko (EVE, DEF)           6.530
 9. Spence (TOT, DEF)              6.475
10. Pedro Porro (TOT, DEF)         6.425
```
---

## Top 10 Midfielders
```text
 1. Schade (BRE, MID)             10.995
 2. Garner (EVE, MID)             10.010
 3. Gravenberch (LIV, MID)         9.535
 4. Cherki (MCI, MID)              8.875
 5. Ødegaard (ARS, MID)            7.530
 6. Trossard (ARS, MID)            7.520
 7. Laurent (BUR, MID)             7.450
 8. L.Paquetá (WHU, MID)           6.915
 9. Wirtz (LIV, MID)               5.985
10. Hutchinson (NFO, MID)          5.945
```
---

## Top 10 Forwards
```text
 1. Watkins (AVL, FWD)             8.635
 2. Mateta (CRY, FWD)              6.620
 3. Broja (BUR, FWD)               5.415
 4. Thiago (BRE, FWD)              5.285
 5. Raúl (FUL, FWD)                5.175
 6. Ekitiké (LIV, FWD)             4.260
 7. Calvert-Lewin (LEE, FWD)       3.945
 8. Wissa (NEW, FWD)               3.680
 9. Richarlison (TOT, FWD)         3.585
10. Bowen (WHU, FWD)               3.510
```
