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
5. This is not a complete and fine-tuned model yet. For example: I have not included latest data on injuries so the model doesn't know if, for eg: Martinelli is injured right now (as of Nov 2, 2025). In addition, the model does not account for injuries, performance of players in other tournaments, etc.
---

## 🗓 Latest Predictions
**Upcoming GW11 (model forecast)**
### Top 20 Overall 
```text
 1. Casemiro (MUN, MID)            10.440
 2. Mbeumo (MUN, MID)               7.860
 3. J.Timber (ARS, DEF)             7.320
 4. Raya (ARS, GK)                  7.235
 5. Neto (CHE, MID)                 7.095
 6. Calafiori (ARS, DEF)            6.795
 7. Sessegnon (FUL, MID)            6.750
 8. Cash (AVL, DEF)                 6.440
 9. Gabriel (ARS, DEF)              6.435
10. Garnacho (CHE, MID)             6.200
11. Rice (ARS, MID)                 6.190
12. Mateta (CRY, FWD)               6.015
13. Amad (MUN, MID)                 5.810
14. M.Salah (LIV, MID)              5.630
15. Gomez (BHA, MID)                5.520
16. Szoboszlai (LIV, MID)           5.510
17. Mitchell (CRY, DEF)             5.415
18. Martinez (AVL, GK)              5.410
19. Gravenberch (LIV, MID)          5.260
20. Haaland (MCI, FWD)              5.165
```
---

### Top 10 GoalKeepers
```text
1. Raya (ARS, GK)                  7.235
 2. Martinez (AVL, GK)              5.410
 3. Sánchez (CHE, GK)               4.505
 4. Verbruggen (BHA, GK)            4.360
 5. Henderson (CRY, GK)             4.285
 6. Mamardashvili (LIV, GK)         3.155
 7. Vicario (TOT, GK)               2.395
 8. Petrović (BOU, GK)              2.335
 9. Pickford (EVE, GK)              2.085
10. Donnarumma (MCI, GK)            2.085
```
---

### Top 10 Defenders
``` text
 1. J.Timber (ARS, DEF)             7.320
 2. Calafiori (ARS, DEF)            6.795
 3. Cash (AVL, DEF)                 6.440
 4. Gabriel (ARS, DEF)              6.435
 5. Mitchell (CRY, DEF)             5.415
 6. O’Reilly (MCI, DEF)             5.055
 7. Lacroix (CRY, DEF)              4.945
 8. Gusto (CHE, DEF)                4.755
 9. Van de Ven (TOT, DEF)           4.690
10. Guéhi (CRY, DEF)                4.470
```
---

## Top 10 Midfielders
```text
 1. Casemiro (MUN, MID)            10.440
 2. Mbeumo (MUN, MID)               7.860
 3. Neto (CHE, MID)                 7.095
 4. Sessegnon (FUL, MID)            6.750
 5. Garnacho (CHE, MID)             6.200
 6. Rice (ARS, MID)                 6.190
 7. Amad (MUN, MID)                 5.810
 8. M.Salah (LIV, MID)              5.630
 9. Gomez (BHA, MID)                5.520
10. Szoboszlai (LIV, MID)           5.510
```
---

## Top 10 Forwards
```text
1. Casemiro (MUN, MID)            10.440
 2. Mbeumo (MUN, MID)               7.860
 3. Neto (CHE, MID)                 7.095
 4. Sessegnon (FUL, MID)            6.750
 5. Garnacho (CHE, MID)             6.200
 6. Rice (ARS, MID)                 6.190
 7. Amad (MUN, MID)                 5.810
 8. M.Salah (LIV, MID)              5.630
 9. Gomez (BHA, MID)                5.520
10. Szoboszlai (LIV, MID)           5.510
```
