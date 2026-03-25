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
**Upcoming GW32 (model forecast)**
### Top 20 Overall 
```text
 1. Saliba (ARS, DEF)             13.870
 2. Gabriel (ARS, DEF)            12.530
 3. J.Timber (ARS, DEF)           12.430
 4. Damsgaard (BRE, MID)          12.345
 5. J.Gomes (WOL, MID)            11.615
 6. Aït-Nouri (MCI, DEF)          10.405
 7. José Sá (WOL, GK)              9.860
 8. Semenyo (MCI, MID)             9.775
 9. Mosquera (WOL, DEF)            9.755
10. Ekitiké (LIV, FWD)             9.375
11. Mac Allister (LIV, MID)        8.700
12. Hannibal (BUR, MID)            8.530
13. R.Gomes (WOL, DEF)             8.465
14. Rice (ARS, MID)                8.405
15. Wilson (FUL, MID)              8.330
16. Rodrigo (MCI, MID)             8.220
17. Šeško (MUN, FWD)               8.120
18. B.Fernandes (MUN, MID)         8.120
19. Iwobi (FUL, MID)               7.920
20. Matheus N. (MCI, DEF)          7.745
```
---

### Top 10 GoalKeepers
```text
 1. José Sá (WOL, GK)              9.860
 2. Raya (ARS, GK)                 6.780
 3. Lammens (MUN, GK)              5.445
 4. Donnarumma (MCI, GK)           3.910
 5. Henderson (CRY, GK)            3.235
 6. Darlow (LEE, GK)               3.130
 7. Verbruggen (BHA, GK)           2.640
 8. Leno (FUL, GK)                 2.430
 9. Petrović (BOU, GK)             2.255
10. Sánchez (CHE, GK)              1.885
```
---

### Top 10 Defenders
``` text
 1. Saliba (ARS, DEF)             13.870
 2. Gabriel (ARS, DEF)            12.530
 3. J.Timber (ARS, DEF)           12.430
 4. Aït-Nouri (MCI, DEF)          10.405
 5. Mosquera (WOL, DEF)            9.755
 6. R.Gomes (WOL, DEF)             8.465
 7. Matheus N. (MCI, DEF)          7.745
 8. Rúben (MCI, DEF)               7.550
 9. Canvot (CRY, DEF)              6.865
10. Tchatchoua (WOL, DEF)          6.765
```
---

## Top 10 Midfielders
```text
 1. Damsgaard (BRE, MID)          12.345
 2. J.Gomes (WOL, MID)            11.615
 3. Semenyo (MCI, MID)             9.775
 4. Mac Allister (LIV, MID)        8.700
 5. Hannibal (BUR, MID)            8.530
 6. Rice (ARS, MID)                8.405
 7. Wilson (FUL, MID)              8.330
 8. Rodrigo (MCI, MID)             8.220
 9. B.Fernandes (MUN, MID)         8.120
10. Iwobi (FUL, MID)               7.920
```
---

## Top 10 Forwards
```text
 1. Ekitiké (LIV, FWD)             9.375
 2. Šeško (MUN, FWD)               8.120
 3. Mayenda (SUN, FWD)             7.095
 4. Welbeck (BHA, FWD)             6.460
 5. Bowen (WHU, FWD)               6.375
 6. Thiago (BRE, FWD)              5.880
 7. Armstrong (WOL, FWD)           5.685
 8. Evanilson (BOU, FWD)           4.925
 9. Taty (WHU, FWD)                4.855
10. Gyökeres (ARS, FWD)            4.770
```
