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
**Upcoming GW14 (model forecast)**
### Top 20 Overall 
```text
 1. Gomez (BHA, MID)               10.295
 2. Sessegnon (FUL, MID)           10.260
 3. Haaland (MCI, FWD)              9.780
 4. Rice (ARS, MID)                 9.690
 5. Guéhi (CRY, DEF)                9.630
 6. Casemiro (MUN, MID)             8.940
 7. Wieffer (BHA, MID)              8.880
 8. Gravenberch (LIV, MID)          8.780
 9. Amad (MUN, MID)                 8.230
10. Xhaka (SUN, MID)                7.925
11. João Pedro (CHE, FWD)           7.745
12. Cherki (MCI, MID)               7.720
13. Gabriel (ARS, DEF)              7.330
14. J.Murphy (NEW, MID)             7.275
15. Ndiaye (EVE, MID)               7.215
16. Wilson (FUL, MID)               7.175
17. Gibbs-White (NFO, MID)          7.170
18. Lacroix (CRY, DEF)              6.895
19. Cucurella (CHE, DEF)            6.890
20. Leno (FUL, GK)                  6.715
```
---

### Top 10 GoalKeepers
```text
 1. Leno (FUL, GK)                  6.715
 2. Mamardashvili (LIV, GK)         5.775
 3. Henderson (CRY, GK)             4.885
 4. Raya (ARS, GK)                  4.515
 5. Sánchez (CHE, GK)               4.320
 6. Verbruggen (BHA, GK)            4.050
 7. Vicario (TOT, GK)               3.775
 8. Martinez (AVL, GK)              3.045
 9. Pope (NEW, GK)                  2.375
10. Pickford (EVE, GK)              2.330
```
---

### Top 10 Defenders
``` text
 1. Guéhi (CRY, DEF)                9.630
 2. Gabriel (ARS, DEF)              7.330
 3. Lacroix (CRY, DEF)              6.895
 4. Cucurella (CHE, DEF)            6.890
 5. Muñoz (CRY, DEF)                6.545
 6. Saliba (ARS, DEF)               6.115
 7. Calafiori (ARS, DEF)            6.015
 8. Chalobah (CHE, DEF)             5.780
 9. Van Hecke (BHA, DEF)            5.745
10. Konaté (LIV, DEF)               5.670
```
---

## Top 10 Midfielders
```text
 1. Gomez (BHA, MID)               10.295
 2. Sessegnon (FUL, MID)           10.260
 3. Rice (ARS, MID)                 9.690
 4. Casemiro (MUN, MID)             8.940
 5. Wieffer (BHA, MID)              8.880
 6. Gravenberch (LIV, MID)          8.780
 7. Amad (MUN, MID)                 8.230
 8. Xhaka (SUN, MID)                7.925
 9. Cherki (MCI, MID)               7.720
10. J.Murphy (NEW, MID)             7.275
```
---

## Top 10 Forwards
```text
 1. Haaland (MCI, FWD)              9.780
 2. João Pedro (CHE, FWD)           7.745
 3. Mateta (CRY, FWD)               5.805
 4. Welbeck (BHA, FWD)              5.700
 5. Raúl (FUL, FWD)                 5.125
 6. Gyökeres (ARS, FWD)             4.645
 7. Thiago (BRE, FWD)               3.660
 8. Igor Jesus (NFO, FWD)           3.245
 9. Wilson (WHU, FWD)               2.170
10. Ekitiké (LIV, FWD)              1.975
```
