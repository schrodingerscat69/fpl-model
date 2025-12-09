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
**Upcoming GW16 (model forecast)**
### Top 20 Overall 
```text
 1. Eze (ARS, MID)                 10.480
 2. Barnes (NEW, MID)              10.480
 3. Mitchell (CRY, DEF)             9.925
 4. Rogers (AVL, MID)               9.605
 5. Gibbs-White (NFO, MID)          8.980
 6. Nmecha (LEE, FWD)               8.945
 7. Wilson (WHU, FWD)               8.935
 8. N.Williams (NFO, DEF)           8.930
 9. Lacroix (CRY, DEF)              8.845
10. Anderson (NFO, MID)             8.825
11. Muñoz (CRY, DEF)                8.730
12. Andersen (FUL, DEF)             8.445
13. Martinez (AVL, GK)              8.245
14. Henderson (CRY, GK)             8.210
15. Matheus N. (MCI, DEF)           7.905
16. Enzo (CHE, MID)                 7.825
17. Milenković (NFO, DEF)           7.820
18. Trossard (ARS, MID)             7.725
19. Verbruggen (BHA, GK)            7.705
20. Mykolenko (EVE, DEF)            7.670
```
---

### Top 10 GoalKeepers
```text
 1. Martinez (AVL, GK)              8.245
 2. Henderson (CRY, GK)             8.210
 3. Verbruggen (BHA, GK)            7.705
 4. Areola (WHU, GK)                7.230
 5. Pickford (EVE, GK)              6.735
 6. Sels (NFO, GK)                  4.480
 7. Pope (NEW, GK)                  3.245
 8. Leno (FUL, GK)                  3.220
 9. Sánchez (CHE, GK)               2.950
10. Donnarumma (MCI, GK)            2.935
```
---

### Top 10 Defenders
``` text
 1. Mitchell (CRY, DEF)             9.925
 2. N.Williams (NFO, DEF)           8.930
 3. Lacroix (CRY, DEF)              8.845
 4. Muñoz (CRY, DEF)                8.730
 5. Andersen (FUL, DEF)             8.445
 6. Matheus N. (MCI, DEF)           7.905
 7. Milenković (NFO, DEF)           7.820
 8. Mykolenko (EVE, DEF)            7.670
 9. Savona (NFO, DEF)               7.640
10. Bassey (FUL, DEF)               6.895
```
---

## Top 10 Midfielders
```text
 1. Eze (ARS, MID)                 10.480
 2. Barnes (NEW, MID)              10.480
 3. Rogers (AVL, MID)               9.605
 4. Gibbs-White (NFO, MID)          8.980
 5. Anderson (NFO, MID)             8.825
 6. Enzo (CHE, MID)                 7.825
 7. Trossard (ARS, MID)             7.725
 8. Doku (MCI, MID)                 7.440
 9. Barkley (AVL, MID)              7.310
10. Minteh (BHA, MID)               7.030
```
---

## Top 10 Forwards
```text
 1. Nmecha (LEE, FWD)               8.945
 2. Wilson (WHU, FWD)               8.935
 3. Thiago (BRE, FWD)               6.910
 4. Welbeck (BHA, FWD)              4.830
 5. Richarlison (TOT, FWD)          4.080
 6. Raúl (FUL, FWD)                 3.760
 7. Marc Guiu (CHE, FWD)            3.430
 8. Bowen (WHU, FWD)                3.010
 9. Enes Ünal (BOU, FWD)            2.775
10. Evanilson (BOU, FWD)            2.660
```
