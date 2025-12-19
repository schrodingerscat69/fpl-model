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
**Upcoming GW17 (model forecast)**
### Top 20 Overall 
```text
 1. Foden (MCI, MID)              11.425
 2. Barnes (NEW, MID)             10.735
 3. Thiaw (NEW, DEF)               9.920
 4. Gakpo (LIV, MID)               9.160
 5. L.Miley (NEW, MID)             9.000
 6. O.Dango (BRE, MID)             8.705
 7. E.Le Fée (SUN, MID)            8.545
 8. Tete (FUL, DEF)                8.405
 9. Kamara (AVL, MID)              8.280
10. De Cuyper (BHA, DEF)           8.195
11. Lacroix (CRY, DEF)             8.165
12. Anderson (NFO, MID)            8.095
13. Verbruggen (BHA, GK)           8.025
14. Murillo (NFO, DEF)             7.700
15. Wilson (WHU, FWD)              7.545
16. Nmecha (LEE, FWD)              7.400
17. Minteh (BHA, MID)              7.295
18. Thiago (BRE, FWD)              7.050
19. Andersen (FUL, DEF)            6.870
20. Zirkzee (MUN, FWD)             6.660
```
---

### Top 10 GoalKeepers
```text
 1. Verbruggen (BHA, GK)           8.025
 2. Areola (WHU, GK)               6.160
 3. Martinez (AVL, GK)             6.090
 4. Sels (NFO, GK)                 5.500
 5. Pickford (EVE, GK)             5.125
 6. Sánchez (CHE, GK)              3.170
 7. A.Becker (LIV, GK)             3.130
 8. Henderson (CRY, GK)            3.030
 9. Leno (FUL, GK)                 2.990
10. Johnstone (WOL, GK)            2.795
```
---

### Top 10 Defenders
``` text
 1. Thiaw (NEW, DEF)               9.920
 2. Tete (FUL, DEF)                8.405
 3. De Cuyper (BHA, DEF)           8.195
 4. Lacroix (CRY, DEF)             8.165
 5. Murillo (NFO, DEF)             7.700
 6. Andersen (FUL, DEF)            6.870
 7. Konsa (AVL, DEF)               6.200
 8. Pau (AVL, DEF)                 6.190
 9. Savona (NFO, DEF)              6.070
10. Muñoz (CRY, DEF)               6.065
```
---

## Top 10 Midfielders
```text
 1. Foden (MCI, MID)              11.425
 2. Barnes (NEW, MID)             10.735
 3. Gakpo (LIV, MID)               9.160
 4. L.Miley (NEW, MID)             9.000
 5. O.Dango (BRE, MID)             8.705
 6. E.Le Fée (SUN, MID)            8.545
 7. Kamara (AVL, MID)              8.280
 8. Anderson (NFO, MID)            8.095
 9. Minteh (BHA, MID)              7.295
10. Eze (ARS, MID)                 6.635
```
---

## Top 10 Forwards
```text
 1. Wilson (WHU, FWD)              7.545
 2. Nmecha (LEE, FWD)              7.400
 3. Thiago (BRE, FWD)              7.050
 4. Zirkzee (MUN, FWD)             6.660
 5. Welbeck (BHA, FWD)             6.400
 6. Woltemade (NEW, FWD)           5.990
 7. Isak (LIV, FWD)                5.230
 8. Brobbey (SUN, FWD)             4.955
 9. Evanilson (BOU, FWD)           4.590
10. Flemming (BUR, FWD)            4.200
```
