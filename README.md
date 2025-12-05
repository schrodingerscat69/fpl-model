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
**Upcoming GW15 (model forecast)**
### Top 20 Overall 
```text
 1. Gusto (CHE, DEF)              11.360
 2. Martinez (AVL, GK)            10.170
 3. Keane (EVE, DEF)               9.750
 4. Souček (WHU, MID)              9.395
 5. O’Reilly (MCI, DEF)            9.300
 6. Matheus N. (MCI, DEF)          9.190
 7. Gibbs-White (NFO, MID)         8.720
 8. Thiago (BRE, FWD)              8.165
 9. Trossard (ARS, MID)            8.080
10. Buendía (AVL, MID)             7.815
11. De Ligt (MUN, DEF)             7.775
12. Saka (ARS, MID)                7.670
13. Cullen (BUR, MID)              7.530
14. Dewsbury-Hall (EVE, MID)       7.400
15. Mitchell (CRY, DEF)            7.330
16. Ballard (SUN, DEF)             7.270
17. Amad (MUN, MID)                7.255
18. Barnes (NEW, MID)              7.230
19. Digne (AVL, DEF)               7.030
20. Hutchinson (NFO, MID)          6.865
```
---

### Top 10 GoalKeepers
```text
 1. Martinez (AVL, GK)            10.170
 2. Henderson (CRY, GK)            5.895
 3. Mamardashvili (LIV, GK)        5.655
 4. Pickford (EVE, GK)             5.210
 5. Verbruggen (BHA, GK)           4.420
 6. Donnarumma (MCI, GK)           4.285
 7. Sánchez (CHE, GK)              4.170
 8. Pope (NEW, GK)                 3.025
 9. Sels (NFO, GK)                 1.920
10. Dúbravka (BUR, GK)             1.895
```
---

### Top 10 Defenders
``` text
 1. Gusto (CHE, DEF)              11.360
 2. Keane (EVE, DEF)               9.750
 3. O’Reilly (MCI, DEF)            9.300
 4. Matheus N. (MCI, DEF)          9.190
 5. De Ligt (MUN, DEF)             7.775
 6. Mitchell (CRY, DEF)            7.330
 7. Ballard (SUN, DEF)             7.270
 8. Digne (AVL, DEF)               7.030
 9. Lacroix (CRY, DEF)             5.970
10. Pau (AVL, DEF)                 5.595
```
---

## Top 10 Midfielders
```text
 1. Souček (WHU, MID)              9.395
 2. Gibbs-White (NFO, MID)         8.720
 3. Trossard (ARS, MID)            8.080
 4. Buendía (AVL, MID)             7.815
 5. Saka (ARS, MID)                7.670
 6. Cullen (BUR, MID)              7.530
 7. Dewsbury-Hall (EVE, MID)       7.400
 8. Amad (MUN, MID)                7.255
 9. Barnes (NEW, MID)              7.230
10. Hutchinson (NFO, MID)          6.865
```
---

## Top 10 Forwards
```text
 1. Thiago (BRE, FWD)              8.165
 2. João Pedro (CHE, FWD)          5.450
 3. Richarlison (TOT, FWD)         5.430
 4. Nmecha (LEE, FWD)              4.990
 5. Wilson (WHU, FWD)              4.470
 6. Woltemade (NEW, FWD)           4.225
 7. Haaland (MCI, FWD)             3.805
 8. Flemming (BUR, FWD)            3.660
 9. Brobbey (SUN, FWD)             3.165
10. Isidor (SUN, FWD)              1.970
```
