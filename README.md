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
**Upcoming GW18 (model forecast)**
### Top 20 Overall 
```text
 1. Foden (MCI, MID)              11.435
 2. Haaland (MCI, FWD)            10.985
 3. Wilson (FUL, MID)              9.450
 4. Romero (TOT, DEF)              9.205
 5. Guéhi (CRY, DEF)               8.830
 6. Calvert-Lewin (LEE, FWD)       8.615
 7. Chukwueze (FUL, MID)           8.510
 8. E.Le Fée (SUN, MID)            8.190
 9. Tarkowski (EVE, DEF)           8.080
10. White (ARS, DEF)               7.885
11. Van Hecke (BHA, DEF)           7.830
12. Grealish (EVE, MID)            7.775
13. Kudus (TOT, MID)               7.775
14. O.Dango (BRE, MID)             7.525
15. Mykolenko (EVE, DEF)           7.490
16. Merino (ARS, MID)              7.480
17. Watkins (AVL, FWD)             7.165
18. Thiago (BRE, FWD)              7.140
19. Pickford (EVE, GK)             7.110
20. Kamara (AVL, MID)              6.995
```
---

### Top 10 GoalKeepers
```text
 1. Pickford (EVE, GK)             7.110
 2. Henderson (CRY, GK)            5.145
 3. Raya (ARS, GK)                 3.730
 4. Sels (NFO, GK)                 3.590
 5. A.Becker (LIV, GK)             3.530
 6. Martinez (AVL, GK)             3.450
 7. Johnstone (WOL, GK)            3.345
 8. Verbruggen (BHA, GK)           2.430
 9. Areola (WHU, GK)               2.185
10. Perri (LEE, GK)                2.140
```
---

### Top 10 Defenders
``` text
 1. Romero (TOT, DEF)              9.205
 2. Guéhi (CRY, DEF)               8.830
 3. Tarkowski (EVE, DEF)           8.080
 4. White (ARS, DEF)               7.885
 5. Van Hecke (BHA, DEF)           7.830
 6. Mykolenko (EVE, DEF)           7.490
 7. Gomez (LIV, DEF)               6.885
 8. Thiaw (NEW, DEF)               6.875
 9. Dalot (MUN, DEF)               6.770
10. Konaté (LIV, DEF)              6.745
```
---

## Top 10 Midfielders
```text
 1. Foden (MCI, MID)              11.435
 2. Wilson (FUL, MID)              9.450
 3. Chukwueze (FUL, MID)           8.510
 4. E.Le Fée (SUN, MID)            8.190
 5. Grealish (EVE, MID)            7.775
 6. Kudus (TOT, MID)               7.775
 7. O.Dango (BRE, MID)             7.525
 8. Merino (ARS, MID)              7.480
 9. Kamara (AVL, MID)              6.995
10. L.Miley (NEW, MID)             6.990
```
---

## Top 10 Forwards
```text
 1. Haaland (MCI, FWD)            10.985
 2. Calvert-Lewin (LEE, FWD)       8.615
 3. Watkins (AVL, FWD)             7.165
 4. Thiago (BRE, FWD)              7.140
 5. Igor Jesus (NFO, FWD)          5.705
 6. Woltemade (NEW, FWD)           5.420
 7. Zirkzee (MUN, FWD)             5.220
 8. Ekitiké (LIV, FWD)             5.095
 9. Tzimas (BHA, FWD)              4.120
10. Isak (LIV, FWD)                3.285
```
