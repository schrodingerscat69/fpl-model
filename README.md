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
**Upcoming GW29 (model forecast)**
### Top 20 Overall 
```text
 1. O.Dango (BRE, MID)            11.930
 2. Palmer (CHE, MID)              9.050
 3. Zubimendi (ARS, MID)           8.805
 4. Summerville (WHU, MID)         8.610
 5. Haaland (MCI, FWD)             8.375
 6. B.Fernandes (MUN, MID)         7.980
 7. Sarr (CRY, MID)                7.970
 8. Hill (BOU, DEF)                7.610
 9. Groß (BHA, MID)                7.555
10. Rayan (BOU, MID)               7.425
11. Lacroix (CRY, DEF)             7.330
12. Diarra (SUN, MID)              7.300
13. Casemiro (MUN, MID)            7.250
14. Solanke (TOT, FWD)             7.240
15. Scott (BOU, MID)               7.075
16. Mainoo (MUN, MID)              7.030
17. Gruev (LEE, MID)               7.005
18. Szoboszlai (LIV, MID)          6.985
19. Bernardo (MCI, MID)            6.865
20. Dalot (MUN, DEF)               6.835
```
---

### Top 10 GoalKeepers
```text
 1. Hermansen (WHU, GK)            5.520
 2. Kelleher (BRE, GK)             5.210
 3. Petrović (BOU, GK)             4.990
 4. Lammens (MUN, GK)              4.675
 5. Raya (ARS, GK)                 4.490
 6. Henderson (CRY, GK)            3.835
 7. Roefs (SUN, GK)                3.380
 8. Martinez (AVL, GK)             3.360
 9. Verbruggen (BHA, GK)           2.740
10. Pickford (EVE, GK)             2.350
```
---

### Top 10 Defenders
``` text
 1. Hill (BOU, DEF)                7.610
 2. Lacroix (CRY, DEF)             7.330
 3. Dalot (MUN, DEF)               6.835
 4. Virgil (LIV, DEF)              6.230
 5. Wan-Bissaka (WHU, DEF)         6.155
 6. Mavropanos (WHU, DEF)          5.555
 7. O'Reilly (MCI, DEF)            5.530
 8. Martinez (MUN, DEF)            5.305
 9. Ajer (BRE, DEF)                5.285
10. Konaté (LIV, DEF)              5.080
```
---

## Top 10 Midfielders
```text
 1. O.Dango (BRE, MID)            11.930
 2. Palmer (CHE, MID)              9.050
 3. Zubimendi (ARS, MID)           8.805
 4. Summerville (WHU, MID)         8.610
 5. B.Fernandes (MUN, MID)         7.980
 6. Sarr (CRY, MID)                7.970
 7. Groß (BHA, MID)                7.555
 8. Rayan (BOU, MID)               7.425
 9. Diarra (SUN, MID)              7.300
10. Casemiro (MUN, MID)            7.250
```
---

## Top 10 Forwards
```text
 1. Haaland (MCI, FWD)             8.375
 2. Solanke (TOT, FWD)             7.240
 3. Gyökeres (ARS, FWD)            6.375
 4. João Pedro (CHE, FWD)          5.745
 5. Ekitiké (LIV, FWD)             5.355
 6. Calvert-Lewin (LEE, FWD)       5.100
 7. Bowen (WHU, FWD)               4.860
 8. Tolu (WOL, FWD)                4.190
 9. Kroupi.Jr (BOU, FWD)           4.130
10. Mayenda (SUN, FWD)             3.665
```
