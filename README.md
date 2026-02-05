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
**Upcoming GW25 (model forecast)**
### Top 20 Overall 
```text
 1. Barnes (NEW, MID)             11.960
 2. Aaronson (LEE, MID)           11.845
 3. Bruno G. (NEW, MID)            9.120
 4. Collins (BRE, DEF)             8.790
 5. Kelleher (BRE, GK)             8.465
 6. Tel (TOT, MID)                 8.320
 7. Šeško (MUN, FWD)               8.105
 8. Mané (WOL, FWD)                8.050
 9. Thiago (BRE, FWD)              7.870
10. Berge (FUL, MID)               7.550
11. J.Palhinha (TOT, MID)          7.490
12. Calvert-Lewin (LEE, FWD)       7.440
13. A.Becker (LIV, GK)             7.195
14. Wilson (FUL, MID)              6.775
15. João Pedro (CHE, FWD)          6.595
16. Kerkez (LIV, DEF)              6.535
17. Lacroix (CRY, DEF)             6.400
18. Gibbs-White (NFO, MID)         6.175
19. Mitoma (BHA, MID)              6.030
20. Gabriel (ARS, DEF)             6.030
```
---

### Top 10 GoalKeepers
```text
 1. Kelleher (BRE, GK)             8.465
 2. A.Becker (LIV, GK)             7.195
 3. Raya (ARS, GK)                 5.165
 4. Henderson (CRY, GK)            5.010
 5. Sels (NFO, GK)                 3.445
 6. Dúbravka (BUR, GK)             3.350
 7. Pickford (EVE, GK)             2.810
 8. Sánchez (CHE, GK)              2.610
 9. Leno (FUL, GK)                 2.575
10. M.Bizot (AVL, GK)              2.080
```
---

### Top 10 Defenders
``` text
 1. Collins (BRE, DEF)             8.790
 2. Kerkez (LIV, DEF)              6.535
 3. Lacroix (CRY, DEF)             6.400
 4. Gabriel (ARS, DEF)             6.030
 5. Saliba (ARS, DEF)              5.940
 6. Lindelöf (AVL, DEF)            5.620
 7. Canvot (CRY, DEF)              5.535
 8. Digne (AVL, DEF)               5.485
 9. Konsa (AVL, DEF)               5.465
10. Henry (BRE, DEF)               4.945
```
---

## Top 10 Midfielders
```text
 1. Barnes (NEW, MID)             11.960
 2. Aaronson (LEE, MID)           11.845
 3. Bruno G. (NEW, MID)            9.120
 4. Tel (TOT, MID)                 8.320
 5. Berge (FUL, MID)               7.550
 6. J.Palhinha (TOT, MID)          7.490
 7. Wilson (FUL, MID)              6.775
 8. Gibbs-White (NFO, MID)         6.175
 9. Mitoma (BHA, MID)              6.030
10. Semenyo (MCI, MID)             5.980
```
---

## Top 10 Forwards
```text
 1. Šeško (MUN, FWD)               8.105
 2. Mané (WOL, FWD)                8.050
 3. Thiago (BRE, FWD)              7.870
 4. Calvert-Lewin (LEE, FWD)       7.440
 5. João Pedro (CHE, FWD)          6.595
 6. Evanilson (BOU, FWD)           5.575
 7. Delap (CHE, FWD)               5.065
 8. Kroupi.Jr (BOU, FWD)           4.745
 9. Raúl (FUL, FWD)                4.725
10. Haaland (MCI, FWD)             4.515
```
