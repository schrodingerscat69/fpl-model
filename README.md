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
**Upcoming GW31 (model forecast)**
### Top 20 Overall 
```text
 1. O'Reilly (MCI, DEF)           11.175
 2. Henderson (CRY, GK)           10.220
 3. Virgil (LIV, DEF)              9.730
 4. Iwobi (FUL, MID)               9.465
 5. Raúl (FUL, FWD)                8.915
 6. Stach (LEE, MID)               8.510
 7. Lammens (MUN, GK)              8.240
 8. Welbeck (BHA, FWD)             8.235
 9. Hermansen (WHU, GK)            8.010
10. Mac Allister (LIV, MID)        7.955
11. E.Le Fée (SUN, MID)            7.730
12. Diouf (WHU, DEF)               7.570
13. Mitchell (CRY, DEF)            7.285
14. Šeško (MUN, FWD)               7.240
15. Wilson (FUL, MID)              6.965
16. João Pedro (CHE, FWD)          6.940
17. Gomez (BHA, MID)               6.840
18. Dalot (MUN, DEF)               6.635
19. Ampadu (LEE, MID)              6.595
20. Petrović (BOU, GK)             6.500
```
---

### Top 10 GoalKeepers
```text
 1. Henderson (CRY, GK)           10.220
 2. Lammens (MUN, GK)              8.240
 3. Hermansen (WHU, GK)            8.010
 4. Petrović (BOU, GK)             6.500
 5. Verbruggen (BHA, GK)           5.085
 6. A.Becker (LIV, GK)             4.845
 7. Donnarumma (MCI, GK)           4.195
 8. Martinez (AVL, GK)             2.865
 9. Ortega Moreno (NFO, GK)        2.650
10. Pope (NEW, GK)                 1.940
```
---

### Top 10 Defenders
``` text
 1. O'Reilly (MCI, DEF)           11.175
 2. Virgil (LIV, DEF)              9.730
 3. Diouf (WHU, DEF)               7.570
 4. Mitchell (CRY, DEF)            7.285
 5. Dalot (MUN, DEF)               6.635
 6. Muñoz (CRY, DEF)               6.430
 7. Maguire (MUN, DEF)             6.260
 8. F.Kadıoğlu (BHA, DEF)          5.745
 9. Van Hecke (BHA, DEF)           5.560
10. Wan-Bissaka (WHU, DEF)         5.525
```
---

## Top 10 Midfielders
```text
 1. Iwobi (FUL, MID)               9.465
 2. Stach (LEE, MID)               8.510
 3. Mac Allister (LIV, MID)        7.955
 4. E.Le Fée (SUN, MID)            7.730
 5. Wilson (FUL, MID)              6.965
 6. Gomez (BHA, MID)               6.840
 7. Ampadu (LEE, MID)              6.595
 8. B.Fernandes (MUN, MID)         6.045
 9. J.Ramsey (NEW, MID)            5.360
10. Neto (CHE, MID)                5.340
```
---

## Top 10 Forwards
```text
 1. Raúl (FUL, FWD)                8.915
 2. Welbeck (BHA, FWD)             8.235
 3. Šeško (MUN, FWD)               7.240
 4. João Pedro (CHE, FWD)          6.940
 5. Flemming (BUR, FWD)            5.765
 6. Abraham (AVL, FWD)             5.520
 7. Kolo Muani (TOT, FWD)          4.915
 8. Haaland (MCI, FWD)             4.780
 9. Bowen (WHU, FWD)               4.380
10. Ekitiké (LIV, FWD)             2.505
```
