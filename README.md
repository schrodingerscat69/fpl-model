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
**Upcoming GW36 (model forecast)**
### Top 20 Overall 
```text
 1. Guéhi (MCI, DEF)              12.900
 2. Mavropanos (WHU, DEF)         12.775
 3. Wieffer (BHA, MID)             9.825
 4. Scott (BOU, MID)               9.725
 5. O'Reilly (MCI, DEF)            9.260
 6. N.Williams (NFO, DEF)          9.115
 7. Bowen (WHU, FWD)               9.020
 8. Thiago (BRE, FWD)              8.880
 9. Doku (MCI, MID)                8.505
10. Mateta (CRY, FWD)              8.315
11. Virgil (LIV, DEF)              8.185
12. Roefs (SUN, GK)                8.010
13. Taty (WHU, FWD)                8.000
14. Okafor (LEE, MID)              7.825
15. Cherki (MCI, MID)              7.630
16. Boscagli (BHA, DEF)            7.460
17. Alderete (SUN, DEF)            7.185
18. Dewsbury-Hall (EVE, MID)       7.145
19. Ngumoha (LIV, MID)             7.060
20. Casemiro (MUN, MID)            6.845
```
---

### Top 10 GoalKeepers
```text
 1. Roefs (SUN, GK)                8.010
 2. Verbruggen (BHA, GK)           6.790
 3. Hermansen (WHU, GK)            6.055
 4. Mamardashvili (LIV, GK)        5.825
 5. Donnarumma (MCI, GK)           5.670
 6. Darlow (LEE, GK)               4.140
 7. Sels (NFO, GK)                 3.180
 8. Lammens (MUN, GK)              3.150
 9. Henderson (CRY, GK)            3.055
10. Kinsky (TOT, GK)               3.000
```
---

### Top 10 Defenders
``` text
 1. Guéhi (MCI, DEF)              12.900
 2. Mavropanos (WHU, DEF)         12.775
 3. O'Reilly (MCI, DEF)            9.260
 4. N.Williams (NFO, DEF)          9.115
 5. Virgil (LIV, DEF)              8.185
 6. Boscagli (BHA, DEF)            7.460
 7. Alderete (SUN, DEF)            7.185
 8. F.Kadıoğlu (BHA, DEF)          6.585
 9. Reinildo (SUN, DEF)            6.245
10. Khusanov (MCI, DEF)            6.075
```
---

## Top 10 Midfielders
```text
 1. Wieffer (BHA, MID)             9.825
 2. Scott (BOU, MID)               9.725
 3. Doku (MCI, MID)                8.505
 4. Okafor (LEE, MID)              7.825
 5. Cherki (MCI, MID)              7.630
 6. Dewsbury-Hall (EVE, MID)       7.145
 7. Ngumoha (LIV, MID)             7.060
 8. Casemiro (MUN, MID)            6.845
 9. M.Salah (LIV, MID)             6.565
10. Ayari (BHA, MID)               6.115
```
---

## Top 10 Forwards
```text
 1. Bowen (WHU, FWD)               9.020
 2. Thiago (BRE, FWD)              8.880
 3. Mateta (CRY, FWD)              8.315
 4. Taty (WHU, FWD)                8.000
 5. Beto (EVE, FWD)                5.865
 6. Osula (NEW, FWD)               5.500
 7. Gyökeres (ARS, FWD)            5.145
 8. Igor Jesus (NFO, FWD)          4.975
 9. Kroupi.Jr (BOU, FWD)           4.840
10. Pablo (WHU, FWD)               3.955
```
