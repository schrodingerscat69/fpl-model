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
**Upcoming GW28 (model forecast)**
### Top 20 Overall 
```text
 1. Casemiro (MUN, MID)           11.380
 2. Solanke (TOT, FWD)            10.210
 3. Scott (BOU, MID)               9.730
 4. Wirtz (LIV, MID)               9.690
 5. Ekitiké (LIV, FWD)             9.520
 6. Groß (BHA, MID)                9.245
 7. Madueke (ARS, MID)             9.150
 8. Diarra (SUN, MID)              8.890
 9. Konaté (LIV, DEF)              8.605
10. Kelleher (BRE, GK)             8.015
11. O.Dango (BRE, MID)             7.430
12. Gibbs-White (NFO, MID)         7.425
13. Bowen (WHU, FWD)               7.265
14. Semenyo (MCI, MID)             6.915
15. Haaland (MCI, FWD)             6.910
16. Zubimendi (ARS, MID)           6.870
17. Summerville (WHU, MID)         6.685
18. Cunha (MUN, MID)               6.585
19. Wan-Bissaka (WHU, DEF)         6.505
20. Reinildo (SUN, DEF)            6.500
```
---

### Top 10 GoalKeepers
```text
 1. Kelleher (BRE, GK)             8.015
 2. Petrović (BOU, GK)             6.180
 3. Roefs (SUN, GK)                5.380
 4. Raya (ARS, GK)                 3.250
 5. Henderson (CRY, GK)            3.000
 6. Lammens (MUN, GK)              2.875
 7. Verbruggen (BHA, GK)           2.800
 8. Martinez (AVL, GK)             2.755
 9. Donnarumma (MCI, GK)           2.365
10. Pickford (EVE, GK)             2.290
```
---

### Top 10 Defenders
``` text
 1. Konaté (LIV, DEF)              8.605
 2. Wan-Bissaka (WHU, DEF)         6.505
 3. Reinildo (SUN, DEF)            6.500
 4. Mukiele (SUN, DEF)             6.440
 5. Ballard (SUN, DEF)             6.255
 6. Smith (BOU, DEF)               5.900
 7. De Cuyper (BHA, DEF)           5.475
 8. Alderete (SUN, DEF)            5.460
 9. Henry (BRE, DEF)               5.435
10. Hill (BOU, DEF)                5.425
```
---

## Top 10 Midfielders
```text
 1. Casemiro (MUN, MID)           11.380
 2. Scott (BOU, MID)               9.730
 3. Wirtz (LIV, MID)               9.690
 4. Groß (BHA, MID)                9.245
 5. Madueke (ARS, MID)             9.150
 6. Diarra (SUN, MID)              8.890
 7. O.Dango (BRE, MID)             7.430
 8. Gibbs-White (NFO, MID)         7.425
 9. Semenyo (MCI, MID)             6.915
10. Zubimendi (ARS, MID)           6.870
```
---

## Top 10 Forwards
```text
 1. Solanke (TOT, FWD)            10.210
 2. Ekitiké (LIV, FWD)             9.520
 3. Bowen (WHU, FWD)               7.265
 4. Haaland (MCI, FWD)             6.910
 5. Šeško (MUN, FWD)               5.625
 6. Kroupi.Jr (BOU, FWD)           5.200
 7. João Pedro (CHE, FWD)          4.875
 8. Beto (EVE, FWD)                4.460
 9. Delap (CHE, FWD)               3.990
10. Raúl (FUL, FWD)                3.845
```
