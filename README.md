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
**Upcoming GW24 (model forecast)**
### Top 20 Overall 
```text
 1. Gabriel (ARS, DEF)            11.400
 2. McGinn (AVL, MID)             10.915
 3. Janelt (BRE, MID)             10.265
 4. Thiaw (NEW, DEF)              10.185
 5. Garner (EVE, MID)              9.365
 6. H.Bueno (WOL, DEF)             9.350
 7. Enzo (CHE, MID)                8.185
 8. Rice (ARS, MID)                8.095
 9. Bruno G. (NEW, MID)            7.910
10. Hee Chan (WOL, MID)            7.885
11. Struijk (LEE, DEF)             7.750
12. Collins (BRE, DEF)             7.745
13. Thiago (BRE, FWD)              7.525
14. L.Miley (NEW, MID)             7.525
15. F.Kadıoğlu (BHA, DEF)          7.355
16. Van de Ven (TOT, DEF)          7.140
17. Veltman (BHA, DEF)             6.965
18. Pedro Porro (TOT, DEF)         6.935
19. Mané (WOL, FWD)                6.915
20. Bradley (LIV, DEF)             6.825
```
---

### Top 10 GoalKeepers
```text
 1. José Sá (WOL, GK)              5.410
 2. Perri (LEE, GK)                5.400
 3. Roefs (SUN, GK)                5.120
 4. A.Becker (LIV, GK)             3.970
 5. Pickford (EVE, GK)             3.940
 6. Verbruggen (BHA, GK)           3.690
 7. Vicario (TOT, GK)              3.675
 8. Kelleher (BRE, GK)             3.560
 9. Donnarumma (MCI, GK)           3.110
10. Pope (NEW, GK)                 2.680
```
---

### Top 10 Defenders
``` text
 1. Gabriel (ARS, DEF)            11.400
 2. Thiaw (NEW, DEF)              10.185
 3. H.Bueno (WOL, DEF)             9.350
 4. Struijk (LEE, DEF)             7.750
 5. Collins (BRE, DEF)             7.745
 6. F.Kadıoğlu (BHA, DEF)          7.355
 7. Van de Ven (TOT, DEF)          7.140
 8. Veltman (BHA, DEF)             6.965
 9. Pedro Porro (TOT, DEF)         6.935
10. Bradley (LIV, DEF)             6.825
```
---

## Top 10 Midfielders
```text
 1. McGinn (AVL, MID)             10.915
 2. Janelt (BRE, MID)             10.265
 3. Garner (EVE, MID)              9.365
 4. Enzo (CHE, MID)                8.185
 5. Rice (ARS, MID)                8.095
 6. Bruno G. (NEW, MID)            7.910
 7. Hee Chan (WOL, MID)            7.885
 8. L.Miley (NEW, MID)             7.525
 9. Wirtz (LIV, MID)               6.410
10. Casemiro (MUN, MID)            6.285
```
---

## Top 10 Forwards
```text
 1. Thiago (BRE, FWD)              7.525
 2. Mané (WOL, FWD)                6.915
 3. Bowen (WHU, FWD)               5.615
 4. Watkins (AVL, FWD)             5.310
 5. Mateta (CRY, FWD)              5.135
 6. Brobbey (SUN, FWD)             5.105
 7. Evanilson (BOU, FWD)           4.840
 8. Zirkzee (MUN, FWD)             4.435
 9. Broja (BUR, FWD)               4.215
10. Beto (EVE, FWD)                4.130
```
