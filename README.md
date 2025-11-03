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
5. This is not a complete and fine-tuned model yet. For example: I have not included latest data on injuries so the model doesn't know if, for eg: Martinelli is injured right now (as of Nov 2, 2025)
---

## 🗓 Latest Predictions
**Upcoming GW11 (model forecast)**
### Top 20 Overall 
```text
 1. Malen (AVL, MID)                11.895
 2. Gabriel (ARS, DEF)              10.390
 3. Gvardiol (MCI, DEF)              9.385
 4. Burn (NEW, DEF)                  9.270
 5. Bruno G. (NEW, MID)              8.875
 6. Rúben (MCI, DEF)                 8.145
 7. Grealish (EVE, MID)              7.845
 8. Thiaw (NEW, DEF)                 7.600
 9. Neto (CHE, MID)                  7.330
10. Mbeumo (MUN, MID)                6.730
11. Van Hecke (BHA, DEF)             6.695
12. Sarr (CRY, MID)                  6.610
13. H.Bueno (WOL, DEF)               6.595
14. Van de Ven (TOT, DEF)            6.565
15. James (CHE, DEF)                 6.470
16. Trossard (ARS, MID)              6.110
17. Trippier (NEW, DEF)              6.110
18. Pope (NEW, GK)                   6.025
19. Saka (ARS, MID)                  5.785
20. Talbi (SUN, MID)                 5.715
```
---

### Top 10 GoalKeepers
```text
 1. Pope (NEW, GK)                   6.025
 2. Raya (ARS, GK)                   5.675
 3. Roefs (SUN, GK)                  5.440
 4. Donnarumma (MCI, GK)             4.530
 5. Petrović (BOU, GK)               3.125
 6. Leno (FUL, GK)                   2.665
 7. Sels (NFO, GK)                   2.635
 8. Kelleher (BRE, GK)               2.475
 9. Vicario (TOT, GK)                2.455
10. Sánchez (CHE, GK)                2.365
```
---

### Top 10 Defenders
``` text
 1. Gabriel (ARS, DEF)              10.390
 2. Gvardiol (MCI, DEF)              9.385
 3. Burn (NEW, DEF)                  9.270
 4. Rúben (MCI, DEF)                 8.145
 5. Thiaw (NEW, DEF)                 7.600
 6. Van Hecke (BHA, DEF)             6.695
 7. H.Bueno (WOL, DEF)               6.595
 8. Van de Ven (TOT, DEF)            6.565
 9. James (CHE, DEF)                 6.470
10. Trippier (NEW, DEF)              6.110
```
---

## Top 10 Midfielders
```text
 1. Malen (AVL, MID)                11.895
 2. Bruno G. (NEW, MID)              8.875
 3. Grealish (EVE, MID)              7.845
 4. Neto (CHE, MID)                  7.330
 5. Mbeumo (MUN, MID)                6.730
 6. Sarr (CRY, MID)                  6.610
 7. Trossard (ARS, MID)              6.110
 8. Saka (ARS, MID)                  5.785
 9. Talbi (SUN, MID)                 5.715
10. Rice (ARS, MID)                  5.495
```
---

## Top 10 Forwards
```text
 1. Woltemade (NEW, FWD)             5.120
 2. Haaland (MCI, FWD)               4.695
 3. Kroupi.Jr (BOU, FWD)             3.990
 4. João Pedro (CHE, FWD)            3.810
 5. Mateta (CRY, FWD)                3.675
 6. Welbeck (BHA, FWD)               3.565
 7. Tzimas (BHA, FWD)                3.290
 8. Isidor (SUN, FWD)                2.880
 9. Raúl (FUL, FWD)                  2.690
10. Gyökeres (ARS, FWD)              2.435
```
