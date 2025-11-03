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
 1. Malen (AVL, MID)        12.16
 2. Semenyo (BOU, MID)      11.66
 3. Gabriel (ARS, DEF)       9.92
 4. Burn (NEW, DEF)          9.13
 5. Bruno G. (NEW, MID)      9.10
 6. Kudus (TOT, MID)         8.92
 7. Gvardiol (MCI, DEF)      8.59
 8. Mount (MUN, MID)         8.49
 9. Rúben (MCI, DEF)         8.26
10. Estêvão (CHE, MID)       8.21
11. J. Timber (ARS, DEF)     7.76
12. Thiaw (NEW, DEF)         7.62
13. Rice (ARS, MID)          7.37
14. Ndiaye (EVE, MID)        7.13
15. Saka (ARS, MID)          7.00
16. Šeško (MUN, FWD)         6.95
17. Grealish (EVE, MID)      6.92
18. Sessegnon (FUL, MID)     6.89
19. Shaw (MUN, DEF)          6.77
20. Sarr (CRY, MID)          6.73
```
---

### Top 10 GoalKeepers
```text
 1. Pope (NEW)        6.16
 2. Lammens (MUN)     6.13
 3. Raya (ARS)        5.60
 4. Donnarumma (MCI)  5.58
 5. Sels (NFO)        2.66
 6. Vicario (TOT)     2.56
 7. Pickford (EVE)    2.45
 8. Kelleher (BRE)    2.27
 9. Petrović (BOU)    2.18
10. Areola (WHU)      1.97
```
---

### Top 10 Defenders
``` text
 1. Gabriel (ARS)     9.92
 2. Burn (NEW)        9.13
 3. Gvardiol (MCI)    8.59
 4. Rúben (MCI)       8.26
 5. J. Timber (ARS)   7.76
 6. Thiaw (NEW)       7.62
 7. Shaw (MUN)        6.77
 8. H. Bueno (WOL)    6.50
 9. Van Hecke (BHA)   6.37
10. Dalot (MUN)       6.33
```
---

## Top 10 Midfielders
```text
 1. Malen (AVL)      12.16
 2. Semenyo (BOU)    11.66
 3. Bruno G. (NEW)    9.10
 4. Kudus (TOT)       8.92
 5. Mount (MUN)       8.49
 6. Estêvão (CHE)     8.21
 7. Rice (ARS)        7.37
 8. Ndiaye (EVE)      7.13
 9. Saka (ARS)        7.00
10. Grealish (EVE)    6.92
```
---

## Top 10 Forwards
```text
 1. Šeško (MUN)        6.95
 2. Woltemade (NEW)    5.34
 3. Haaland (MCI)      5.11
 4. Isak (LIV)         4.45
 5. Welbeck (BHA)      3.78
 6. Mateta (CRY)       3.73
 7. Tzimas (BHA)       3.36
 8. Gyökeres (ARS)     2.40
 9. Thiago (BRE)       1.96
10. João Pedro (CHE)   1.80
```
