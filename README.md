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

---

## 🗓 Latest Predictions

**Upcoming GW10 (model forecast)**

### Top 20 Overall
```text
1. Gabriel (ARS, DEF)        11.72
2. Alderete (SUN, DEF)       10.42
3. Haaland (MCI, FWD)         9.91
4. Sarr (CRY, MID)            9.47
5. Thiago (BRE, FWD)          8.89
6. Buendía (AVL, MID)         8.56
7. Keane (EVE, DEF)           8.37
8. S. Bueno (WOL, DEF)        8.25
9. Xhaka (SUN, MID)           8.18
10. Longstaff (LEE, MID)      7.99
11. Doku (MCI, MID)           7.92
12. Enzo (CHE, MID)           7.71
13. Matheus N. (MCI, DEF)     7.68
14. Kelleher (BRE, GK)        7.63
15. Watkins (AVL, FWD)        7.56
16. Roefs (SUN, GK)           7.54
17. Mukiele (SUN, DEF)        7.38
18. Welbeck (BHA, FWD)        6.82
19. De Cuyper (BHA, DEF)      6.82
20. Guéhi (CRY, DEF)          6.78

### Top 10 GoalKeepers
1. Kelleher (BRE)    7.64
2. Roefs (SUN)       7.54
3. Dúbravka (BUR)    2.57
4. Areola (WHU)      2.46
5. Martinez (AVL)    2.36
6. Petrović (BOU)    2.20
7. Raya (ARS)        2.07
8. Verbruggen (BHA)  1.98
9. Pope (NEW)        1.93
10. Vicario (TOT)    1.90

### Top 10 Defenders
1. Gabriel (ARS)        11.72
2. Alderete (SUN)       10.42
3. Keane (EVE)           8.37
4. S. Bueno (WOL)        8.25
5. Matheus N. (MCI)      7.68
6. Mukiele (SUN)         7.38
7. De Cuyper (BHA)       6.82
8. Guéhi (CRY)           6.78
9. James (CHE)           5.74
10. J. Timber (ARS)      5.20

## Top 10 Midfielders
1. Sarr (CRY)           9.47
2. Buendía (AVL)        8.56
3. Xhaka (SUN)          8.18
4. Longstaff (LEE)      7.99
5. Doku (MCI)           7.92
6. Enzo (CHE)           7.71
7. Semenyo (BOU)        6.68
8. Gravenberch (LIV)    6.67
9. Rice (ARS)           6.21
10. Yarmoliuk (BRE)     5.95


## Top 10 Forwards
1. Haaland (MCI)        9.91
2. Thiago (BRE)         8.89
3. Watkins (AVL)        7.56
4. Welbeck (BHA)        6.82
5. Bowen (WHU)          6.35
6. Woltemade (NEW)      5.88
7. Raúl (FUL)           5.06
8. Šeško (MUN)          4.34
9. Kroupi Jr (BOU)      4.23
10. Nketiah (CRY)       3.51

