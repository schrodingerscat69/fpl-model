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
**Upcoming GW27 (model forecast)**
### Top 20 Overall 
```text
 1. Estêvão (CHE, MID)            11.685
 2. João Pedro (CHE, FWD)         10.635
 3. Buendía (AVL, MID)            10.235
 4. Matheus N. (MCI, DEF)          9.915
 5. Aina (NFO, DEF)                9.375
 6. Szoboszlai (LIV, MID)          9.260
 7. Bowen (WHU, FWD)               9.000
 8. Palmer (CHE, MID)              8.265
 9. Marmoush (MCI, MID)            8.080
10. Wilson (FUL, MID)              7.690
11. Barry (EVE, FWD)               7.530
12. Justin (LEE, DEF)              7.360
13. Semenyo (MCI, MID)             7.300
14. Cunha (MUN, MID)               7.060
15. Igor Jesus (NFO, FWD)          7.050
16. Madueke (ARS, MID)             7.045
17. Van de Ven (TOT, DEF)          6.890
18. Mbeumo (MUN, MID)              6.820
19. M.Fernandes (WHU, MID)         6.735
20. Andersen (FUL, DEF)            6.580
```
---

### Top 10 GoalKeepers
```text
 1. Donnarumma (MCI, GK)           6.570
 2. José Sá (WOL, GK)              6.310
 3. Martinez (AVL, GK)             5.740
 4. Sels (NFO, GK)                 3.215
 5. Dúbravka (BUR, GK)             3.130
 6. Leno (FUL, GK)                 2.415
 7. Raya (ARS, GK)                 2.075
 8. Pickford (EVE, GK)             2.050
 9. Sánchez (CHE, GK)              1.980
10. Darlow (LEE, GK)               1.930
```
---

### Top 10 Defenders
``` text
 1. Matheus N. (MCI, DEF)          9.915
 2. Aina (NFO, DEF)                9.375
 3. Justin (LEE, DEF)              7.360
 4. Van de Ven (TOT, DEF)          6.890
 5. Andersen (FUL, DEF)            6.580
 6. James (CHE, DEF)               6.410
 7. Guéhi (MCI, DEF)               6.280
 8. Milenković (NFO, DEF)          5.675
 9. Murillo (NFO, DEF)             5.615
10. Virgil (LIV, DEF)              5.370
```
---

## Top 10 Midfielders
```text
 1. Estêvão (CHE, MID)            11.685
 2. Buendía (AVL, MID)            10.235
 3. Szoboszlai (LIV, MID)          9.260
 4. Palmer (CHE, MID)              8.265
 5. Marmoush (MCI, MID)            8.080
 6. Wilson (FUL, MID)              7.690
 7. Semenyo (MCI, MID)             7.300
 8. Cunha (MUN, MID)               7.060
 9. Madueke (ARS, MID)             7.045
10. Mbeumo (MUN, MID)              6.820
```
---

## Top 10 Forwards
```text
 1. João Pedro (CHE, FWD)         10.635
 2. Bowen (WHU, FWD)               9.000
 3. Barry (EVE, FWD)               7.530
 4. Igor Jesus (NFO, FWD)          7.050
 5. Watkins (AVL, FWD)             6.130
 6. Brobbey (SUN, FWD)             5.065
 7. Kostoulas (BHA, FWD)           4.350
 8. Evanilson (BOU, FWD)           4.330
 9. Foster (BUR, FWD)              3.685
10. Awoniyi (NFO, FWD)             3.430
```
