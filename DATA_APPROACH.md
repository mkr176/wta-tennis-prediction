# WTA Tennis Prediction - Data Approach

## Philosophy: Real Data Only

This WTA prediction system uses **ONLY real, verified match data**. No simulated or generated statistics.

## Current Dataset

### ✅ What You Have
- **25,901 real WTA matches** (2015-2024)
- Source: Jeff Sackmann's tennis_wta repository
- Includes: All tour-level singles matches with complete statistics
- Years: 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024

### 📊 Data Breakdown by Year
```
2015: 2,651 matches
2016: 2,923 matches
2017: 2,862 matches
2018: 2,847 matches
2019: 2,652 matches
2020: 1,276 matches (COVID-shortened season)
2021: 2,597 matches
2022: 2,594 matches
2023: 2,810 matches
2024: 2,689 matches
```

## How 2025 Tournament Data Gets Updated

There's a simple update script: `update_2025_data.py`

To check for new data, run:
```bash
python3 update_2025_data.py
```

This script does 3 things:
1. Runs `src/wta_2025_data_collector.py` to check for 2025 real data
2. If found, runs `src/real_wta_data_collector.py` to integrate it
3. Prompts you to retrain the model for maximum benefit

## Is the 2025 Data Real?

**YES - When 2025 data is added, it will be 100% REAL.**

### ✅ What's Real (Everything)
- Tournament results - Actual match outcomes (who won/lost)
- Tournament details - Real event names, dates, surfaces
- Match scores - Actual final scores
- Player names - Real WTA players
- **Match statistics - Real aces, double faults, serve %, break points**
  - Only added when available from official source
  - Never randomly generated

### ❌ What's NOT Used
- No randomly generated statistics
- No simulated matches
- No predicted outcomes treated as real data

## Why Real Data Only?

Training a prediction model on fake statistics **reduces accuracy** on real matches.

### The Problem with Generated Data

**Bad approach** (what we DON'T do):
```python
'winner_aces': np.random.randint(5, 18),          # Random for everyone
'winner_first_serve_pct': np.random.uniform(0.62, 0.75),  # Same range for all
```

**Why it's harmful:**
- Aryna Sabalenka (rank 1) gets random serve % = 68%
- Unknown rank 100 player also gets random serve % = 68%
- Model learns: "Rankings don't correlate with performance" ❌
- **Result: Worse predictions on real matches**

### Better to Have Less Real Data

- ✅ **25,901 real matches** → Excellent training data
- ✅ **26,000 real matches** (with 2025) → Slightly better
- ❌ **26,500 matches** (500 fake) → WORSE predictions

## Where the Data Comes From

### Primary Source: Jeff Sackmann's Repository
- URL: https://github.com/JeffSackmann/tennis_wta
- Most comprehensive free tennis database
- Updated regularly (usually a few months after season ends)
- **2025 Status:** Not yet available (as of October 2025)

### Alternative Sources (Manual Curation)
If you need 2025 predictions before Sackmann updates:

1. **Wait** (Recommended)
   - Historical data predicts current players well
   - Player styles don't change drastically year-to-year

2. **Purchase Real Data**
   - BigDataBall: $25 for 2025 WTA season
   - Includes complete match statistics

3. **Manual Entry** (Advanced)
   - Collect from WTA.com, Flashscore, Tennis Abstract
   - Only add matches with complete real statistics
   - Time-intensive but maintains accuracy

## Model Training

### Current Training Data
```python
# File: data/real_wta_matches.csv
Years: 2015-2024
Matches: 25,901
Players: ~500 unique players
```

### What the Model Learns From
- **Win/Loss patterns** by player, surface, ranking
- **Head-to-head records**
- **Statistical correlations**: serve %, aces, break points vs outcomes
- **Ranking strength**: how ranking differences predict results
- **Surface effects**: player performance on hard/clay/grass

### Why Historical Data Works for Current Predictions
- Player rankings reflect recent form
- Top players maintain consistent performance levels
- Surface preferences are stable
- Statistical patterns are consistent across years

## Summary

| Aspect | Status |
|--------|--------|
| Training data | ✅ 25,901 real matches (2015-2024) |
| 2025 data | ⏳ Will be added when real data available |
| Generated statistics | ❌ Never used |
| Prediction accuracy | ✅ Optimized for real matches |
| Update frequency | 📅 Check monthly for Sackmann updates |

## Quick Reference

### Check for 2025 Updates
```bash
python3 update_2025_data.py
```

### Verify Data Quality
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/real_wta_matches.csv'); print(f'Matches: {len(df)}'); print(df['date'].astype(str).str[:4].value_counts().sort_index())"
```

### Train Model
```bash
python3 train_model.py
```

---

**Last Updated:** October 2025
**Data Coverage:** 2015-2024 (Real data only)
**Next Expected Update:** When Jeff Sackmann releases 2025 data
