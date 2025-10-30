# Waiting for 2025 WTA Data Strategy

## Current Status

✅ **Training Data Ready:** 25,901 real WTA matches (2015-2024)
⏳ **2025 Data:** Waiting for Jeff Sackmann to release
📅 **Expected:** Usually 1-3 months after season ends (December 2025 - February 2026)

## What Happens When 2025 Data Arrives

When Jeff Sackmann updates his repository with real 2025 data:

1. **Caroline Werner** ✅ Will be included
   - Made WTA Tour debut in 2025
   - Ranking: #222 (career high)
   - Has tour-level match statistics

2. **Mia Pohankova** ⚠️ Might NOT be included
   - Only 16 years old
   - Primarily plays ITF events (not WTA tour-level)
   - Unless she plays WTA main draws, won't be in tour-level dataset

3. **Other New Players** ✅ Will be included
   - Any player who competed in WTA tour-level events in 2025
   - All with complete real statistics

## How to Check for 2025 Data

### Manual Check
Visit: https://github.com/JeffSackmann/tennis_wta

Look for: `wta_matches_2025.csv`

### Automated Check
Run this command:
```bash
python3 src/check_2025_availability.py
```

This will:
- Check if 2025 data is available
- Show release date if found
- Automatically download and integrate if available

### Regular Update Check
Run this monthly:
```bash
python3 update_2025_data.py
```

## What About Players Not in the Dataset?

### The Cold-Start Problem

When predicting a match involving a player who isn't in your training data (like Pohankova or Werner currently), the model has several options:

#### Strategy 1: Ranking-Based Prediction (Current Default)
```python
# For unknown players, use:
- Current WTA ranking (most important)
- Ranking points
- Surface type
- Tournament level
- Opponent's historical data (if known)
```

**Example:**
```
Match: Iga Swiatek (known) vs Caroline Werner (unknown)
Model uses:
✅ Swiatek's historical stats, ranking #2
✅ Werner's current ranking #222
✅ Surface: hard court
✅ Tournament: WTA 250
→ Prediction: Swiatek heavily favored (ranking diff = 220)
```

#### Strategy 2: Average Player Profile
For completely unknown players:
```python
# Use average statistics for their ranking tier
Ranking 1-10: Elite player averages
Ranking 11-50: Top player averages
Ranking 51-100: Established pro averages
Ranking 100+: Lower-ranked pro averages
```

#### Strategy 3: Reject Prediction
```python
# Don't make predictions for matches with unknown players
if player_not_in_dataset:
    return "Insufficient data for prediction"
```

### Which Strategy is Best?

**For now (before 2025 data):**
- Use **Strategy 1** (Ranking-Based)
- Ranking is the strongest predictor anyway
- Most new players perform according to their ranking

**After 2025 data arrives:**
- Full statistics for Werner and other tour-level debutantes
- Better predictions with actual playing style data

## Timeline Expectations

### Jeff Sackmann Update History
Based on past years:
- Data typically releases 1-3 months after season ends
- Season ends: Late November (WTA Finals)
- Expected release: **January-February 2026**

### What to Do While Waiting

1. ✅ **Use Current System**
   - 25,901 matches is excellent training data
   - Rankings reflect current form
   - Model still makes accurate predictions

2. ✅ **Check Monthly**
   - Run `python3 update_2025_data.py` once per month
   - Automated check for new data

3. ✅ **Monitor GitHub**
   - Watch: https://github.com/JeffSackmann/tennis_wta
   - Star the repository for notifications

4. ⚠️ **Accept Limitations**
   - Predictions for brand-new players less accurate
   - Predictions for established players still excellent
   - Trade-off worth it for data quality

## When Should You Manually Add Data?

Only if you NEED predictions for new players RIGHT NOW and:

1. Can access complete real statistics (not just scores)
2. Have time to manually curate from WTA.com/Flashscore
3. Willing to maintain data quality standards

Otherwise: **Wait for Sackmann** ✅

## Summary

| Aspect | Status |
|--------|--------|
| Current training data | ✅ Excellent (25,901 matches) |
| Predictions for Swiatek, Sabalenka, etc. | ✅ Accurate (lots of history) |
| Predictions for Werner, Pohankova | ⚠️ Ranking-based only |
| When this improves | 📅 Jan-Feb 2026 (expected) |
| What you should do | ✅ Wait & check monthly |

---

**Philosophy:** Better to wait for real data than compromise accuracy with synthetic data.

**Next Action:** Run `python3 update_2025_data.py` in January 2026.
