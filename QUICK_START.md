# WTA Tennis Predictor - Quick Start

## Current Setup ✅

**Training Data:** 25,901 real WTA matches (2015-2024)  
**Players Included:** ~500 established tour players  
**Data Quality:** 100% real, no simulated statistics

## Why Some Players Are Missing

### Missing: Caroline Werner, Mia Pohankova, other 2025 debutantes

**Reason:** Your dataset contains WTA tour-level matches through 2024.
- **Caroline Werner:** Made WTA debut in 2025 ✅ Will appear when 2025 data released
- **Mia Pohankova:** Only 16, plays mostly ITF events ⚠️ May not appear (not tour-level)
- **Other new players:** Will appear when 2025 data released

## Checking for 2025 Data

### Quick Check
```bash
python3 src/check_2025_availability.py
```

### When Is It Expected?
- **WTA Season Ends:** Late November 2025
- **Expected Release:** January-February 2026
- **Check Monthly:** Run the command above

### When Data Arrives
```bash
python3 update_2025_data.py    # Downloads and integrates 2025 data
python3 train_model.py         # Retrain with new data
```

## How Predictions Work Now

### For Established Players (e.g., Iga Swiatek, Aryna Sabalenka)
✅ **Excellent accuracy** - lots of historical data
- Full playing style statistics
- Surface preferences
- Head-to-head records
- Form trends

### For New/Unknown Players (e.g., Werner, Pohankova)
⚠️ **Ranking-based predictions** - limited data
- Uses current WTA ranking (strongest predictor)
- Uses surface type and tournament level
- Uses opponent's historical data if available
- Still reasonably accurate (ranking matters most)

**Example:**
```
Match: Iga Swiatek (#2) vs Caroline Werner (#222)
Prediction: Swiatek heavily favored
Basis: 220 ranking difference + Swiatek's historical dominance
Accuracy: Good (ranking differential is highly predictive)
```

## Files You Should Know

### Data
- `data/real_wta_matches.csv` - Your 25,901 training matches

### Documentation
- `DATA_APPROACH.md` - Philosophy: real data only
- `WAITING_FOR_2025_DATA.md` - Strategy for new players
- `QUICK_START.md` - This file

### Scripts
- `src/check_2025_availability.py` - Check if 2025 data is available
- `src/wta_2025_data_collector.py` - Fetches real 2025 data (when available)
- `update_2025_data.py` - One-command update

## Monthly Routine

**Set a monthly reminder:**

```bash
# Check for 2025 data (takes 5 seconds)
python3 src/check_2025_availability.py

# If available, update
python3 update_2025_data.py
python3 train_model.py
```

## FAQ

**Q: Why don't you add Caroline Werner manually?**  
A: We'd need complete match statistics (aces, serve %, break points) from official sources. Without them, fake statistics would hurt prediction accuracy.

**Q: Can the model still predict matches with unknown players?**  
A: Yes! It uses their ranking, which is the strongest predictor anyway. Just less refined than with full statistics.

**Q: What if I need Werner/Pohankova predictions RIGHT NOW?**  
A: The model will use ranking-based predictions. For a rank #222 vs rank #2, it will still predict correctly (higher rank wins). Just won't have playing style nuances.

**Q: Should I use synthetic/generated data while waiting?**  
A: No. Random statistics teach the model false patterns, reducing accuracy on real matches. Better to wait.

## Summary

| What | Status |
|------|--------|
| Current data quality | ✅ Excellent (25,901 real matches) |
| Predictions for top players | ✅ Highly accurate |
| Predictions for new players | ⚠️ Ranking-based (good, not great) |
| When this improves | 📅 Jan-Feb 2026 |
| Your action needed | 🗓️ Check monthly |

---

**You're all set!** The system prioritizes accuracy over completeness. When 2025 data arrives, you'll get predictions for Werner and other new tour-level players automatically.
