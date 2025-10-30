#!/usr/bin/env python3
"""
Check if 2025 WTA data is available from Jeff Sackmann's repository
Run this monthly to see when real 2025 data is released
"""

import requests
from datetime import datetime

def check_2025_data_available():
    """Check if wta_matches_2025.csv exists in Jeff Sackmann's repo"""

    print("🔍 CHECKING FOR 2025 WTA DATA")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    urls_to_check = [
        'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2025.csv',
        'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/main/wta_matches_2025.csv'
    ]

    for url in urls_to_check:
        print(f"Checking: {url}")
        try:
            response = requests.head(url, timeout=10)

            if response.status_code == 200:
                print(f"\n✅ 2025 DATA FOUND!")
                print(f"   URL: {url}")
                print(f"   Status: Available")
                print()
                print("🎉 Real 2025 WTA data is now available!")
                print()
                print("Next steps:")
                print("1. Run: python3 update_2025_data.py")
                print("2. This will download and integrate the data")
                print("3. Retrain your model for best results")
                print()
                print("Expected additions:")
                print("   ✅ Caroline Werner (debuted 2025)")
                print("   ✅ Other tour-level 2025 debutantes")
                print("   ✅ All 2025 WTA tour matches with real stats")
                return True

            elif response.status_code == 404:
                print(f"   ❌ Not found (404)")

        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Error: {str(e)[:50]}")

    print()
    print("⏳ 2025 DATA NOT YET AVAILABLE")
    print()
    print("📌 Current status:")
    print("   - Jeff Sackmann repository not updated for 2025 yet")
    print("   - This is normal - data typically releases 1-3 months after season")
    print("   - WTA season ends: Late November")
    print("   - Expected availability: January-February 2026")
    print()
    print("💡 What you're using now:")
    print("   - 25,901 real WTA matches (2015-2024)")
    print("   - Excellent for predictions of established players")
    print("   - New players (Werner, Pohankova) predicted via ranking only")
    print()
    print("🔄 Check again later:")
    print("   - Run this script monthly: python3 src/check_2025_availability.py")
    print("   - Or run: python3 update_2025_data.py")
    print("   - Watch: https://github.com/JeffSackmann/tennis_wta")
    print()

    return False

def get_latest_available_year():
    """Check what the latest year available is"""

    current_year = datetime.now().year

    for year in range(current_year, 2023, -1):
        url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv'
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"📊 Latest available data: {year}")
                print(f"   URL: {url}")
                return year
        except:
            continue

    return None

if __name__ == "__main__":
    is_available = check_2025_data_available()

    if not is_available:
        print()
        latest = get_latest_available_year()
        if latest:
            print(f"\n✅ You can use data through {latest} (already in your dataset)")
