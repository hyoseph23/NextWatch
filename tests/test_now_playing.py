from datetime import date

import pandas as pd

from movierec.now_playing import flag_rereleases


def test_flag_rereleases():
    df = pd.DataFrame(
        {
            "tmdb_title": ["Classic", "Late US release", "New", "Undated"],
            "release_date": pd.to_datetime(["1959-08-06", "2025-11-01", "2026-10-17", None]),
        }
    )
    flagged = flag_rereleases(df, date(2026, 10, 23), max_age_days=365)
    assert flagged["is_rerelease"].tolist() == [True, False, False, False]
