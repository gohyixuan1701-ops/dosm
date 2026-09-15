
from pathlib import Path
import pandas as pd

from gdp_forecast import NAME_ALIASES  # already extended with Labuan/Putrajaya

DATA_DIR = Path(__file__).parent
SOURCE_CSV = DATA_DIR / 'accommodation_by_state.csv'

_cache = None


def get_state_accommodation(state_name):
    """Return {'establishments': int, 'rooms': int, 'year': int} for a
    state, or None if not found. No year parameter -- there's only
    ever one snapshot (see module docstring)."""
    global _cache
    if _cache is None:
        _cache = pd.read_csv(SOURCE_CSV)

    crime_name = NAME_ALIASES.get(state_name, state_name)
    row = _cache[_cache['state'] == crime_name]
    if row.empty:
        return None
    return {
        'establishments': int(row['establishments'].iloc[0]),
        'rooms': int(row['rooms'].iloc[0]),
        'year': int(row['year'].iloc[0]),
    }


def load_all():
    """Return the full table, for a rating factor that ranks states
    relative to each other (e.g. rooms per state, min-max scaled)."""
    global _cache
    if _cache is None:
        _cache = pd.read_csv(SOURCE_CSV)
    return _cache


def accommodation_score(state_name):
    """Return (score 0-100, reason). Relative ranking by ROOM COUNT
    across all 16 states -- more rooms = more capacity to host
    tourists = higher score. No year parameter: this is the single
    2022 snapshot (see module docstring)."""
    df = load_all()
    crime_name = NAME_ALIASES.get(state_name, state_name)
    if crime_name not in set(df['state']):
        return 75, 'No accommodation data available for this state; a neutral score is used.'

    values = df.set_index('state')['rooms']
    lo, hi = values.min(), values.max()
    this_value = values[crime_name]
    score = 100 if hi == lo else round(100 * (this_value - lo) / (hi - lo))
    reason = f'{this_value:,} rooms across {df.set_index("state")["establishments"][crime_name]:,} establishments (2022), relative to other states.'
    return score, reason


if __name__ == '__main__':

    print("\nAll 16 states, sorted by room count:")
    df = load_all().sort_values('rooms', ascending=False)
    print(df.to_string(index=False))
