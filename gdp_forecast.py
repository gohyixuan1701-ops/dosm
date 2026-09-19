"""
gdp_forecast.py -- RNN & LSTM state GDP forecasting.

METHODOLOGY NOTE (for the report):
Unlike population (smooth, near-linear -- see population_score.py),
GDP growth shows a genuine irregular shock: every single state's
growth went NEGATIVE in 2020 (real recession, -1.1% to -9.1%), then
sharply REBOUNDED in 2022 (up to +13.3%), before settling back to a
more normal range by 2024-2025. A straight line cannot capture a
dip-then-overshoot pattern like that -- this is exactly the kind of
irregular, non-linear movement RNN/LSTM is suited for, unlike
population's steady trend. Two different data shapes, two different
justified methods -- not one technique applied everywhere by default.

SMALLER POOL THAN CRIME (limitation to disclose): crime's shared-model
approach worked well because 153 districts could be pooled together.
GDP only has 15 states with complete history (W.P. Putrajaya excluded,
see below) -- a much smaller pool. A smaller network (8 units, vs
crime's 16) is used here to reduce overfitting risk on that smaller
dataset; results should be read with correspondingly more caution.

DATA COVERAGE: uses sector p0 ("GDP at purchasers' prices" -- the
overall total, not a specific industry) from the 'abs' (absolute
value, RM million) series in gdp_state_real_supply.csv. 'Supra' is a
national/supra-regional adjustment row, not a real state, and is
excluded. W.P. Putrajaya only has data from 2023 onward (3 years --
too little for a 4-year window) and is excluded from model training;
its real known values are still served directly by get_state_gdp()
for the years that exist.

FORECAST HORIZON: unlike crime (last real year 2023, forecast
2024-2028), GDP data already includes real 2024-2025 values. The
model therefore only forecasts 2026-2028 (3 recursive steps, not 5) --
get_state_gdp() returns the REAL value directly for any year <= 2025,
and only uses the model's prediction for 2026 onward.

WORKFLOW:
  1. Run this file directly (`python gdp_forecast.py`) to (re)train --
     writes gdp_forecast.csv (2026-2028 predictions only; real
     2015-2025 values are already in the source file and aren't
     duplicated here).
  2. Call get_state_gdp(state_name, year) for a blended real/forecast
     lookup -- no TensorFlow needed at request time.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from crime_forecast import NAME_ALIASES as CRIME_NAME_ALIASES

# GDP data includes Labuan and Putrajaya (crime data doesn't have either,
# so crime_forecast.py's alias map has no entry for them) -- extend it
# locally for this module's own state-name needs.
NAME_ALIASES = {**CRIME_NAME_ALIASES, 'Labuan': 'W.P. Labuan', 'Putrajaya': 'W.P. Putrajaya'}

DATA_DIR = Path(__file__).parent
SOURCE_CSV = DATA_DIR / 'gdp_state_real_supply.csv'
FORECAST_CSV = DATA_DIR / 'gdp_forecast.csv'

WINDOW = 4
LAST_REAL_YEAR = 2025
FORECAST_YEARS = [2026, 2027, 2028]
EXCLUDED_STATES = {'Supra', 'W.P. Putrajaya'}  # not real states / insufficient history

# The overall-GDP sector label as it appears in the 'sector' column.
# NOTE: this must match your CSV exactly. DOSM's own code for this is
# 'p0' (see gdp_lookup.csv) -- if your file uses the short code instead
# of the readable label, change this one line to 'p0'.
SECTOR_OVERALL = "GDP at purchasers' prices"


def _load_state_series():
    df = pd.read_csv(SOURCE_CSV)
    df = df[(df['sector'] == SECTOR_OVERALL) & (df['series'] == 'abs') & (~df['state'].isin(EXCLUDED_STATES))]
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year

    pivot = df.pivot_table(index='state', columns='year', values='value')
    pivot = pivot.dropna()  # only states with complete history
    years = sorted(pivot.columns)
    pivot = pivot[years]

    labels = pivot.index.to_frame(index=False)
    series = pivot.values.astype('float32')
    return labels, series, years


def _make_windows(norm, start_idx):
    X = norm[:, start_idx:start_idx + WINDOW]
    y = norm[:, start_idx + WINDOW]
    return X, y


def train_and_forecast(epochs=150, seed=42):
    """Train both models, print a validation comparison, and return a
    long-format DataFrame: state, year, predicted_gdp for 2026-2028."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    np.random.seed(seed)

    labels, series, years = _load_state_series()
    state_mean = series.mean(axis=1, keepdims=True)
    state_mean[state_mean == 0] = 1.0
    norm = series / state_mean

    n_windows = len(years) - WINDOW  # 11 years, window 4 -> 7 possible windows
    train_offsets = list(range(n_windows - 1))  # all but the last -> predicts up to 2024
    val_offset = n_windows - 1  # predicts the final real year, 2025 -- genuinely held out

    X_train = np.concatenate([_make_windows(norm, i)[0] for i in train_offsets], axis=0).reshape(-1, WINDOW, 1)
    y_train = np.concatenate([_make_windows(norm, i)[1] for i in train_offsets], axis=0)
    X_val, y_val = _make_windows(norm, val_offset)
    X_val = X_val.reshape(-1, WINDOW, 1)

    def build_rnn():
        return keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.SimpleRNN(8, activation='tanh'),  # smaller than crime's 16 -- see module docstring
            layers.Dense(1),
        ])

    def build_lstm():
        return keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.LSTM(8, activation='tanh'),
            layers.Dense(1),
        ])

    comparison = {}
    for name, builder in [('SimpleRNN', build_rnn), ('LSTM', build_lstm)]:
        model = builder()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=str(DATA_DIR / 'logs' / f'GDP_{name}'), histogram_freq=1
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=8,
                  validation_data=(X_val, y_val), verbose=0,
                  callbacks=[tb_callback])
        pred_val_real = model.predict(X_val, verbose=0).flatten() * state_mean.flatten()
        actual_val_real = y_val * state_mean.flatten()
        mae_real = float(np.mean(np.abs(pred_val_real - actual_val_real)))
        comparison[name] = {'model': model, 'val_mae_real': mae_real}
        print(f"{name}: validation MAE = RM{mae_real:,.0f} million (predicting real 2025 GDP)")

    best_name = min(comparison, key=lambda k: comparison[k]['val_mae_real'])
    print(f"Best on validation: {best_name}")
    best_model = comparison[best_name]['model']

    # --- Recursive forecast, 2026-2028 (only 3 steps -- 2024/2025 are real) ---
    window = norm[:, -WINDOW:].copy()  # last known real years: 2022-2025
    predictions = []
    for _ in FORECAST_YEARS:
        pred = best_model.predict(window.reshape(-1, WINDOW, 1), verbose=0).flatten()
        pred = np.maximum(pred, 0)  # GDP can't be negative
        predictions.append(pred)
        window = np.concatenate([window[:, 1:], pred.reshape(-1, 1)], axis=1)

    rows = []
    for year, pred_norm in zip(FORECAST_YEARS, predictions):
        pred_real = pred_norm * state_mean.flatten()
        year_df = labels.copy()
        year_df['year'] = year
        year_df['predicted_gdp'] = pred_real
        rows.append(year_df)
    return pd.concat(rows, ignore_index=True)


def load_forecast_table():
    if not FORECAST_CSV.exists():
        return None
    return pd.read_csv(FORECAST_CSV)


_real_gdp_cache = None


def get_state_gdp(state_name, year):
    """Blended lookup: returns the REAL known value for any year <=
    2025, or the model's forecast for 2026-2028 (clamped beyond that).
    Returns None if the state has no data at all for the requested
    year (e.g. Putrajaya before 2023)."""
    global _real_gdp_cache
    crime_name = NAME_ALIASES.get(state_name, state_name)

    if year <= LAST_REAL_YEAR:
        if _real_gdp_cache is None:
            df = pd.read_csv(SOURCE_CSV)
            df = df[(df['sector'] == SECTOR_OVERALL) & (df['series'] == 'abs')]
            df = df.copy()
            df['year'] = pd.to_datetime(df['date']).dt.year
            _real_gdp_cache = df
        row = _real_gdp_cache[(_real_gdp_cache['state'] == crime_name) & (_real_gdp_cache['year'] == year)]
        return float(row['value'].iloc[0]) if len(row) else None

    table = load_forecast_table()
    if table is None:
        return None
    clamped_year = min(FORECAST_YEARS[-1], year)
    row = table[(table['state'] == crime_name) & (table['year'] == clamped_year)]
    return float(row['predicted_gdp'].iloc[0]) if len(row) else None


ALL_STATES = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
              'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor',
              'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']


def get_chart_series(state_name):
    """Return {'years': [...], 'values': [...], 'real_count': N} for
    the "Show more detail" chart -- real 2015-2025, forecast
    2026-2028. None entries mean no data for that year (e.g. Putrajaya
    before 2023) -- chart should skip/gap them, not treat as zero."""
    years = list(range(2015, FORECAST_YEARS[-1] + 1))
    values = [get_state_gdp(state_name, y) for y in years]
    real_count = sum(1 for y in years if y <= LAST_REAL_YEAR)
    return {'years': years, 'values': values, 'real_count': real_count}


def gdp_score(state_name, year):
    """Return (score 0-100, reason). Relative ranking by GDP across
    all states with a value for the given year (Putrajaya may be
    missing for years it has no data -- see get_state_gdp()).

    ASSUMPTION (same judgment call as population_score() -- confirm/
    flip if you disagree): HIGHER GDP -> HIGHER score, on the basis
    that a bigger state economy generally means more infrastructure
    and amenities for a visitor."""
    values = {}
    for s in ALL_STATES:
        v = get_state_gdp(s, year)
        if v is not None:
            values[s] = v

    crime_name = NAME_ALIASES.get(state_name, state_name)
    if crime_name not in values or len(values) < 2:
        return 75, 'No GDP data available for this state/year; a neutral score is used.'

    series = pd.Series(values)
    lo, hi = series.min(), series.max()
    this_value = series[crime_name]
    score = 100 if hi == lo else round(100 * (this_value - lo) / (hi - lo))
    reason = f'GDP ~RM{this_value:,.0f} million in {year}, relative to other states (higher = more infrastructure/amenities, assumption).'
    return score, reason


if __name__ == '__main__':
    forecast = train_and_forecast()
    print("\n2026-2028 forecast (RM million):")
    print(forecast.pivot(index='state', columns='year', values='predicted_gdp').round(0))

    forecast.to_csv(FORECAST_CSV, index=False)
    print(f"\nSaved to {FORECAST_CSV}")

    print("\nBlended lookup spot-check:")
    for state, year in [('Selangor', 2024), ('Selangor', 2027), ('Putrajaya', 2024), ('Putrajaya', 2027)]:
        print(f"  {state} {year}: {get_state_gdp(state, year)}")
