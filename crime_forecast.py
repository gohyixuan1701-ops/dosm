"""
crime_forecast.py -- RNN & LSTM crime forecasting for the Safety/Security
rating factor.

METHODOLOGY NOTE (for the report):
Individual district series are too short to model alone (only 8 annual
points: 2016-2023 from crime_district.csv). Instead of training one model
per district, a SINGLE shared model is trained across all 153 districts
that have complete 8-year data -- each district becomes one training
example, so the model learns general crime-trend dynamics by pooling
across districts, not from one long series.

Sliding windows (4 input years -> 1 output year), per district:
  - Training windows:   2016-19->2020, 2017-20->2021, 2018-21->2022
  - Validation window:  2019-22->2023   (held out, no leakage)
  - Forecast start:     2020-23->2024, then RECURSIVE from there
Both a SimpleRNN and an LSTM are trained so they can be compared (LSTM
won on validation MAE in testing: ~55 vs ~71 crimes/district/year).

MULTI-YEAR FORECAST (2024-2028), and its honest limitation:
The last known real data is 2023. To reach 2028, the model forecasts one
year at a time, then feeds its OWN prediction back in as if it were real
data to forecast the next year (this is called recursive/iterative
forecasting). This is standard practice for extending a short-horizon
model, but uncertainty compounds with distance -- 2024 is a genuine
one-step forecast from real data, while 2028 is five steps removed and
built partly on the model's own earlier guesses. State this explicitly
in the report; don't present 2028 with the same confidence as 2024.

SAFETY SCORE LIMITATION (for the report): the score below is a relative
RANKING across the 14 states with crime data (lowest forecast crime =
highest score), not normalised by population. A large state can look
"less safe" than a small one purely from having more people, not a
higher crime rate. Normalising by population_state.csv (if added to
this project) would be a straightforward improvement.

WORKFLOW:
  1. Run this file directly (`python crime_forecast.py`) to (re)train --
     writes data/crime_forecast.csv (long format: state, year, crimes).
     Not run automatically on app start -- training a neural net on every
     Flask restart would be slow and pointless.
  2. The Flask app calls safety_score(state_name, year) -- this reads the
     cached CSV, no TensorFlow needed at request time.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
SOURCE_CSV = DATA_DIR / 'crime_district.csv'
FORECAST_CSV = DATA_DIR / 'crime_forecast.csv'
FORECAST_DISTRICT_CSV = DATA_DIR / 'crime_forecast_district.csv'

WINDOW = 4
HISTORY_YEARS = list(range(2016, 2024))   # 2016..2023, real data
FORECAST_YEARS = list(range(2024, 2029))  # 2024..2028, predicted

# app.py's STATE_DATA uses different names for 3 states than this crime
# dataset does. Labuan and Putrajaya aren't in the crime dataset at all
# (too small to report separately) -- handled as a neutral fallback in
# safety_score().
NAME_ALIASES = {
    'Malacca': 'Melaka',
    'Penang': 'Pulau Pinang',
    'Kuala Lumpur': 'W.P. Kuala Lumpur',
}

FALLBACK_SCORE = 75
FALLBACK_REASON = 'No crime data available for this state; a neutral score is used.'


def _load_district_series():
    df = pd.read_csv(SOURCE_CSV)
    df = df[(df['type'] == 'all') & (df['district'] != 'All') & (df['state'] != 'Malaysia')]
    totals = df.groupby(['state', 'district', 'date'])['crimes'].sum().reset_index()
    totals['year'] = pd.to_datetime(totals['date']).dt.year

    pivot = totals.pivot_table(index=['state', 'district'], columns='year', values='crimes')
    pivot = pivot.dropna()
    pivot = pivot[HISTORY_YEARS]

    labels = pivot.index.to_frame(index=False)
    series = pivot.values.astype('float32')
    return labels, series


def _make_windows(norm, start_idx):
    X = norm[:, start_idx:start_idx + WINDOW]
    y = norm[:, start_idx + WINDOW]
    return X, y


def train_and_forecast(epochs=100, seed=42):
    """Train both models, print a validation comparison, and return a
    long-format DataFrame: state, year, predicted_crimes for 2024-2028."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    np.random.seed(seed)

    labels, series = _load_district_series()
    district_mean = series.mean(axis=1, keepdims=True)
    district_mean[district_mean == 0] = 1.0
    norm = series / district_mean

    X_train = np.concatenate([_make_windows(norm, i)[0] for i in range(3)], axis=0).reshape(-1, WINDOW, 1)
    y_train = np.concatenate([_make_windows(norm, i)[1] for i in range(3)], axis=0)
    X_val, y_val = _make_windows(norm, 3)
    X_val = X_val.reshape(-1, WINDOW, 1)

    def build_rnn():
        return keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.SimpleRNN(16, activation='tanh'),
            layers.Dense(1),
        ])

    def build_lstm():
        return keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.LSTM(16, activation='tanh'),
            layers.Dense(1),
        ])

    comparison = {}
    for name, builder in [('SimpleRNN', build_rnn), ('LSTM', build_lstm)]:
        model = builder()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # TensorBoard: logs one subfolder per model, so both training
        # curves can be compared side by side in the same dashboard.
        # This is standard TensorFlow tooling, not a PyTorch-specific
        # thing -- TensorBoard was originally built for TensorFlow.
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=str(DATA_DIR / 'logs' / f'CRIME_{name}'), histogram_freq=1
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=16,
                  validation_data=(X_val, y_val), verbose=0,
                  callbacks=[tb_callback])
        pred_val_real = model.predict(X_val, verbose=0).flatten() * district_mean.flatten()
        actual_val_real = y_val * district_mean.flatten()
        mae_real = float(np.mean(np.abs(pred_val_real - actual_val_real)))
        comparison[name] = {'model': model, 'val_mae_real': mae_real}
        print(f"{name}: validation MAE = {mae_real:.1f} crimes/district/year")

    best_name = min(comparison, key=lambda k: comparison[k]['val_mae_real'])
    print(f"Best on validation: {best_name}")
    best_model = comparison[best_name]['model']

    # --- Recursive multi-step forecast, 2024-2028 -------------------------
    window = norm[:, 4:8].copy()  # start from 2020-2023, shape (n_districts, 4)
    all_predictions = []  # list of arrays, one per forecast year (normalised)
    for _ in FORECAST_YEARS:
        step_input = window.reshape(-1, WINDOW, 1)
        pred = best_model.predict(step_input, verbose=0).flatten()
        pred = np.maximum(pred, 0)  # crime counts can't be negative
        all_predictions.append(pred)
        window = np.concatenate([window[:, 1:], pred.reshape(-1, 1)], axis=1)

    rows = []
    for year, pred_norm in zip(FORECAST_YEARS, all_predictions):
        pred_real = pred_norm * district_mean.flatten()
        year_df = labels.copy()
        year_df['year'] = year
        year_df['predicted_crimes'] = pred_real
        rows.append(year_df)
    district_forecast = pd.concat(rows, ignore_index=True)

    state_forecast = (
        district_forecast.groupby(['state', 'year'])['predicted_crimes']
        .sum().reset_index()
    )
    return district_forecast, state_forecast


def load_forecast_table():
    """Return the cached state-level forecast DataFrame, or None if
    train_and_forecast() hasn't been run yet."""
    if not FORECAST_CSV.exists():
        return None
    return pd.read_csv(FORECAST_CSV)


def load_district_forecast_table():
    """Return the cached district-level forecast DataFrame, or None."""
    if not FORECAST_DISTRICT_CSV.exists():
        return None
    return pd.read_csv(FORECAST_DISTRICT_CSV)


def get_chart_series(state_name, district_name=None):
    """Return {'years': [...], 'values': [...], 'real_count': N} for
    the "Show more detail" chart -- real 2016-2023, forecast
    2024-2028. If district_name is given, shows just that district
    (using only districts with complete real history, same set
    list_districts() offers); otherwise sums real history across ALL
    districts in the state to match how the state-level forecast
    (crime_forecast.csv) is itself built."""
    crime_name = NAME_ALIASES.get(state_name, state_name)
    labels, series = _load_district_series()

    if district_name:
        mask = (labels['state'] == crime_name) & (labels['district'] == district_name)
    else:
        mask = labels['state'] == crime_name
    if not mask.any():
        return None

    real_values = [float(v) for v in series[mask.values].sum(axis=0)]
    real_years = list(HISTORY_YEARS)

    if district_name:
        table = load_district_forecast_table()
        rows = table[(table['state'] == crime_name) & (table['district'] == district_name)] if table is not None else None
    else:
        table = load_forecast_table()
        rows = table[table['state'] == crime_name] if table is not None else None

    forecast_values = []
    for y in FORECAST_YEARS:
        if rows is None:
            forecast_values.append(None)
            continue
        r = rows[rows['year'] == y]
        col = 'predicted_crimes'
        forecast_values.append(float(r[col].iloc[0]) if len(r) else None)

    years = real_years + list(FORECAST_YEARS)
    values = real_values + forecast_values
    return {'years': years, 'values': values, 'real_count': len(real_years)}


def list_districts(state_name):
    """Return a sorted list of districts with a forecast available for
    this state (app.py's state name -- aliases are handled here), or []
    if no district data exists (e.g. Labuan, Putrajaya, or the forecast
    hasn't been generated yet)."""
    table = load_district_forecast_table()
    if table is None:
        return []
    crime_name = NAME_ALIASES.get(state_name, state_name)
    return sorted(table[table['state'] == crime_name]['district'].unique())


def safety_score(state_name, year):
    """Return (score 0-100, reason) for a state + year, higher = safer.
    Ranks by crime PER CAPITA (crimes per 1,000 people) using state
    population projections from population_score.py -- this is a real
    fix for the earlier raw-count bias (Selangor was scoring 0 purely
    for having more people, not more crime per person). Falls back to
    raw-count ranking only if population data is somehow unavailable.
    Years outside 2024-2028 are clamped to the nearest edge."""
    table = load_forecast_table()
    if table is None:
        return FALLBACK_SCORE, 'Crime forecast has not been generated yet; a neutral score is used.'

    crime_name = NAME_ALIASES.get(state_name, state_name)
    clamped_year = max(FORECAST_YEARS[0], min(FORECAST_YEARS[-1], year))
    year_rows = table[table['year'] == clamped_year]

    if crime_name not in set(year_rows['state']):
        return FALLBACK_SCORE, FALLBACK_REASON

    note = '' if clamped_year == year else f' (forecast clamped to {clamped_year}, closest available year)'

    from population_score import predict_state_population, crime_rate_per_capita
    rates = {}
    for _, row in year_rows.iterrows():
        pop = predict_state_population(row['state'], clamped_year)
        rate = crime_rate_per_capita(row['predicted_crimes'], pop)
        if rate is not None:
            rates[row['state']] = rate

    if crime_name in rates and len(rates) >= 2:
        values = pd.Series(rates)
        lo, hi = values.min(), values.max()
        this_value = values[crime_name]
        score = 100 if hi == lo else round(100 * (hi - this_value) / (hi - lo))
        reason = (f'Forecast ~{this_value:.2f} crimes per 1,000 people in {clamped_year}{note}, '
                  f'relative to other states.')
        return score, reason

    # Fallback: per-capita unavailable -- raw-count ranking (old behaviour)
    values = year_rows.set_index('state')['predicted_crimes']
    lo, hi = values.min(), values.max()
    this_value = values[crime_name]
    score = 100 if hi == lo else round(100 * (hi - this_value) / (hi - lo))
    reason = (f'Forecast ~{this_value:,.0f} crimes statewide in {clamped_year}{note}, relative to '
              f'other states (raw count -- population data unavailable).')
    return score, reason


def district_safety_score(state_name, district_name, year):
    """Return (score 0-100, reason) for a specific district, ranked
    against OTHER DISTRICTS IN THE SAME STATE (not nationally) -- this
    matches the actual decision a visitor is making ("which part of
    this state"), and avoids a decent district in a high-crime state
    looking artificially worse than it is relative to its neighbours.

    Uses crime PER CAPITA when this district has a verified population
    match (see population_score.py's DISTRICT POPULATION COVERAGE note
    -- about 72% of districts do). Falls back to the raw-count ranking
    otherwise, since crime_district.csv uses police-district boundaries
    that don't all line up with population_district.csv's administrative
    districts, and fabricating a population split would be worse than
    the raw-count limitation it would replace. The reason text always
    states which method was actually used.

    Falls back to the state-level safety_score() if no district is
    given, or the district isn't found in the crime forecast at all
    (this is the "optional, state is fine" behaviour)."""
    if not district_name:
        return safety_score(state_name, year)

    table = load_district_forecast_table()
    if table is None:
        return safety_score(state_name, year)

    crime_name = NAME_ALIASES.get(state_name, state_name)
    clamped_year = max(FORECAST_YEARS[0], min(FORECAST_YEARS[-1], year))
    state_rows = table[(table['state'] == crime_name) & (table['year'] == clamped_year)]

    if district_name not in set(state_rows['district']):
        return safety_score(state_name, year)

    note = '' if clamped_year == year else f' (forecast clamped to {clamped_year}, closest available year)'

    from population_score import predict_district_population, crime_rate_per_capita
    this_pop = predict_district_population(state_name, district_name, clamped_year)

    if this_pop is not None:
        rates = {}
        for _, row in state_rows.iterrows():
            pop = predict_district_population(state_name, row['district'], clamped_year)
            rate = crime_rate_per_capita(row['predicted_crimes'], pop)
            if rate is not None:
                rates[row['district']] = rate
        if district_name in rates and len(rates) >= 2:
            values = pd.Series(rates)
            lo, hi = values.min(), values.max()
            this_value = values[district_name]
            score = 100 if hi == lo else round(100 * (hi - this_value) / (hi - lo))
            reason = (f'Forecast ~{this_value:.2f} crimes per 1,000 people in {district_name} in '
                      f'{clamped_year}{note}, relative to other districts in {state_name} with population data.')
            return score, reason

    # Fallback: no verified population match for this district -- raw count
    values = state_rows.set_index('district')['predicted_crimes']
    lo, hi = values.min(), values.max()
    this_value = values[district_name]
    score = 100 if hi == lo else round(100 * (hi - this_value) / (hi - lo))
    reason = (f'Forecast ~{this_value:,.0f} crimes in {district_name} in {clamped_year}{note}, relative to '
              f'other districts in {state_name} (raw count -- no matching population data for this district).')
    return score, reason


if __name__ == '__main__':
    district_forecast, state_forecast = train_and_forecast()
    print("\n2024-2028 forecast, aggregated to state level:")
    print(state_forecast.pivot(index='state', columns='year', values='predicted_crimes').round(0))

    state_forecast.to_csv(FORECAST_CSV, index=False)
    district_forecast.to_csv(FORECAST_DISTRICT_CSV, index=False)
    print(f"\nSaved state-level forecast to {FORECAST_CSV}")
    print(f"Saved district-level forecast to {FORECAST_DISTRICT_CSV}")
