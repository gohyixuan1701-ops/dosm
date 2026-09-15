
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

#training time=100, weight=42
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
    Relative ranking across the 14 states with crime data for that year
    (see SAFETY SCORE LIMITATION above -- not population-normalised).
    Years outside 2024-2028 are clamped to the nearest edge."""
    table = load_forecast_table()
    if table is None:
        return FALLBACK_SCORE, 'Crime forecast has not been generated yet; a neutral score is used.'

    crime_name = NAME_ALIASES.get(state_name, state_name)
    clamped_year = max(FORECAST_YEARS[0], min(FORECAST_YEARS[-1], year))
    year_rows = table[table['year'] == clamped_year]

    if crime_name not in set(year_rows['state']):
        return FALLBACK_SCORE, FALLBACK_REASON

    values = year_rows.set_index('state')['predicted_crimes']
    lo, hi = values.min(), values.max()
    this_value = values[crime_name]
    if hi == lo:
        score = 100
    else:
        score = round(100 * (hi - this_value) / (hi - lo))

    note = '' if clamped_year == year else f' (forecast clamped to {clamped_year}, closest available year)'
    reason = f'Forecast ~{this_value:,.0f} crimes statewide in {clamped_year}{note}, relative to other states.'
    return score, reason


def district_safety_score(state_name, district_name, year):
    """Return (score 0-100, reason) for a specific district, ranked
    against OTHER DISTRICTS IN THE SAME STATE (not nationally) -- this
    matches the actual decision a visitor is making ("which part of
    this state"), and avoids a decent district in a high-crime state
    looking artificially worse than it is relative to its neighbours.

    Falls back to the state-level safety_score() if no district is
    given, or the district isn't found (this is the "optional, state
    is fine" behaviour)."""
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

    values = state_rows.set_index('district')['predicted_crimes']
    lo, hi = values.min(), values.max()
    this_value = values[district_name]
    if hi == lo:
        score = 100
    else:
        score = round(100 * (hi - this_value) / (hi - lo))

    note = '' if clamped_year == year else f' (forecast clamped to {clamped_year}, closest available year)'
    reason = (f'Forecast ~{this_value:,.0f} crimes in {district_name} in {clamped_year}{note}, '
              f'relative to other districts in {state_name}.')
    return score, reason


if __name__ == '__main__':
    district_forecast, state_forecast = train_and_forecast()
    print("\n2024-2028 forecast, aggregated to state level:")
    print(state_forecast.pivot(index='state', columns='year', values='predicted_crimes').round(0))

    state_forecast.to_csv(FORECAST_CSV, index=False)
    district_forecast.to_csv(FORECAST_DISTRICT_CSV, index=False)
    print(f"\nSaved state-level forecast to {FORECAST_CSV}")
    print(f"Saved district-level forecast to {FORECAST_DISTRICT_CSV}")
