"""
tourism_forecast.py -- RNN & LSTM state tourism demand forecasting.
This REPLACES the placeholder mock 'growth' logic that demand_score()
in rating.py has been running on since the dashboard was first built.

METHODOLOGY NOTE (for the report):
This is the most volatile dataset in the whole project -- more so than
GDP. Domestic visitor arrivals (state, 2018-2025, from DOSM's Domestic
Tourism Survey Table 9) show a genuine collapse-and-recovery cycle:
Selangor went 30,179 (2018) -> 33,589 (2019) -> 19,715 (2020) ->
10,211 thousand visitors (2021) -- a 66% COLLAPSE from peak at the
COVID trough -- then recovered to 21,990 -> 27,579 -> 34,461 -> 36,376
by 2025, ending ABOVE pre-COVID levels. Every state shows the same
V-shaped pattern. A straight line cannot represent a near-70% collapse
followed by a multi-year recovery -- this is unambiguously an RNN/LSTM
case (like crime and GDP), not a linear-regression case (like
population and CPI, which were genuinely smooth).

HONEST LIMITATION -- this is a harder problem than crime or GDP:
only 16 states x 4 usable windows = 48 training examples (fewer than
GDP's already-small pool, let alone crime's 153 districts) to learn a
pattern this dramatic. Treat this model's output with real caution,
especially the recursive 2027/2028 predictions -- state this
explicitly in the report rather than presenting it with false
confidence. A smaller network (8 units, matching gdp_forecast.py's
choice) is used to reduce overfitting risk on this small dataset.

NO POPULATION NORMALISATION NEEDED (unlike crime): a bigger visitor
count is a real, meaningful "more popular" signal here, not an
unfairness to correct for the way raw crime counts penalised
populous states. Selangor genuinely attracting more visitors than
Perlis is the actual signal this factor is supposed to capture.

WORKFLOW: same as crime_forecast.py / gdp_forecast.py -- run this file
directly to train and save tourism_forecast.csv (2026-2028 only; real
2018-2025 values are already in tourism_visitors_by_state.csv and
aren't duplicated here). get_state_visitors() is the blended lookup
for the app to eventually call -- no TensorFlow needed at request time.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from gdp_forecast import NAME_ALIASES  # already extended with Labuan/Putrajaya

DATA_DIR = Path(__file__).parent
SOURCE_CSV = DATA_DIR / 'tourism_visitors_by_state.csv'
FORECAST_CSV = DATA_DIR / 'tourism_forecast.csv'

WINDOW = 4
LAST_REAL_YEAR = 2025
FORECAST_YEARS = [2026, 2027, 2028]


def _load_state_series():
    df = pd.read_csv(SOURCE_CSV)
    pivot = df.pivot_table(index='state', columns='year', values='visitors_thousands')
    pivot = pivot.dropna()
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
    long-format DataFrame: state, year, predicted_visitors_thousands
    for 2026-2028."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    np.random.seed(seed)

    labels, series, years = _load_state_series()
    state_mean = series.mean(axis=1, keepdims=True)
    state_mean[state_mean == 0] = 1.0
    norm = series / state_mean

    n_windows = len(years) - WINDOW  # 8 years, window 4 -> 4 possible windows
    train_offsets = list(range(n_windows - 1))  # predicts up to 2024
    val_offset = n_windows - 1  # predicts the final real year, 2025

    X_train = np.concatenate([_make_windows(norm, i)[0] for i in train_offsets], axis=0).reshape(-1, WINDOW, 1)
    y_train = np.concatenate([_make_windows(norm, i)[1] for i in train_offsets], axis=0)
    X_val, y_val = _make_windows(norm, val_offset)
    X_val = X_val.reshape(-1, WINDOW, 1)

    def build_rnn():
        return keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.SimpleRNN(8, activation='tanh'),  # small pool (16 states) -- see module docstring
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
            log_dir=str(DATA_DIR / 'logs' / f'TOURISM_{name}'), histogram_freq=1
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=8,
                  validation_data=(X_val, y_val), verbose=0,
                  callbacks=[tb_callback])
        pred_val_real = model.predict(X_val, verbose=0).flatten() * state_mean.flatten()
        actual_val_real = y_val * state_mean.flatten()
        mae_real = float(np.mean(np.abs(pred_val_real - actual_val_real)))
        comparison[name] = {'model': model, 'val_mae_real': mae_real}
        print(f"{name}: validation MAE = {mae_real:,.0f} thousand visitors (predicting real 2025)")

    best_name = min(comparison, key=lambda k: comparison[k]['val_mae_real'])
    print(f"Best on validation: {best_name}")
    best_model = comparison[best_name]['model']

    # --- Recursive forecast, 2026-2028 (2018-2025 are all real) ---
    window = norm[:, -WINDOW:].copy()  # last known real years: 2022-2025
    predictions = []
    for _ in FORECAST_YEARS:
        pred = best_model.predict(window.reshape(-1, WINDOW, 1), verbose=0).flatten()
        pred = np.maximum(pred, 0)  # visitor counts can't be negative
        predictions.append(pred)
        window = np.concatenate([window[:, 1:], pred.reshape(-1, 1)], axis=1)

    rows = []
    for year, pred_norm in zip(FORECAST_YEARS, predictions):
        pred_real = pred_norm * state_mean.flatten()
        year_df = labels.copy()
        year_df['year'] = year
        year_df['predicted_visitors_thousands'] = pred_real
        rows.append(year_df)
    return pd.concat(rows, ignore_index=True)


def load_forecast_table():
    if not FORECAST_CSV.exists():
        return None
    return pd.read_csv(FORECAST_CSV)


_real_cache = None


def get_state_visitors(state_name, year):
    """Blended lookup: real known value for year <= 2025, model
    forecast for 2026-2028 (clamped beyond that). Returns None if no
    data exists for that state/year at all."""
    global _real_cache
    crime_name = NAME_ALIASES.get(state_name, state_name)

    if year <= LAST_REAL_YEAR:
        if _real_cache is None:
            _real_cache = pd.read_csv(SOURCE_CSV)
        row = _real_cache[(_real_cache['state'] == crime_name) & (_real_cache['year'] == year)]
        return float(row['visitors_thousands'].iloc[0]) if len(row) else None

    table = load_forecast_table()
    if table is None:
        return None
    clamped_year = min(FORECAST_YEARS[-1], year)
    row = table[(table['state'] == crime_name) & (table['year'] == clamped_year)]
    return float(row['predicted_visitors_thousands'].iloc[0]) if len(row) else None


ALL_STATES = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
              'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor',
              'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']


def get_chart_series(state_name):
    """Return {'years': [...], 'values': [...], 'real_count': N} for
    the "Show more detail" chart -- the first `real_count` points are
    real (2018-2025), the rest are the model's forecast (2026-2028).
    None entries in values mean no data for that year (chart should
    skip/gap them, not treat as zero)."""
    years = list(range(2018, FORECAST_YEARS[-1] + 1))
    values = [get_state_visitors(state_name, y) for y in years]
    real_count = sum(1 for y in years if y <= LAST_REAL_YEAR)
    return {'years': years, 'values': values, 'real_count': real_count}


def tourism_trend_score(state_name, year):
    """Return (score 0-100, reason). Relative ranking by visitor
    volume across all 16 states for the given year -- more visitors
    = more popular/established as a destination = higher score. This
    REPLACES rating.py's old mock 'growth' placeholder logic."""
    values = {}
    for s in ALL_STATES:
        v = get_state_visitors(s, year)
        if v is not None:
            values[s] = v

    crime_name = NAME_ALIASES.get(state_name, state_name)
    if crime_name not in values or len(values) < 2:
        return 75, 'No tourism trend data available for this state/year; a neutral score is used.'

    series = pd.Series(values)
    lo, hi = series.min(), series.max()
    this_value = series[crime_name]
    score = 100 if hi == lo else round(100 * (this_value - lo) / (hi - lo))
    reason = f'~{this_value:,.0f} thousand visitors forecast for {year}, relative to other states.'
    return score, reason


if __name__ == '__main__':
    forecast = train_and_forecast()
    print("\n2026-2028 forecast (thousand visitors):")
    print(forecast.pivot(index='state', columns='year', values='predicted_visitors_thousands').round(0))

    forecast.to_csv(FORECAST_CSV, index=False)
    print(f"\nSaved to {FORECAST_CSV}")

    print("\nBlended lookup spot-check:")
    for state, year in [('Selangor', 2024), ('Selangor', 2027)]:
        print(f"  {state} {year}: {get_state_visitors(state, year):,.0f} thousand")
