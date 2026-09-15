"""
currency.py -- currency conversion, NOT machine learning (see app notes).

Reads Bank Negara Malaysia mid rates from data/exchange-rates.csv. Local,
offline, no network call -- fast and no dependency on an external API
being up at demo time.

DATA QUALITY NOTE (kept for the report): this file's data (1 Sep 2020 to
11 Sep 2026, daily) is genuinely clean -- rows after the last real date
are blank padding from the spreadsheet export, not corrupted values like
an earlier download had. A sanity filter (USD must fall in a realistic
MYR/USD range) and date-parse check are still applied defensively so a
malformed or blank row is skipped rather than crashing the app.

Covers 26 currencies, including Malaysia's top tourism source markets:
Singapore (SGD), Indonesia (IDR), Thailand (THB) and China (CNY).
"""

from pathlib import Path
import csv
from datetime import datetime

DATA_PATH = Path(__file__).parent 'exchange-rates.csv'

CURRENCY_LABELS = {
    'USD': 'US Dollar', 'GBP': 'British Pound', 'EUR': 'Euro',
    'JPY': 'Japanese Yen', 'CHF': 'Swiss Franc', 'AUD': 'Australian Dollar',
    'CAD': 'Canadian Dollar', 'SGD': 'Singapore Dollar', 'HKD': 'Hong Kong Dollar',
    'THB': 'Thai Baht', 'PHP': 'Philippine Peso', 'TWD': 'Taiwan Dollar',
    'KRW': 'South Korean Won', 'IDR': 'Indonesian Rupiah', 'SAR': 'Saudi Riyal',
    'SDR': 'IMF Special Drawing Rights', 'CNY': 'Chinese Yuan', 'BND': 'Brunei Dollar',
    'VND': 'Vietnamese Dong', 'KHR': 'Cambodian Riel', 'NZD': 'New Zealand Dollar',
    'MMK': 'Myanmar Kyat', 'INR': 'Indian Rupee', 'AED': 'UAE Dirham',
    'PKR': 'Pakistani Rupee', 'NPR': 'Nepalese Rupee', 'EGP': 'Egyptian Pound',
}

# Shown first in the converter dropdown -- Malaysia's top tourism source
# markets (per DOSM's Tourist Arrivals by Country data), ahead of the rest.
PRIORITY_CODES = ['SGD', 'IDR', 'THB', 'CNY', 'USD', 'EUR', 'GBP', 'JPY', 'AUD']

# Source columns quoted per 100 units, not per unit -- normalised below.
PER_100_COLUMNS = {
    'JPY100': 'JPY', 'HKD100': 'HKD', 'THB100': 'THB', 'PHP100': 'PHP',
    'TWD100': 'TWD', 'KRW100': 'KRW', 'IDR100': 'IDR', 'SAR100': 'SAR',
    'VND100': 'VND', 'KHR100': 'KHR', 'MMK100': 'MMK', 'INR100': 'INR',
    'AED100': 'AED', 'PKR100': 'PKR', 'NPR100': 'NPR',
}

# Plausible bounds for MYR-per-USD -- guards against a bad/malformed row.
SANITY_MIN_USD, SANITY_MAX_USD = 2.0, 8.0


def _parse_date(raw):
    for fmt in ('%d-%b-%y', '%d %b %Y', '%d-%b-%Y'):
        try:
            return datetime.strptime((raw or '').strip(), fmt)
        except ValueError:
            continue
    return None


def _load_latest_rates():
    """Read the CSV, keep only rows at a plausible MYR scale with a
    parseable date, and return (date_str, {code: myr_per_unit}) for the
    most recent valid row."""
    if not DATA_PATH.exists():
        return None, {}
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    best_row, best_date = None, None
    for row in rows:
        date = _parse_date(row.get(''))
        if date is None:
            continue
        try:
            usd = float(row.get('USD', ''))
        except (TypeError, ValueError):
            continue
        if not (SANITY_MIN_USD <= usd <= SANITY_MAX_USD):
            continue
        if best_date is None or date > best_date:
            best_date, best_row = date, row

    if best_row is None:
        return None, {}

    rates = {}
    for col, value in best_row.items():
        if not col:
            continue
        code = PER_100_COLUMNS.get(col, col)
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        rates[code] = num / 100 if col in PER_100_COLUMNS else num
    return best_row.get(''), rates


LATEST_DATE, LATEST_RATES = _load_latest_rates()

CURRENCIES = [
    {'code': code, 'label': CURRENCY_LABELS.get(code, code)}
    for code in PRIORITY_CODES
    if code in LATEST_RATES
] + [
    {'code': code, 'label': CURRENCY_LABELS.get(code, code)}
    for code in LATEST_RATES
    if code not in PRIORITY_CODES
]


def _load_usd_history():
    """Read the CSV and return every valid (date, MYR-per-USD) pair,
    sorted oldest to newest -- used by currency_score() to find where
    today's rate sits relative to its own recent history. Unlike
    _load_latest_rates(), this keeps the whole valid series, not just
    the newest row."""
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    history = []
    for row in rows:
        date = _parse_date(row.get(''))
        if date is None:
            continue
        try:
            usd = float(row.get('USD', ''))
        except (TypeError, ValueError):
            continue
        if not (SANITY_MIN_USD <= usd <= SANITY_MAX_USD):
            continue
        history.append((date, usd))
    history.sort(key=lambda item: item[0])
    return history


def currency_score():
    """Return (score 0-100, reason). This rating is built for DOMESTIC
    tourists choosing a destination WITHIN Malaysia -- a Malaysian
    visiting Selangor never converts currency, so exchange rates don't
    actually affect their decision. Defaults to full marks (100) for
    that reason, rather than fluctuating on a signal that's irrelevant
    to the actual user.

    The underlying "how weak is MYR right now" calculation is kept in
    _foreign_currency_score() below (unused by the live rating, but
    available if an international-tourist mode is ever added -- that
    version WOULD matter, since a foreign visitor's currency does
    affect their spending power)."""
    return 100, 'This rating targets domestic tourists, for whom exchange rates are not a relevant factor -- full marks by default.'


def _foreign_currency_score():
    """Not currently used by the rating (see currency_score() above).
    Kept for a possible future international-tourist mode: higher
    score = ringgit is WEAKER than usual (more attractive to foreign
    spenders), based on where the latest MYR/USD rate sits within its
    own available history.

    Uses MYR/USD specifically, not a source-market basket (SGD/IDR/THB/
    CNY, Malaysia's actual top markets per the Tourist Arrivals by
    Country data) -- a simplification; USD is the standard global
    reference and the cleanest column in the source file.

    Also does not vary by visit date even if reactivated -- exchange
    rates were deliberately kept out of the forecasting pipeline (see
    the module docstring above), so it would still be today's actual
    position, not a forecast for a future date."""
    history = _load_usd_history()
    if len(history) < 2:
        return 75, 'Not enough exchange-rate history to score; a neutral score is used.'

    values = [v for _, v in history]
    latest_date, latest_value = history[-1]
    lo, hi = min(values), max(values)
    # Higher MYR-per-USD = weaker ringgit = more attractive to foreign
    # spenders = higher score.
    score = 100 if hi == lo else round(100 * (latest_value - lo) / (hi - lo))

    if score >= 70:
        position = 'near a multi-year weak point for the ringgit'
    elif score <= 30:
        position = 'near a multi-year strong point for the ringgit'
    else:
        position = 'in the mid-range of its recent history'
    reason = (f'MYR/USD at {latest_value:.4f} as of {latest_date.strftime("%d %b %Y")} -- '
              f'{position} (range {lo:.4f}-{hi:.4f} over the available history).')
    return score, reason


def convert(amount, from_code, to_code):
    """Convert between MYR and a foreign currency (or foreign-to-foreign
    via MYR). Returns None if amount/currency/rate isn't available."""
    if amount is None:
        return None

    def to_myr(value, code):
        if code == 'MYR':
            return value
        rate = LATEST_RATES.get(code)
        return value * rate if rate else None

    def from_myr(value, code):
        if code == 'MYR':
            return value
        rate = LATEST_RATES.get(code)
        return value / rate if rate else None

    myr_value = to_myr(amount, from_code)
    if myr_value is None:
        return None
    return from_myr(myr_value, to_code)
