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

DATA_PATH = Path(__file__).parent / 'exchange-rates.csv'

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
