from datetime import datetime
from seasonal import seasonal_score
from crime_forecast import district_safety_score
from cpi_score import affordability_score
from tourism_forecast import tourism_trend_score
from accommodation_data import accommodation_score
from population_score import population_score
from gdp_forecast import gdp_score
from currency import currency_score

# Weights -- originally 0.20/0.20/0.15/0.15/0.10/0.10/0.05/0.03/0.02
# across 9 factors including transport (X), which was dropped for
# having no underlying data at all. Remaining 8 weights were scaled up
# proportionally so they still sum to 1.00, preserving the original
# relative priority order rather than dumping X's 0.10 onto one factor
# arbitrarily.
WEIGHTS = {
    'monsoon': 0.22,        # M -- was 0.22
    'safety': 0.22,         # S -- was 0.22
    'affordability': 0.17,  # C -- was 0.17
    'tourism': 0.17,        # T -- was 0.17
    'accommodation': 0.11,  # A -- was 0.11
    'population': 0.06,     # P -- was 0.06
    'gdp': 0.03,            # G -- was 0.03 
    'currency': 0.02,       # U -- was 0.02 
}
assert abs(sum(WEIGHTS.values()) - 1.00) < 1e-9, "WEIGHTS must sum to 1.00"


def verdict_for(score):
    if score >= 70:
        return 'Excellent time to visit', '#198c72'
    if score >= 55:
        return 'Good time to visit', '#c98a1c'
    if score >= 40:
        return 'Fair — check conditions', '#d9740a'
    return 'Not recommended right now', '#c93a3a'


def compute_rating(slug, date_str, state_data, district=None, visitor_type='domestic', currency_code='USD'):
    state_name = next((n for n, d in state_data.items() if d['slug'] == slug), None)
    if not state_name:
        return None
    try:
        visit_date = datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        visit_date = datetime.today()
    year = visit_date.year

    m_score, m_reason = seasonal_score(slug, visit_date.month)
    # district is optional -- district_safety_score() falls back to the
    # state-level safety_score() automatically if district is None, not
    # found, or has no forecast data (e.g. Labuan/Putrajaya).
    s_score, s_reason = district_safety_score(state_name, district, year)
    c_score, c_reason = affordability_score(state_name, year)
    t_score, t_reason = tourism_trend_score(state_name, year)
    a_score, a_reason = accommodation_score(state_name)
    p_score, p_reason = population_score(state_name, year)
    g_score, g_reason = gdp_score(state_name, year)
    # currency_score() branches on visitor_type: flat 100 for domestic
    # (exchange rates don't affect a Malaysian visiting Malaysia), the
    # real weak/strong-ringgit calculation (against the tourist's own
    # currency_code) for foreign, where it genuinely matters. Still
    # doesn't vary by visit date either way -- see currency.py's own
    # docstring.
    u_score, u_reason = currency_score(visitor_type, currency_code)

    categories = [
        {'key': 'monsoon', 'label': 'Seasonal / weather risk', 'score': m_score, 'reason': m_reason},
        {'key': 'safety', 'label': 'Safety / crime outlook', 'score': s_score, 'reason': s_reason},
        {'key': 'affordability', 'label': 'Cost affordability', 'score': c_score, 'reason': c_reason},
        {'key': 'tourism', 'label': 'Tourism demand trend', 'score': t_score, 'reason': t_reason},
        {'key': 'accommodation', 'label': 'Accommodation capacity', 'score': a_score, 'reason': a_reason},
        {'key': 'population', 'label': 'Population / infrastructure', 'score': p_score, 'reason': p_reason},
        {'key': 'gdp', 'label': 'Economic strength (GDP)', 'score': g_score, 'reason': g_reason},
        {'key': 'currency', 'label': 'Currency (MYR weakness)', 'score': u_score, 'reason': u_reason},
    ]

    overall = round(sum(WEIGHTS[c['key']] * c['score'] for c in categories))
    verdict, verdict_colour = verdict_for(overall)

    return {
        'state_name': state_name,
        'district_name': district,
        'visitor_type': visitor_type,
        'state': state_data[state_name],
        'date_display': visit_date.strftime('%d %B %Y'),
        'overall': overall,
        'verdict': verdict,
        'verdict_colour': verdict_colour,
        'categories': categories,
    }
