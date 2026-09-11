from datetime import datetime
from seasonal import seasonal_score

# PLACEHOLDER LOGIC ONLY for demand_score/compute_rating. Swap for the
# trained model's prediction once the ML pipeline is ready -- app.py's
# /rate route and the result template don't need to change.


def demand_score(slug, state_data):
    name = next((n for n, d in state_data.items() if d['slug'] == slug), None)
    data = state_data.get(name)
    if not data:
        return 50
    try:
        growth = float(data['growth'].replace('%', '').replace('+', ''))
    except (ValueError, AttributeError):
        growth = 0
    return max(0, min(100, round(50 + growth * 3)))


def verdict_for(score):
    if score >= 80:
        return 'Excellent time to visit', '#198c72'
    if score >= 60:
        return 'Good time to visit', '#c98a1c'
    if score >= 40:
        return 'Fair — check conditions', '#d9740a'
    return 'Not recommended right now', '#c93a3a'


def compute_rating(slug, date_str, state_data):
    state_name = next((n for n, d in state_data.items() if d['slug'] == slug), None)
    if not state_name:
        return None
    try:
        visit_date = datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        visit_date = datetime.today()

    s_score, s_reason = seasonal_score(slug, visit_date.month)
    d_score = demand_score(slug, state_data)
    overall = round(0.5 * s_score + 0.5 * d_score)
    verdict, verdict_colour = verdict_for(overall)

    return {
        'state_name': state_name,
        'state': state_data[state_name],
        'date_display': visit_date.strftime('%d %B %Y'),
        'overall': overall,
        'verdict': verdict,
        'verdict_colour': verdict_colour,
        'seasonal_score': s_score,
        'seasonal_reason': s_reason,
        'demand_score': d_score,
    }
