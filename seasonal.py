# --- Seasonal / Monsoon Risk -------------------------------------------
# ASSUMPTION-BASED, not trained on data. State-level quarterly tourism
# data isn't publicly available from DOSM (only annual-by-state and
# quarterly-national are published separately), so this factor is built
# from MetMalaysia's officially published Northeast Monsoon season
# instead of a statistical model.
#
# Source: MetMalaysia's annual Northeast Monsoon announcements (e.g.
# Bernama, 10 Nov 2025) -- season runs mid-November to March, with
# December-January as peak risk and March as the withdrawal phase.
# Primary impact zones: Kelantan, Terengganu, Pahang (heaviest -- island
# and coastal access often closed). Secondary: Sabah, Sarawak (rain, not
# closures).
#
# DISCLOSE THIS AS AN ASSUMPTION IN THE REPORT -- it is not derived from
# a trained model or state-level statistics. See project notes.

MONSOON_HEAVY = {'pahang', 'terengganu', 'kelantan'}
MONSOON_LIGHT = {'sabah', 'sarawak'}
MONSOON_MONTHS = {11, 12, 1, 2, 3}
MONSOON_SHOULDER_MONTHS = {10}
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_chart_series(slug, visit_month=None):
    """Return {'months': [...12 names...], 'values': [0-100 per month],
    'visit_month': N or None} for the "Show more detail" calendar
    chart. Uses the ACTUAL seasonal_score() value for every month, not
    a simplified binary version -- this means the chart is a true
    picture of what's driving the Seasonal/weather risk row above, and
    every state gets a meaningful gradient (even states with no
    monsoon classification show a flat line, honestly reflecting "low
    risk all year" rather than being left blank)."""
    values = [seasonal_score(slug, m)[0] for m in range(1, 13)]
    return {'months': MONTH_NAMES, 'values': values, 'visit_month': visit_month}


def seasonal_score(slug, month):
    """Return (score 0-100, reason) for a state/month combination."""
    if slug in MONSOON_HEAVY:
        if month in MONSOON_MONTHS:
            return 35, 'Peak monsoon season — island/coastal access is often limited or closed.'
        if month in MONSOON_SHOULDER_MONTHS:
            return 70, 'Shoulder season — monsoon risk is easing but weather can still be unpredictable.'
        return 95, 'Outside monsoon season — good weather reliability expected.'
    if slug in MONSOON_LIGHT:
        if month in MONSOON_MONTHS:
            return 60, 'Wetter season in East Malaysia — plan for rain, but no major closures expected.'
        return 90, 'Favourable weather window.'
    return 95, 'No monsoon-risk months identified for this state — no seasonal weather concern year-round.'