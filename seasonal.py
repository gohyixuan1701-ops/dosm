
MONSOON_HEAVY = {'pahang', 'terengganu', 'kelantan'}
MONSOON_LIGHT = {'sabah', 'sarawak'}
MONSOON_MONTHS = {11, 12, 1, 2, 3}
MONSOON_SHOULDER_MONTHS = {10}


def seasonal_score(slug, month):
    """Return (score 0-100, reason) for a state/month combination."""
    if slug in MONSOON_HEAVY:
        if month in MONSOON_MONTHS:
            return 25, 'Peak monsoon season — island/coastal access is often limited or closed.'
        if month in MONSOON_SHOULDER_MONTHS:
            return 65, 'Shoulder season — monsoon risk is easing but weather can still be unpredictable.'
        return 95, 'Outside monsoon season — good weather reliability expected.'
    if slug in MONSOON_LIGHT:
        if month in MONSOON_MONTHS:
            return 60, 'Wetter season in East Malaysia — plan for rain, but no major closures expected.'
        return 90, 'Favourable weather window.'
    return 88, 'No major seasonal weather risk identified for this state.'
