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
