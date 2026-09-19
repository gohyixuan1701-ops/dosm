# --- AI Chatbox -----------------------------------------------------------
# NOT an LLM, NOT trained on anything, and no training step exists to run.
# This is keyword detection (does the message mention a state name? a
# topic?) that then calls the project's ALREADY-TRAINED functions and
# formats their real output into a sentence. Accurately described as "a
# rule-based assistant surfacing the project's real model outputs through a
# conversational interface" -- not "a local AI agent" (see project notes).

from datetime import datetime

import crime_forecast
import gdp_forecast
import cpi_score
import tourism_forecast
import population_score
import accommodation_data
import seasonal

# The 16 canonical state names used throughout this app (STATE_DATA's keys
# in app.py). Every scoring function expects one of these -- they apply
# their own NAME_ALIASES internally to reach the DOSM-style name.
ALL_STATES = ['Johor', 'Kedah', 'Kelantan', 'Malacca', 'Negeri Sembilan', 'Pahang',
              'Penang', 'Perak', 'Perlis', 'Selangor', 'Terengganu', 'Sabah',
              'Sarawak', 'Kuala Lumpur', 'Labuan', 'Putrajaya']

# Extra ways a user might type a state name in chat, beyond the canonical
# spelling itself (which is always checked too).
STATE_ALIASES = {
    'kl': 'Kuala Lumpur',
    'melaka': 'Malacca',
    'pulau pinang': 'Penang',
    'n sembilan': 'Negeri Sembilan',
    'n. sembilan': 'Negeri Sembilan',
}

TOPIC_KEYWORDS = {
    'safety': ['safety', 'safe', 'crime', 'danger', 'dangerous'],
    'gdp': ['gdp', 'economy', 'economic'],
    'affordability': ['afford', 'cost', 'expensive', 'cheap', 'price', 'cpi', 'cost of living'],
    'tourism': ['tourist', 'tourism', 'visitor', 'popular', 'busy', 'crowd'],
    'population': ['population', 'people', 'residents', 'how many live'],
    'accommodation': ['hotel', 'accommodation', 'room', 'stay', 'lodging'],
    'seasonal': ['monsoon', 'rain', 'flood', 'weather', 'season'],
}


def _slugify(state_name):
    return state_name.lower().replace(' ', '-')


def _detect_state(text):
    for alias, canonical in STATE_ALIASES.items():
        if alias in text:
            return canonical
    for state in ALL_STATES:
        if state.lower() in text:
            return state
    return None


def _detect_topics(text):
    return [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in text for w in words)]


def _answer_for(state_name, topic, year, month):
    slug = _slugify(state_name)

    if topic == 'safety':
        score, reason = crime_forecast.district_safety_score(state_name, None, year)
        return f"{state_name}'s safety score is {score}/100. {reason}"

    if topic == 'gdp':
        value = gdp_forecast.get_state_gdp(state_name, year)
        if value is None:
            return f"I don't have GDP data for {state_name} in {year}."
        return f"{state_name}'s GDP is forecast at roughly RM {value:,.0f} million for {year}."

    if topic == 'affordability':
        score, reason = cpi_score.affordability_score(state_name, year)
        return f"{state_name}'s affordability score is {score}/100. {reason}"

    if topic == 'tourism':
        visitors = tourism_forecast.get_state_visitors(state_name, year)
        if visitors is None:
            return f"I don't have a tourism forecast for {state_name} in {year}."
        return f"{state_name} is forecast to get around {visitors:,.0f} thousand visitors in {year}."

    if topic == 'population':
        pop = population_score.predict_state_population(state_name, year)
        if pop is None:
            return f"I don't have a population estimate for {state_name} in {year}."
        return f"{state_name}'s population is estimated at roughly {pop:,.0f} thousand in {year}."

    if topic == 'accommodation':
        acc = accommodation_data.get_state_accommodation(state_name)
        if not acc:
            return f"I don't have accommodation data for {state_name}."
        return (f"{state_name} has about {acc['rooms']:,} rooms across {acc['establishments']:,} "
                f"establishments (from the {acc['year']} Economic Census -- this is a single-year "
                f"snapshot, not a yearly trend).")

    if topic == 'seasonal':
        score, reason = seasonal.seasonal_score(slug, month)
        return f"{state_name}'s current seasonal weather score is {score}/100. {reason}"

    return None


def _state_summary(state_name, year, month):
    slug = _slugify(state_name)
    safety, _ = crime_forecast.district_safety_score(state_name, None, year)
    gdp = gdp_forecast.get_state_gdp(state_name, year)
    cpi = cpi_score.get_state_cpi(state_name, year)
    visitors = tourism_forecast.get_state_visitors(state_name, year)
    pop = population_score.predict_state_population(state_name, year)
    acc = accommodation_data.get_state_accommodation(state_name)
    weather, _ = seasonal.seasonal_score(slug, month)

    facts = [f"Safety {safety}/100"]
    if gdp is not None:
        facts.append(f"GDP ~RM {gdp:,.0f}M")
    if cpi is not None:
        facts.append(f"CPI index ~{cpi:.1f}")
    if visitors is not None:
        facts.append(f"~{visitors:,.0f}K visitors forecast")
    if pop is not None:
        facts.append(f"population ~{pop:,.0f}K")
    if acc:
        facts.append(f"{acc['rooms']:,} hotel rooms")
    facts.append(f"seasonal weather score {weather}/100")
    return (f"Here's what I have on {state_name} for {year}: " + ', '.join(facts) +
            ". Ask me about one of these specifically for more detail, or use the Trip Rating Generator above for the full 8-factor rating.")


GREETINGS = ('hello', 'hi', 'hey')

CURRENCY_KEYWORDS = ['currency', 'exchange', 'convert', 'myr', 'ringgit']
RATING_KEYWORDS = ['rating', 'best time', 'when should', 'recommend']

DEFAULT_REPLY = (
    "Ask me about a state and a topic -- e.g. \"is Selangor safe?\", \"GDP of Sabah\", "
    "\"how expensive is Penang\", or just a state name like \"Kelantan\" for a full summary. "
    "I can also point you to the Currency Converter or Trip Rating Generator above."
)


def reply_to(message):
    text = (message or '').lower()
    now = datetime.now()

    if any(g in text for g in GREETINGS) and len(text.split()) <= 3:
        return "Hi! Ask me about a state (e.g. \"is Selangor safe?\", \"GDP of Sabah\") or the best time to visit."

    if any(k in text for k in CURRENCY_KEYWORDS):
        return ("You can convert between MYR and 27 other currencies using the Currency "
                "Converter on this page -- rates are Bank Negara Malaysia's real daily mid rates.")

    if any(k in text for k in RATING_KEYWORDS):
        return ("Pick a destination and date in the Trip Rating Generator at the top of the "
                "page -- it combines all 8 real factors (seasonal, safety, cost, tourism, "
                "accommodation, population, GDP, currency) into one score.")

    state_name = _detect_state(text)
    topics = _detect_topics(text)

    if state_name and topics:
        answers = [_answer_for(state_name, t, now.year, now.month) for t in topics]
        return ' '.join(a for a in answers if a)

    if state_name:
        return _state_summary(state_name, now.year, now.month)

    if topics:
        return (f"I can tell you about that for a specific state -- which one? "
                f"(e.g. \"{topics[0]} in Selangor\")")

    return DEFAULT_REPLY