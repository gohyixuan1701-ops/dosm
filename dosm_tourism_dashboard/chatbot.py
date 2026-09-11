# --- AI Chatbox ---------------------------------------------------------
# PLACEHOLDER LOGIC ONLY. This is simple keyword matching so the chat UI
# is demoable now. Swap reply_to() for a real LLM API call later -- the
# /chat route and chat template don't need to change, only this function.

RULES = [
    (('monsoon', 'rain', 'flood', 'weather', 'season'),
     "Monsoon season in Malaysia runs mid-November to March, hitting Kelantan, "
     "Terengganu and Pahang hardest (island access often closes). Try the Trip "
     "Rating Generator above with your date and destination for a specific score."),
    (('currency', 'exchange', 'convert', 'myr', 'rate'),
     "You can convert between MYR and major currencies using the Currency "
     "Converter on this page — rates are from Bank Negara Malaysia."),
    (('rating', 'score', 'best time', 'when should'),
     "Pick a destination and date in the Trip Rating Generator at the top of "
     "the page — it scores seasonal risk and tourism demand for that state."),
    (('hello', 'hi', 'hey'),
     "Hi! Ask me about monsoon season, currency conversion, or the best time "
     "to visit a state."),
]

DEFAULT_REPLY = (
    "I'm a placeholder assistant for now — ask me about monsoon season, "
    "currency conversion, or trip ratings. A trained model will replace me later."
)


def reply_to(message):
    text = (message or '').lower()
    for keywords, reply in RULES:
        if any(k in text for k in keywords):
            return reply
    return DEFAULT_REPLY
