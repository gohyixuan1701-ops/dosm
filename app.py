from flask import Flask, render_template, abort, request
from pathlib import Path
 
from rating import compute_rating
from crime_forecast import list_districts
import currency
import chatbot
 
app = Flask(__name__)
 
STATE_DATA = {
    'Johor': {'code':'MY01','colour':'#F59E0B','visitors':'4.82M','growth':'+8.4%','highlights':['Desaru Coast','Johor Bahru','Mersing'], 'slug':'johor'},
    'Kedah': {'code':'MY02','colour':'#10B981','visitors':'3.15M','growth':'+5.7%','highlights':['Langkawi','Alor Setar','Kilim Geoforest Park'], 'slug':'kedah'},
    'Kelantan': {'code':'MY03','colour':'#8B5CF6','visitors':'1.94M','growth':'+3.2%','highlights':['Kota Bharu','Siti Khadijah Market','Pantai Cahaya Bulan'], 'slug':'kelantan'},
    'Malacca': {'code':'MY04','colour':'#EF4444','visitors':'2.76M','growth':'+6.1%','highlights':['Jonker Street','A Famosa','Melaka River'], 'slug':'malacca'},
    'Negeri Sembilan': {'code':'MY05','colour':'#06B6D4','visitors':'1.83M','growth':'+4.9%','highlights':['Port Dickson','Seremban','Kuala Pilah'], 'slug':'negeri-sembilan'},
    'Pahang': {'code':'MY06','colour':'#F97316','visitors':'3.68M','growth':'+7.3%','highlights':['Cameron Highlands','Taman Negara','Kuantan'], 'slug':'pahang'},
    'Penang': {'code':'MY07','colour':'#EC4899','visitors':'5.46M','growth':'+9.8%','highlights':['George Town','Batu Ferringhi','Balik Pulau'], 'slug':'penang'},
    'Perak': {'code':'MY08','colour':'#84CC16','visitors':'4.01M','growth':'+6.5%','highlights':['Ipoh','Pangkor Island','Royal Belum'], 'slug':'perak'},
    'Perlis': {'code':'MY09','colour':'#14B8A6','visitors':'0.72M','growth':'+2.8%','highlights':['Kuala Perlis','Wang Kelian','Gua Kelam'], 'slug':'perlis'},
    'Selangor': {'code':'MY10','colour':'#6366F1','visitors':'7.12M','growth':'+10.4%','highlights':['Petaling Jaya','Shah Alam','Sepang'], 'slug':'selangor'},
    'Terengganu': {'code':'MY11','colour':'#0EA5E9','visitors':'2.87M','growth':'+5.2%','highlights':['Perhentian Islands','Redang Island','Kuala Terengganu'], 'slug':'terengganu'},
    'Sabah': {'code':'MY12','colour':'#22C55E','visitors':'3.92M','growth':'+11.2%','highlights':['Kota Kinabalu','Mount Kinabalu','Semporna'], 'slug':'sabah'},
    'Sarawak': {'code':'MY13','colour':'#EAB308','visitors':'3.51M','growth':'+7.9%','highlights':['Kuching','Mulu','Bako National Park'], 'slug':'sarawak'},
    'Kuala Lumpur': {'code':'MY14','colour':'#7C3AED','visitors':'6.83M','growth':'+12.6%','highlights':['KLCC','Bukit Bintang','Batu Caves'], 'slug':'kuala-lumpur'},
    'Labuan': {'code':'MY15','colour':'#FB7185','visitors':'0.34M','growth':'+1.9%','highlights':['Patau-Patau','Labuan Marine Museum','Financial Park'], 'slug':'labuan'},
    'Putrajaya': {'code':'MY16','colour':'#475569','visitors':'1.18M','growth':'+6.8%','highlights':['Putra Mosque','Persiaran Perdana','Taman Botani'], 'slug':'putrajaya'},
}
 
SVG_PATH = Path(app.root_path, 'static', 'malaysia.svg').read_text(encoding='utf-8')
 
# Turn the uploaded SVG into an inline, dashboard-friendly map.
# The original file's state paths already carry a `name` attribute.
SVG_PATH = SVG_PATH.replace('<svg ', '<svg class="malaysia-map" preserveAspectRatio="xMidYMid meet" ')
 
# The raw SVG uses the official/Malay names for two states, not the
# English display names used elsewhere in this app (STATE_DATA, the
# dropdown, etc.) -- a direct name="{state}" match silently fails for
# just these two, leaving them unclickable on the map with no error.
SVG_NAME_ALIASES = {'Malacca': 'Melaka', 'Penang': 'Pulau Pinang'}
 
for state, data in STATE_DATA.items():
    svg_name = SVG_NAME_ALIASES.get(state, state)
    if f'name="{svg_name}"' in SVG_PATH:
        SVG_PATH = SVG_PATH.replace(
            f'name="{svg_name}"',
            f'name="{svg_name}" data-state="{data["slug"]}"'
        )
 
# Start with a neutral fill; CSS assigns the per-state palette via data-state.
SVG_PATH = SVG_PATH.replace('fill="#6f9c76"', 'fill="currentColor"')
 
 
@app.context_processor
def inject_globals():
    return {
        'states': STATE_DATA,
        'malaysia_svg': SVG_PATH,
        'currencies': currency.CURRENCIES,
        'rates_date': currency.LATEST_DATE,
    }
 
 
@app.get('/')
def index():
    return render_template('index.html', page='dashboard')
 
 
@app.get('/states/<slug>')
def state_detail(slug):
    state = next((name for name, d in STATE_DATA.items() if d['slug'] == slug), None)
    if not state:
        abort(404)
    return render_template('_state_detail.html', state_name=state, state=STATE_DATA[state])
 
 
@app.get('/insights')
def insights():
    return render_template('_insights.html')
 
 
@app.get('/districts/<slug>')
def districts(slug):
    state = next((name for name, d in STATE_DATA.items() if d['slug'] == slug), None)
    if not state:
        return render_template('_district_options.html', districts=[])
    try:
        return render_template('_district_options.html', districts=list_districts(state))
    except Exception:
        app.logger.exception(f'list_districts failed for state={state!r}')
        return render_template('_district_options.html', districts=[])
 
 
@app.post('/rate')
def rate():
    slug = request.form.get('destination', '')
    date_str = request.form.get('visit_date', '')
    district = request.form.get('district', '').strip() or None
    result = compute_rating(slug, date_str, STATE_DATA, district=district)
    if not result:
        return render_template('_rating_result.html', error=True)
    return render_template('_rating_result.html', r=result, error=False)
 
 
@app.post('/convert')
def convert():
    try:
        amount = float(request.form.get('amount', ''))
    except (TypeError, ValueError):
        amount = None
    from_code = request.form.get('from_currency', 'MYR')
    to_code = request.form.get('to_currency', 'USD')
 
    result = currency.convert(amount, from_code, to_code) if amount is not None else None
    return render_template(
        '_currency_result.html',
        amount=amount, from_code=from_code, to_code=to_code,
        result=result, rates_date=currency.LATEST_DATE,
    )
 
 
@app.post('/chat')
def chat():
    message = request.form.get('message', '').strip()
    reply = chatbot.reply_to(message) if message else None
    return render_template('_chat_message.html', message=message, reply=reply)
 
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
 