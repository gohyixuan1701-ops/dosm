# MyTourism — Malaysia Tourism Intelligence Dashboard

DOSM Datathon 2026 submission. A trip-rating dashboard combining 8 real,
data-driven factors (seasonal risk, safety, cost of living, tourism demand,
accommodation capacity, population, GDP, and currency) using official
Malaysian government data sources, real trained ML/statistical models, and
a live interactive interface.

## 1. Requirements

- Python 3.10 specifically (TensorFlow 2.10.1 is not compatible with newer
  Python versions -- see installation steps below)
- A terminal (Command Prompt or PowerShell on Windows) -- no code editor or
  IDE of any kind is required to run this project. Everything below uses
  only `python` and `pip` commands typed directly into a terminal window.
- A short project root path (see the Windows long-path note at the end of
  this section)

### 1a. Installing Python (if not already installed)

1. Go to **https://www.python.org/downloads/** and download **Python 3.10**
   specifically -- not the newest version offered by default, since
   TensorFlow 2.10.1 does not support Python 3.11+. Scroll to "Looking for a
   specific release?" on that page, or go directly to
   **https://www.python.org/downloads/release/python-31011/** and download
   the "Windows installer (64-bit)".
2. Run the installer. **On the first screen, tick the checkbox "Add
   python.exe to PATH"** before clicking Install -- this is the single most
   common setup mistake; without it, typing `python` in a terminal won't
   work anywhere outside the installer's own folder.
3. Once installed, open a **new** Command Prompt or PowerShell window (must
   be opened after installing, not one left open from before) and confirm:
   ```powershell
   python --version
   ```
   This should print `Python 3.10.x`. If it instead says "python is not
   recognized," Python was not added to PATH correctly -- rerun the
   installer, choose "Modify," and enable that checkbox.

### 1b. Windows long-path note (relevant when installing TensorFlow below)

TensorFlow's installer can fail with a path-length error if the project
folder is nested many levels deep (e.g. inside
`Downloads\extracted\some-folder\another-folder\...`). Two ways to avoid
this -- either is fine, only one is needed:

- **Simplest: move the project to a short path** before proceeding, e.g.
  `C:\DOSM\` directly, rather than leaving it inside a deep Downloads
  subfolder structure.
- **Or enable long path support in Windows** (Windows 10/11): open
  **Group Policy Editor** (`gpedit.msc`) -> Computer Configuration ->
  Administrative Templates -> System -> Filesystem -> enable "Enable Win32
  long paths." Requires a restart to take effect. (`gpedit.msc` isn't
  available on Windows Home editions -- use the short-path option instead
  if so.)

## 2. Environment Setup

Open a terminal in the project's root folder (the folder containing
`app.py`), then:

```powershell
# create a virtual environment named tf_env
python -m venv tf_env

# activate it (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
tf_env\Scripts\activate

# if using Command Prompt instead of PowerShell, activate with:
# tf_env\Scripts\activate.bat

# install all dependencies
pip install -r requirements.txt
```

You should see `(tf_env)` appear at the start of your terminal prompt once
activated -- this confirms packages will install into the isolated
environment rather than system-wide.

`requirements.txt` should include at minimum: `flask`, `tensorflow==2.10.1`,
`pandas`, `numpy`, `scikit-learn`. This single `pip install -r
requirements.txt` command installs TensorFlow along with everything else --
no separate TensorFlow installation step is needed once this completes
successfully. This step can take several minutes and will download a few
hundred MB, since TensorFlow is a large package.

**To verify TensorFlow installed correctly**, with `tf_env` still active:
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```
This should print `2.10.1` with no errors. A "path too long" or similar
install failure here means the Section 1b note above needs addressing
before retrying `pip install -r requirements.txt`.

## 3. Project Structure

```
app.py                     Flask routes, STATE_DATA, map/SVG setup
rating.py                  Composite rating engine (weights, verdict thresholds)
seasonal.py                Rule-based monsoon/seasonal scoring
crime_forecast.py          Crime/safety: LSTM vs SimpleRNN, district + state level
gdp_forecast.py             GDP: SimpleRNN vs LSTM, rank-based scoring
cpi_score.py                Affordability: linear regression
tourism_forecast.py        Tourism demand: SimpleRNN vs LSTM
population_score.py        Population: linear regression, rank-based scoring
accommodation_data.py      Accommodation: static lookup (no model)
currency.py                 Live currency lookup + domestic/foreign scoring
chatbox.py                  Rule-based chat assistant (not an LLM)

templates/
  index.html                 Main dashboard page
  _rating_result.html        Rating card partial (HTMX target)
  _district_options.html     District dropdown partial -- REQUIRED, app
                              will 500 on every district request if missing
  _state_detail.html         State Explorer detail panel partial
  _currency_result.html      Currency converter result partial
  _chat_message.html         Chat message partial

static/
  style.css
  malaysia.svg                Base map (state name attributes used for click targets)
  mascot-tiger.png            Optional -- chat widget falls back to an emoji if absent

data/
  exchange-rates.csv          BNM daily mid rates (currency.py's DATA_PATH expects
                               this INSIDE a data/ subfolder, not the project root)

(project root)
  crime_district.csv, crime_forecast.csv, crime_forecast_district.csv
  population_state.csv, population_district.csv,
    population_forecast.csv, population_forecast_district.csv
  gdp_state_real_supply.csv, gdp_forecast.csv
  cpi_2d_state.csv, cpi_forecast.csv
  tourism_visitors_by_state.csv, tourism_forecast.csv
  accommodation_by_state.csv
```

## 4. Running the Dashboard

```powershell
tf_env\Scripts\activate
python app.py
```

Open the printed URL (usually `http://127.0.0.1:5000`).

**Before first run**, make sure the three `*_forecast.csv` files
(crime, GDP, tourism) already exist -- see Section 5. If they're missing,
the app will not crash (graceful neutral-score fallbacks are built in),
but Safety, GDP, and Tourism will all show a flat, uninformative 75/100
instead of real forecasts.

## 5. (Re)training the Models

Three factors use trained RNN/LSTM models and need an explicit training
step. The rest (population, CPI, accommodation, currency, seasonal) compute
live or are static -- no training step needed for them.

```powershell
python crime_forecast.py      # writes crime_forecast.csv, crime_forecast_district.csv
python gdp_forecast.py        # writes gdp_forecast.csv
python tourism_forecast.py    # writes tourism_forecast.csv
python population_score.py    # writes population_forecast.csv, population_forecast_district.csv
```

Each training script prints a validation MAE comparison between SimpleRNN
and LSTM before saving, e.g.:

```
SimpleRNN: validation MAE = RM4,014M
LSTM: validation MAE = RM13,550M
Best on validation: SimpleRNN
```

## 6. Viewing Training Results in TensorBoard

Each of the three RNN/LSTM training scripts logs both models' training and
validation curves (`epoch_loss`, `epoch_mae`, and per-iteration evaluation
curves) to a `logs/` folder, viewable in TensorBoard.

**Clean slate first** (recommended before retraining, so old runs don't
clutter the charts):

```powershell
Remove-Item -Recurse -Force logs
```

**Then train and launch:**

```powershell
python crime_forecast.py
python gdp_forecast.py
python tourism_forecast.py

tensorboard --logdir=logs
```

Open `http://localhost:6006`. Under **Runs**, you should see six entries:

```
CRIME_SimpleRNN     CRIME_LSTM
GDP_SimpleRNN        GDP_LSTM
TOURISM_SimpleRNN   TOURISM_LSTM
```

Use the **SCALARS** tab and the regex filter box (e.g. type `CRIME` to
isolate just those two runs) to compare a specific factor's two
architectures side by side. A converged, healthy run shows a curve that
drops steeply then flattens; a widening gap between a run's `train` and
`validation` lines indicates overfitting.

> A "TensorFlow installation not found" warning in the TensorBoard UI is
> harmless -- the Scalars tab still works correctly.

## 7. Known Operational Notes

- **`templates/_district_options.html` is a hard dependency** for the
  `/districts/<slug>` route. If this file is ever deleted or missing, that
  route will fail; a safety fallback prevents this from crashing the whole
  page, but confirm this file exists before deployment.
- **`data/exchange-rates.csv` must sit inside a `data/` subfolder**, not
  the project root -- `currency.py`'s `DATA_PATH` expects that exact
  location.
- **`static/mascot-tiger.png` is optional.** If absent, the chat widget's
  floating button automatically falls back to a tiger emoji -- no code
  change needed either way.
- Two states (Penang, Malacca) have an internal name-alias mapping to the
  base map SVG's official Malay names (`Pulau Pinang`, `Melaka`) -- if the
  map ever silently stops registering clicks for just these two states,
  check `SVG_NAME_ALIASES` in `app.py` first.

## 8. Data Sources

Crime: OpenDOSM (`open.dosm.gov.my`), Royal Malaysia Police via DOSM.
Population, GDP, CPI: OpenDOSM state/district catalogues.
Tourism: DOSM Domestic Tourism Survey 2025.
Accommodation: DOSM Economic Census 2023.
Currency: Bank Negara Malaysia (`bnm.gov.my/exchange-rates`).
Seasonal/monsoon: MetMalaysia official Northeast Monsoon announcements.
