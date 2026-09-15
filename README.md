# MyTourism Analytics — HTMX Dashboard Prototype

A dashboard-first prototype for a DOSM tourism website competition.

## Stack
- Flask + Jinja templates
- HTMX for partial page updates
- Uploaded Malaysia SVG used as the interactive map
- Plain CSS for the visual design

## Run locally
```bash
# Native Windows TensorFlow requires official 64-bit Python 3.10.
# Do not use Python bundled with Inkscape.
py -3.10 -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

To generate the crime forecast, run this from the project directory:

```bash
python "forecast crime.py"
```

The script reads `crime_district.csv` from the same directory as the script and
writes `crime_forecast_2024.csv` there.

## What is already interactive
- Clicking a state card loads its detail panel with HTMX without a full page reload.
- “Generate prototype insight” loads an AI/ML placeholder insight through HTMX.
- The map paths are coloured by state and respond to hover.

## Replace later
1. Swap illustrative values in `STATE_DATA` with approved DOSM data.
2. Add your own state tourism images inside the state detail panel.
3. Add a real ML service/API to `/insights`.
4. Choose a final hero/background image after the competition team agrees on the visual direction.

## SVG attribution
The supplied SVG contains its original SimpleMaps licensing/attribution notice. Keep that notice when distributing the SVG unless your competition rules allow another treatment.
