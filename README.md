# Fund Manager 13F Dashboard

Interactive Streamlit dashboard for visualizing 13F holdings across managers.

Run (in the devcontainer):

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Default inputs are in the app; you can enter manager names (e.g. "Berkshire Hathaway"), a target quarter like `2025Q3`, and a minimum conviction weight (e.g. `0.02` for 2%).

Notes:
- The app fetches 13F filings from SEC EDGAR and uses `yfinance` for market data and shares outstanding.
- Network access to the SEC and Yahoo endpoints is required.

OpenFIGI (optional, recommended)
- To improve ticker resolution the app can use OpenFIGI to map CUSIP → ticker. This is more reliable than matching company names.
- If you have an OpenFIGI API key, set it in your environment before running the app:

```bash
export OPENFIGI_APIKEY="your_openfigi_api_key_here"
```

The app will automatically attempt OpenFIGI mapping for rows that include a CUSIP and fall back to a Yahoo name search when OpenFIGI is not available.

Styling: dark/Bloomberg-like theme applied via `assets/style.css`.
# fin