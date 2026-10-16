# Pakistan Agri-Market Price Predictor & Advisor

A Streamlit-based data analytics and agricultural advisory project exploring historical producer prices for crops in Pakistan. It combines price visualizations and basic machine-learning forecasts with a rule-based crop advisor that can answer selected questions in English and Roman Urdu.

> **Project status:** Academic prototype. Forecasts and agricultural recommendations are exploratory and should not be treated as official market quotes, financial advice, or professional agronomy guidance.

## Features

- **Historical price analysis:** Loads the included producer-price CSV and filters it to crop producer prices.
- **Price conversion:** Converts recorded prices from LCU/tonne to an estimated price per maund using the project's 25 kg-per-maund calculation.
- **Interactive charts:** Uses Plotly to visualize historical prices and model trends.
- **Machine-learning experiments:** Uses Linear Regression and Decision Tree Regression for price-trend exploration.
- **Crop advisory chat:** A Streamlit chat interface recognizes common intents such as price, storage, sowing season, harvest, irrigation, fertilizer, disease/pests, and subsidy.
- **English and Roman Urdu prompts:** Includes keyword-based recognition for selected Roman Urdu terms.
- **Rule-based knowledge:** Uses PySwip/Prolog when available for crop-specific sowing, harvesting, water, fertilizer, storage, and pest guidance, with fallback behavior when Prolog is unavailable.
- **Dashboard UI:** Streamlit metrics, tabs, custom CSS, and optional Lottie animation.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Application and data-processing logic |
| Streamlit | Interactive web app and chat interface |
| pandas, NumPy | Dataset processing and numerical operations |
| scikit-learn | Linear Regression, Decision Tree Regression, and evaluation metrics |
| Plotly | Interactive data visualization |
| NLTK | Tokenization for keyword-based intent detection |
| PySwip + SWI-Prolog | Optional rule-based agricultural knowledge |
| Requests + streamlit-lottie | Optional remote animation |
| CSV | Historical producer-price dataset |

## Repository Structure

```text
.
├── app.py
├── producer-prices_pak.csv
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### 1. Clone the repository

```bash
git clone https://github.com/sajjadabdullah90/Pakistan-Agri-Market-Price-Predictor-Advisor-Machine-Learning-and-AI-Project-.git
cd Pakistan-Agri-Market-Price-Predictor-Advisor-Machine-Learning-and-AI-Project-
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn nltk requests streamlit-lottie
```

For Prolog-backed recommendations, install SWI-Prolog and the Python package:

```bash
pip install pyswip
```

Some NLTK tokenizer data may be downloaded by the application on its first run. An internet connection is required for that download and the optional remote Lottie animation.

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will print the local URL in the terminal, commonly `http://localhost:8501`.

## Data and Limitations

- The application reads `producer-prices_pak.csv` from the project directory.
- Predictions depend on the available historical records and the simple models used; they are not guaranteed future prices.
- Advisory responses use keyword matching and a manually defined rule base, not a general-purpose language model.
- Dataset coverage, units, crop names, seasonality, and regional differences should be validated before using results for real decisions.
- The current prototype does not establish a live connection to an official market-price feed.

## Deployment

### Option A: 1-Click Vercel Deployment (Static WebAssembly)
This repository includes a standalone `index.html` powered by Stlite (Streamlit on WebAssembly). It runs completely client-side with Pyodide and requires zero backend server configuration.

1. Go to [Vercel](https://vercel.com) and click **Add New Project**.
2. Import this repository.
3. Framework Preset: **Other** / Static HTML.
4. Click **Deploy**. Vercel will serve `index.html` immediately.

### Option B: Streamlit Community Cloud (Official Python Engine)
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account and select this repository.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and run the app with live WebSocket connectivity.

## Future Improvements

- Add dataset validation, missing-value reporting, and clearer source/provenance information.
- Evaluate forecasting models using time-aware train/test splits and appropriate metrics.
- Improve Roman Urdu intent handling and add test coverage.
- Separate agricultural rules from UI code and validate recommendations with domain experts.

## Author

**Syed Abdullah**  
GitHub: [@sajjadabdullah90](https://github.com/sajjadabdullah90)
