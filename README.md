# EDA Project B Starter — Time-Series Forecasting

Student: Maha  
Student ID: PG12S2540565

This repository contains a starter Streamlit app for Mini Project B. The included dataset sample was generated from the uploaded Oman Global Solar Atlas monthly PVOUT GeoTIFF files by summarising each monthly raster into Oman-wide PV output statistics.

## Files

- `app.py` — one-file Streamlit app
- `requirements.txt` — Python dependencies for Streamlit Community Cloud
- `data/dataset_sample.csv` — cleaned/sliced dataset sample

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload these files exactly:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Go to Streamlit Community Cloud.
4. Choose **New app**.
5. Connect your GitHub repo.
6. Set the main file path to `app.py`.
7. Deploy.

## OpenRouter API key for AI grading

The app reads the OpenRouter API key from:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field inside the app

Do not hardcode API keys in the repository.

## What to submit

Submit the following:

- Streamlit deployed app URL
- GitHub repository URL
- `submission.json` exported from the app
- `project_card.md` exported from the app
- Screenshots required by your instructor
