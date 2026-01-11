# Diabetes Prediction (Logistic Regression) — Streamlit App

This app loads a trained Logistic Regression pipeline and predicts the probability of diabetes.

## Files
- `app.py` : Streamlit application
- `logreg_pipeline.joblib` : trained sklearn Pipeline (imputer + scaler + logistic regression)
- `features.json` : feature order used by the model
- `requirements.txt` : dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Create a GitHub repo and upload these files.
2. Go to Streamlit Community Cloud → "New app"
3. Select your repo and `app.py`
4. Deploy.

