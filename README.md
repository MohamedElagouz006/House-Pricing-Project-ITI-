# House Price Prediction — End-to-End ML Web App

Predicts the market price of a residential property from a handful of details
(location, carpet area, floor, bathrooms, furnishing, etc). Built as a
student project: a Jupyter notebook trains and exports a scikit-learn model,
a FastAPI backend serves it, and a React frontend collects input and shows
the estimate.

## Overview

```
User fills in property details (React form)
        │
        ▼
   POST /predict  (FastAPI)
        │
        ▼
  sklearn Pipeline (house_price.pkl)
  ColumnTransformer → RandomForestRegressor
        │
        ▼
   { "predicted_price": 8500000 }
        │
        ▼
   Result page shows ₹ 85.00 Lac
```

## Tech stack

| Layer      | Tech                                             |
|------------|---------------------------------------------------|
| Modeling   | pandas, scikit-learn, joblib, Jupyter              |
| Backend    | FastAPI, Pydantic v2, Uvicorn                      |
| Frontend   | React 18, TypeScript, Vite, React Router           |
| Dataset    | [House Price](https://www.kaggle.com/datasets/juhibhojani/house-price) (Kaggle, ~187k Indian property listings) |

## Project structure

```
house-price-app/
├── notebooks/
│   ├── house_price_model.ipynb   # cleaning, EDA, training, export
│   └── data/                     # dataset CSV goes here (gitignored)
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, CORS, lifespan model loading
│   │   ├── api/routes/prediction.py
│   │   ├── core/config.py
│   │   ├── schemas/prediction.py
│   │   └── services/
│   │       ├── preprocessing.py
│   │       └── inference.py
│   ├── models/                   # house_price.pkl + locations.json go here
│   ├── tests/test_prediction.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api/predictionClient.ts
    │   ├── components/PredictionForm.tsx
    │   ├── pages/HomePage.tsx | ResultPage.tsx | NotFoundPage.tsx
    │   ├── types/prediction.ts
    │   └── App.tsx
    └── .env.example
```

## Dataset

**House Price** by Juhi Bhojani —
https://www.kaggle.com/datasets/juhibhojani/house-price

```bash
pip install kaggle
# Kaggle → Settings → API → "Create New Token", save kaggle.json to
# C:\Users\<you>\.kaggle\ (Windows) or ~/.kaggle/ (macOS/Linux)
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

Always confirm the real column names with `df.columns` after downloading —
don't rely on this README.

## Running the notebook

```bash
cd notebooks
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook house_price_model.ipynb
```

Run all cells top to bottom. The last cells export `house_price.pkl` and
`locations.json` — copy both into `backend/models/`.

```bash
cp notebooks/house_price.pkl backend/models/
cp notebooks/locations.json backend/models/
```

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

Run the tests:

```bash
pytest
```

### Environment variables (backend/.env)

| Variable          | Default                     | Description                                  |
|-------------------|------------------------------|-----------------------------------------------|
| `MODEL_PATH`      | `models/house_price.pkl`     | Path to the exported pipeline                 |
| `LOCATIONS_PATH`  | `models/locations.json`      | Path to the allowed-locations list            |
| `ALLOWED_ORIGINS` | `http://localhost:5173`      | Comma-separated origins allowed by CORS       |

### API reference

**`GET /health`**
```json
{ "status": "ok", "model_loaded": true }
```

**`GET /locations`** → `["Sector 45", "Andheri West", ...]`

**`POST /predict`**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "location": "Sector 45",
        "carpet_area_sqft": 1200,
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East"
      }'
```

```json
{ "predicted_price": 8500000.0, "currency": "INR" }
```

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env

npm run dev
# → http://localhost:5173
```

### Environment variables (frontend/.env)

| Variable              | Default                  | Description                  |
|------------------------|---------------------------|-------------------------------|
| `VITE_API_BASE_URL`    | `http://localhost:8000`   | Base URL of the FastAPI backend |

## Model performance

_Fill this in from your notebook's Phase 2.5 evaluation section, e.g.:_

| Model              | MAE (₹) | RMSE (₹) | R²   |
|---------------------|---------|----------|------|
| Linear Regression   | —       | —        | —    |
| Random Forest        | —       | —        | —    |

Chosen model: **_(fill in)_** — justification: _(fill in)_.

## Screenshots

_Add screenshots of the running app here before submitting._

## Deliverables checklist

- [ ] `notebooks/house_price_model.ipynb` runs top-to-bottom without errors
- [ ] `backend/` — `/health` + `/predict` working, tests passing
- [ ] `frontend/` — form → result page, `npm run build` succeeds
- [ ] `backend/models/house_price.pkl` present and served
- [ ] This README verified by cloning into a fresh folder
- [ ] Public GitHub repo, clean history (no `node_modules`, `.venv`, `.env`, raw CSV)
