import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse

app = FastAPI(title="Wine Cultivar Prediction")

# Mount static files (images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (assumes your index.html is in a "templates" folder)
templates = Jinja2Templates(directory="templates")

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = './model/wine_classification_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: model not found at {MODEL_PATH}. Please run train_model.py first.")
    raise

# Load wine dataset for target names and feature information
wine_data = load_wine()
TARGET_NAMES = wine_data.target_names
FEATURE_NAMES = list(wine_data.feature_names)

# =====================
# LOAD DATA FOR EDA
# =====================
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# =====================
# GENERATE EDA IMAGES (run once at startup)
# =====================
os.makedirs('static', exist_ok=True)

eda_scatter_path = os.path.join('static', 'eda_scatter.png')
class_dist_path = os.path.join('static', 'class_distribution.png')

if not os.path.exists(eda_scatter_path):
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        x='alcohol',
        y='flavanoids',
        hue='target',
        data=wine_df,
        palette='Set2'
    )
    plt.title('Alcohol vs Flavanoids')
    plt.tight_layout()
    plt.savefig(eda_scatter_path)
    plt.close()

if not os.path.exists(class_dist_path):
    plt.figure(figsize=(5, 4))
    sns.countplot(x='target', data=wine_df, palette='Set2')
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Wine Cultivar')
    plt.tight_layout()
    plt.savefig(class_dist_path)
    plt.close()

# =====================
# ROUTES
# =====================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURE_NAMES,
            "result": False,
            "eda_scatter": "/static/eda_scatter.png",
            "class_dist": "/static/class_distribution.png"
        }
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    errors = []
    input_data = []

    # Collect and validate features
    for feature in FEATURE_NAMES:
        value = form_data.get(feature)
        if value is None or value.strip() == "":
            errors.append(f"{feature.replace('_', ' ').title()} is required.")
        else:
            try:
                input_data.append(float(value))
            except ValueError:
                errors.append(f"{feature.replace('_', ' ').title()} must be a valid number.")

    if errors:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": FEATURE_NAMES,
                "result": False,
                "errors": errors,
                "eda_scatter": "/static/eda_scatter.png",
                "class_dist": "/static/class_distribution.png"
            }
        )

    try:
        # Use the model for prediction (handles scaling automatically)
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        confidence = float(max(prediction_proba)) * 100
        cultivar = TARGET_NAMES[prediction]
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": FEATURE_NAMES,
                "result": False,
                "errors": [f"Prediction failed: {str(e)}"],
                "eda_scatter": "/static/eda_scatter.png",
                "class_dist": "/static/class_distribution.png"
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURE_NAMES,
            "result": True,
            "cultivar": cultivar,
            "confidence": f"{confidence:.2f}",
            "eda_scatter": "/static/eda_scatter.png",
            "class_dist": "/static/class_distribution.png",
            "feature_importance": "/static/feature_importance.png",
            "confusion_matrix": "/static/confusion_matrix.png"
        }
    )

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)