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
# LOAD MODEL & SCALER
# =====================
MODEL_PATH = 'best_svm.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

TARGET_NAMES = ['Class 0', 'Class 1', 'Class 2']

FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# =====================
# LOAD DATA FOR EDA
# =====================
wine_data = load_wine()
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
    sns.countplot(x='target', data=wine_df)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(class_dist_path)
    plt.close()

# =====================
# ROUTES
# =====================
@app.get("") 
async def redirect_root(): 
    return RedirectResponse(url="/")

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



# ... rest of your imports ...

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
        # If you trained with a pipeline, just call model.predict directly
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
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
            "eda_scatter": "/static/eda_scatter.png",
            "class_dist": "/static/class_distribution.png"
        }
    )

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # FastAPI commonly uses 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)  # reload=True for dev