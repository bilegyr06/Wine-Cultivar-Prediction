from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

app = Flask(__name__)

# =====================
# LOAD MODEL & SCALER
# =====================
MODEL_PATH = 'best_svm.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

TARGET_NAMES = ['Class 0', 'Class 1', 'Class 2']

FEATURE_NAMES = [
    'alcohol','malic_acid','ash','alcalinity_of_ash','magnesium',
    'total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins',
    'color_intensity','hue','od280/od315_of_diluted_wines','proline'
]

# =====================
# LOAD DATA FOR EDA
# =====================
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# =====================
# GENERATE EDA IMAGES
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
@app.route('/')
def index():
    return render_template(
        'index.html',
        features=FEATURE_NAMES,
        result=False,
        eda_scatter='static/eda_scatter.png',
        class_dist='static/class_distribution.png'
    )

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []

    for feature in FEATURE_NAMES:
        input_data.append(float(request.form[feature]))

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    cultivar = TARGET_NAMES[prediction]

    return render_template(
        'index.html',
        features=FEATURE_NAMES,
        result=True,
        cultivar=cultivar,
        eda_scatter='static/eda_scatter.png',
        class_dist='static/class_distribution.png'
    )

# =====================
# RUN SERVER (LAST!)
# =====================
if __name__ == "__main__":
    app.run(debug=True)
