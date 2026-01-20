import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Wine Dataset
print("Loading Wine Dataset...")
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# 2. Data Preprocessing
print("\n--- Data Preprocessing ---")

# Check for missing values
missing_values = X.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# Feature Selection (keeping all features for Wine dataset as they're all relevant)
print(f"Number of features: {X.shape[1]}")
print(f"Features: {list(X.columns)}")

# 3. Split Dataset
print("\nSplitting dataset (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 4. Build and Train Model using make_pipeline
print("\n--- Building model with StandardScaler and Random Forest ---")
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

print("Training model...")
model.fit(X_train, y_train)

# 5. Make Predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 6. Evaluate Model
print("\n--- Model Evaluation ---")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-Score (Weighted and Macro)
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("\nWeighted Metrics:")
print(f"Precision (Weighted): {precision_weighted:.4f}")
print(f"Recall (Weighted): {recall_weighted:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")

print("\nMacro Metrics:")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 7. Save model
model_path = "./model/wine_classification_model.pkl"

print(f"\nSaving model to {model_path}...")
joblib.dump(model, model_path)
print("model saved successfully!")

# 8. Feature Importance
print("\n--- Feature Importance ---")
rf_model = model.named_steps['randomforestclassifier']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('./static/feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./static/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

print("\n--- model Complete ---")