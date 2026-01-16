# wine_svm_pipeline.py
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline (scaler + linear SVM)
model = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1, random_state=42, max_iter=10000)
)

# Train model
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy on test set: {accuracy_score(y_test, preds):.4f}")

# Save pipeline (single file)
joblib.dump(model, "wine_svm_pipeline.pkl")

print("Pipeline with scaler + LinearSVC saved!")
