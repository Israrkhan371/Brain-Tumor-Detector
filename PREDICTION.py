import joblib
import numpy as np

from utils import (
    extract_features,
    load_image
)

# =========================================================
# LOAD MODELS + SCALER
# =========================================================

svm_model = joblib.load("SVM.pkl")

rf_model = joblib.load("Random_Forest.pkl")

xgb_model = joblib.load("XGBoost.pkl")

scaler = joblib.load("scaler.pkl")

print("Models Loaded Successfully!")

# =========================================================
# CLASS LABELS
# =========================================================

classes = {
    0: "Meningioma",
    1: "Glioma",
    2: "Pituitary Tumor"
}

# =========================================================
# MODEL WEIGHTS
# =========================================================

weights = {
    "svm": 0.332,
    "rf": 0.333,
    "xgb": 0.335
}

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(path):

    image, actual = load_image(path)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    features = extract_features(image)

    features = features.reshape(1, -1)

    # =====================================================
    # SCALE FEATURES
    # =====================================================

    features = scaler.transform(features)

    # =====================================================
    # MODEL PROBABILITIES
    # =====================================================

    svm_proba = svm_model.predict_proba(features)[0]

    rf_proba = rf_model.predict_proba(features)[0]

    xgb_proba = xgb_model.predict_proba(features)[0]

    # =====================================================
    # WEIGHTED PROBABILITY FUSION
    # =====================================================

    final_proba = (
        weights["svm"] * svm_proba +
        weights["rf"] * rf_proba +
        weights["xgb"] * xgb_proba
    )

    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    final_prediction = np.argmax(final_proba)

    return {
        "prediction": classes[final_prediction],
        "prediction_id": final_prediction,
        "actual": classes[actual] if actual is not None else None,
        "image": image,
        "probabilities": {
            "Meningioma": float(final_proba[0] * 100),
            "Glioma": float(final_proba[1] * 100),
            "Pituitary Tumor": float(final_proba[2] * 100)
        }
    }

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    path = "dataset_split/test/123.mat"

    result = predict(path)

    print("Prediction:", result["prediction"])

    print("Actual:", result["actual"])