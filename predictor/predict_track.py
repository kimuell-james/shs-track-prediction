import os
from django.conf import settings
import joblib
import numpy as np

def predict_track_for_student(student, grades):
    # Convert gender to numeric
    gender_map = {"Male": 1, "Female": 0}
    gender_value = gender_map.get(str(student.gender).strip(), 0)

    # Ensure all values are numeric (replace None with 0)
    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # Prepare input data
    input_data = np.array([
        safe_float(student.age),
        gender_value,
        safe_float(grades.g7_filipino),
        safe_float(grades.g7_english),
        safe_float(grades.g7_math),
        safe_float(grades.g7_science),
        safe_float(grades.g7_ap),
        safe_float(grades.g7_tle),
        safe_float(grades.g7_mapeh),
        safe_float(grades.g7_esp),
        safe_float(grades.g8_filipino),
        safe_float(grades.g8_english),
        safe_float(grades.g8_math),
        safe_float(grades.g8_science),
        safe_float(grades.g8_ap),
        safe_float(grades.g8_tle),
        safe_float(grades.g8_mapeh),
        safe_float(grades.g8_esp),
        safe_float(grades.g9_filipino),
        safe_float(grades.g9_english),
        safe_float(grades.g9_math),
        safe_float(grades.g9_science),
        safe_float(grades.g9_ap),
        safe_float(grades.g9_tle),
        safe_float(grades.g9_mapeh),
        safe_float(grades.g9_esp),
        safe_float(grades.g10_filipino),
        safe_float(grades.g10_english),
        safe_float(grades.g10_math),
        safe_float(grades.g10_science),
        safe_float(grades.g10_ap),
        safe_float(grades.g10_tle),
        safe_float(grades.g10_mapeh),
        safe_float(grades.g10_esp),
    ], dtype=float)

    # Load model
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'logreg_unbalanced.pkl')
    model = joblib.load(model_path)

    # Predict
    prediction = model.predict([input_data])[0]

    track_map = {0: "Academic", 1: "TVL"}
    predicted_label = track_map.get(prediction, "Unknown")

    # --- Get contributing features ---
    # Make sure you pass the same feature names used in training
    feature_names = [
        "age", "gender",
        "g7_filipino", "g7_english", "g7_math", "g7_science", "g7_ap", "g7_tle", "g7_mapeh", "g7_esp",
        "g8_filipino", "g8_english", "g8_math", "g8_science", "g8_ap", "g8_tle", "g8_mapeh", "g8_esp",
        "g9_filipino", "g9_english", "g9_math", "g9_science", "g9_ap", "g9_tle", "g9_mapeh", "g9_esp",
        "g10_filipino", "g10_english", "g10_math", "g10_science", "g10_ap", "g10_tle", "g10_mapeh", "g10_esp"
    ]

    # Coefficients per class (assuming binary or multi-class logistic regression)
    coef = model.coef_[0]
    feature_contributions = input_data * coef

    # Exclude age & gender from contributing subjects
    contributions_filtered = [
        (name, contrib)
        for name, contrib in zip(feature_names, feature_contributions)
        if name not in ["age", "gender"]
    ]

    contributing_subjects = sorted(
        contributions_filtered,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    contributing_subject_names = [name for name, _ in contributing_subjects]

    return prediction, predicted_label, contributing_subject_names
