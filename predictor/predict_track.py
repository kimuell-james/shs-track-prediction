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
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'logreg_balanced.pkl')
    model = joblib.load(model_path)

    # Predict
    prediction = model.predict([input_data])[0]

    track_map = {0: "Academic", 1: "TVL"}
    predicted_label = track_map.get(prediction, "Unknown")

    # --- Feature names mapping ---
    feature_names = [
        "age", "gender",
        "g7_filipino", "g7_english", "g7_math", "g7_science", "g7_ap", "g7_tle", "g7_mapeh", "g7_esp",
        "g8_filipino", "g8_english", "g8_math", "g8_science", "g8_ap", "g8_tle", "g8_mapeh", "g8_esp",
        "g9_filipino", "g9_english", "g9_math", "g9_science", "g9_ap", "g9_tle", "g9_mapeh", "g9_esp",
        "g10_filipino", "g10_english", "g10_math", "g10_science", "g10_ap", "g10_tle", "g10_mapeh", "g10_esp"
    ]

    subject_labels = {
        "g7_filipino": "Grade 7 - Filipino",
        "g7_english": "Grade 7 - English",
        "g7_math": "Grade 7 - Mathematics",
        "g7_science": "Grade 7 - Science",
        "g7_ap": "Grade 7 - Araling Panlipunan",
        "g7_tle": "Grade 7 - TLE",
        "g7_mapeh": "Grade 7 - MAPEH",
        "g7_esp": "Grade 7 - ESP",
        "g8_filipino": "Grade 8 - Filipino",
        "g8_english": "Grade 8 - English",
        "g8_math": "Grade 8 - Mathematics",
        "g8_science": "Grade 8 - Science",
        "g8_ap": "Grade 8 - Araling Panlipunan",
        "g8_tle": "Grade 8 - TLE",
        "g8_mapeh": "Grade 8 - MAPEH",
        "g8_esp": "Grade 8 - ESP",
        "g9_filipino": "Grade 9 - Filipino",
        "g9_english": "Grade 9 - English",
        "g9_math": "Grade 9 - Mathematics",
        "g9_science": "Grade 9 - Science",
        "g9_ap": "Grade 9 - Araling Panlipunan",
        "g9_tle": "Grade 9 - TLE",
        "g9_mapeh": "Grade 9 - MAPEH",
        "g9_esp": "Grade 9 - ESP",
        "g10_filipino": "Grade 10 - Filipino",
        "g10_english": "Grade 10 - English",
        "g10_math": "Grade 10 - Mathematics",
        "g10_science": "Grade 10 - Science",
        "g10_ap": "Grade 10 - Araling Panlipunan",
        "g10_tle": "Grade 10 - TLE",
        "g10_mapeh": "Grade 10 - MAPEH",
        "g10_esp": "Grade 10 - ESP",
    }

    # Coefficients per class (use predicted class)
    coef = model.coef_[0]
    feature_contributions = input_data * coef

    # Exclude age & gender
    contributions_filtered = [
        (subject_labels.get(name, name), contrib)
        for name, contrib in zip(feature_names, feature_contributions)
        if name not in ["age", "gender"]
    ]

    # Top 5 contributors
    top_contributing_subjects = sorted(
        contributions_filtered,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return prediction, predicted_label, top_contributing_subjects
