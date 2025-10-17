import os
from django.conf import settings
import joblib
import numpy as np
import pandas as pd
from .models import ModelTrainingHistory
from predictor.load_supabase import load_active_model

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
    input_dict = {
        "age": safe_float(student.age),
        "gender": str(student.gender).strip(),
        "g7_filipino": safe_float(grades.g7_filipino),
        "g7_english": safe_float(grades.g7_english),
        "g7_math": safe_float(grades.g7_math),
        "g7_science": safe_float(grades.g7_science),
        "g7_ap": safe_float(grades.g7_ap),
        "g7_tle": safe_float(grades.g7_tle),
        "g7_mapeh": safe_float(grades.g7_mapeh),
        "g7_esp": safe_float(grades.g7_esp),
        "g8_filipino": safe_float(grades.g8_filipino),
        "g8_english": safe_float(grades.g8_english),
        "g8_math": safe_float(grades.g8_math),
        "g8_science": safe_float(grades.g8_science),
        "g8_ap": safe_float(grades.g8_ap),
        "g8_tle": safe_float(grades.g8_tle),
        "g8_mapeh": safe_float(grades.g8_mapeh),
        "g8_esp": safe_float(grades.g8_esp),
        "g9_filipino": safe_float(grades.g9_filipino),
        "g9_english": safe_float(grades.g9_english),
        "g9_math": safe_float(grades.g9_math),
        "g9_science": safe_float(grades.g9_science),
        "g9_ap": safe_float(grades.g9_ap),
        "g9_tle": safe_float(grades.g9_tle),
        "g9_mapeh": safe_float(grades.g9_mapeh),
        "g9_esp": safe_float(grades.g9_esp),
        "g10_filipino": safe_float(grades.g10_filipino),
        "g10_english": safe_float(grades.g10_english),
        "g10_math": safe_float(grades.g10_math),
        "g10_science": safe_float(grades.g10_science),
        "g10_ap": safe_float(grades.g10_ap),
        "g10_tle": safe_float(grades.g10_tle),
        "g10_mapeh": safe_float(grades.g10_mapeh),
        "g10_esp": safe_float(grades.g10_esp),
    }

    # Load active model
    model, scaler, training_columns = load_active_model()

    # Prepare input data
    input_df = pd.DataFrame([input_dict])
    input_df["gender"] = input_df["gender"].map({"Male":1, "Female":0})

    # Align columns and fill missing ones
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    track_map = {0: "Academic", 1: "TVL"}
    predicted_label = track_map.get(prediction, "Unknown")

    # Feature names mapping
    feature_names = [
        "age", "gender",
        "g7_filipino", "g7_english", "g7_math", "g7_science", "g7_ap", "g7_tle", "g7_mapeh", "g7_esp",
        "g8_filipino", "g8_english", "g8_math", "g8_science", "g8_ap", "g8_tle", "g8_mapeh", "g8_esp",
        "g9_filipino", "g9_english", "g9_math", "g9_science", "g9_ap", "g9_tle", "g9_mapeh", "g9_esp",
        "g10_filipino", "g10_english", "g10_math", "g10_science", "g10_ap", "g10_tle", "g10_mapeh", "g10_esp"
    ]

    subject_labels = {
        # Grade 7
        "g7_filipino": "G7 - Filipino",
        "g7_english": "G7 - English",
        "g7_math": "G7 - Math",
        "g7_science": "G7 - Science",
        "g7_ap": "G7 - AP",
        "g7_tle": "G7 - TLE",
        "g7_mapeh": "G7 - MAPEH",
        "g7_esp": "G7 - ESP",

        # Grade 8
        "g8_filipino": "G8 - Filipino",
        "g8_english": "G8 - English",
        "g8_math": "G8 - Math",
        "g8_science": "G8 - Science",
        "g8_ap": "G8 - AP",
        "g8_tle": "G8 - TLE",
        "g8_mapeh": "G8 - MAPEH",
        "g8_esp": "G8 - ESP",

        # Grade 9
        "g9_filipino": "G9 - Filipino",
        "g9_english": "G9 - English",
        "g9_math": "G9 - Math",
        "g9_science": "G9 - Science",
        "g9_ap": "G9 - AP",
        "g9_tle": "G9 - TLE",
        "g9_mapeh": "G9 - MAPEH",
        "g9_esp": "G9 - ESP",

        # Grade 10
        "g10_filipino": "G10 - Filipino",
        "g10_english": "G10 - English",
        "g10_math": "G10 - Math",
        "g10_science": "G10 - Science",
        "g10_ap": "G10 - AP",
        "g10_tle": "G10 - TLE",
        "g10_mapeh": "G10 - MAPEH",
        "g10_esp": "G10 - ESP",
    }

    # Contributions
    coef = model.coef_[0]
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_contributions = input_df.values.flatten() * coef

    contributions_filtered = []
    for name, contrib in zip(input_df.columns, feature_contributions):
        if name in ["age", "gender"]:
            continue
        # Only keep features that support the predicted track
        if (predicted_label == "Academic" and contrib < 0) or \
           (predicted_label == "TVL" and contrib > 0):
            contributions_filtered.append((subject_labels.get(name, name), float(contrib)))

    # Sort contributions by magnitude (largest positive for predicted track first)
    top_contributing_subjects = sorted(
        contributions_filtered,
        key=lambda x: x[1],
        reverse=True if predicted_label == "Academic" else False
    )[:5]

    return prediction, predicted_label, top_contributing_subjects
