import joblib

def predict_track_for_student(student):
    # Example - replace with your actual data extraction logic
    input_data = [
        student.age,
        student.gender,
        student.g7_filipino,
        student.g7_english,
        student.g7_math,
        student.g7_science,
        student.g7_ap,
        student.g7_tle,
        student.g7_mapeh,
        student.g7_esp,
        student.g8_filipino,
        student.g8_english,
        student.g8_math,
        student.g8_science,
        student.g8_ap,
        student.g8_tle,
        student.g8_mapeh,
        student.g8_esp,
        student.g9_filipino,
        student.g9_english,
        student.g9_math,
        student.g9_science,
        student.g9_ap,
        student.g9_tle,
        student.g9_mapeh,
        student.g9_esp,
        student.g10_filipino,
        student.g10_english,
        student.g10_math,
        student.g10_science,
        student.g10_ap,
        student.g10_tle,
        student.g10_mapeh,
        student.g10_esp,
    ]

    model = joblib.load('ml_models/logreg_unbalanced.pkl')
    prediction = model.predict([input_data])[0]
    
    # Optional: get contributing features (dummy example)
    contributing = ["g10_math", "g10_science"] if prediction == 'STEM' else ["g10_english"]

    return prediction, contributing