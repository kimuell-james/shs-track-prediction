import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from predictor.models import Student, StudentGrade
from django.conf import settings

def train_model():
    # Get students who have both actual and predicted tracks
    students = Student.objects.filter(actual_track__isnull=False, predicted_track__isnull=False)

    if not students.exists():
        return "⚠️ No students with both actual and predicted track."

    # Join student info and grades
    data = []
    for student in students:
        grades = StudentGrade.objects.filter(student_id=student).first()
        if grades:
            data.append({
                "age": student.age,
                "gender": student.gender,
                "actual_track": student.actual_track,
                "g7_filipino": grades.g7_filipino,
                "g7_english": grades.g7_english,
                "g7_math": grades.g7_math,
                "g7_science": grades.g7_science,
                "g7_ap": grades.g7_ap,
                "g7_tle": grades.g7_tle,
                "g7_mapeh": grades.g7_mapeh,
                "g7_esp": grades.g7_esp,
                "g8_filipino": grades.g8_filipino,
                "g8_english": grades.g8_english,
                "g8_math": grades.g8_math,
                "g8_science": grades.g8_science,
                "g8_ap": grades.g8_ap,
                "g8_tle": grades.g8_tle,
                "g8_mapeh": grades.g8_mapeh,
                "g8_esp": grades.g8_esp,
                "g9_filipino": grades.g9_filipino,
                "g9_english": grades.g9_english,
                "g9_math": grades.g9_math,
                "g9_science": grades.g9_science,
                "g9_ap": grades.g9_ap,
                "g9_tle": grades.g9_tle,
                "g9_mapeh": grades.g9_mapeh,
                "g9_esp": grades.g9_esp,
                "g10_filipino": grades.g10_filipino,
                "g10_english": grades.g10_english,
                "g10_math": grades.g10_math,
                "g10_science": grades.g10_science,
                "g10_ap": grades.g10_ap,
                "g10_tle": grades.g10_tle,
                "g10_mapeh": grades.g10_mapeh,
                "g10_esp": grades.g10_esp,
            })

    df = pd.DataFrame(data)

    feature_cols = [col for col in df.columns if col not in ["actual_track"]]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        return f"Missing feature columns: {', '.join(missing)}"

    X = df[feature_cols]
    y = df["actual_track"]

    # Encode categorical variables
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    os.makedirs('ml_models', exist_ok=True)
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'shs_track_model.pkl')
    joblib.dump(model, model_path)

    return f"✅ Model trained successfully (Accuracy: {accuracy:.2%})" # and saved to {model_path}
