import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from django.conf import settings
from predictor.models import Student, StudentGrade, SchoolYear, ModelTrainingHistory

def train_model():
    # Get current school year
    current_sy = SchoolYear.objects.filter(is_current=True).first()
    if not current_sy:
        return "⚠️ No current school year set!"

    # Collect all students with complete data
    students = Student.objects.filter(actual_track__isnull=False)
    if not students.exists():
        return "⚠️ No students found with actual tracks."

    # Combine student records
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
    if df.empty:
        return "⚠️ No valid grade records found."

    # Preprocessing
    X = df.drop(columns=["actual_track"], errors="ignore")
    y = df["actual_track"]

    # Encode gender
    X["gender"] = X["gender"].map({"Male":1, "Female":0})

    # Save feature columns before scaling
    feature_columns = X.columns.tolist()

    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_scaled, y_encoded)

    # Training metrics
    y_pred = model.predict(X_scaled)
    training_accuracy = accuracy_score(y_encoded, y_pred)
    report = classification_report(y_encoded, y_pred)

    # Save model & scaler
    model_dir = os.path.join(settings.BASE_DIR, "predictor", "ml_models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"shs_track_insight_model_{current_sy.school_year.replace(' ','').replace('-','_')}_{timestamp}.pkl"
    
    joblib.dump(model, os.path.join(model_dir, model_filename))
    joblib.dump(scaler, os.path.join(model_dir, f"{model_filename}_scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(model_dir, f"{model_filename}_columns.pkl"))

    # ✅ Only include school years of students that qualify for training
    dataset_size = students.count()
    included_school_years = (
        SchoolYear.objects.filter(sy_id__in=students.values_list("sy_id", flat=True))
        .values_list("school_year", flat=True)
        .distinct()
    )
    included_school_years_text = ", ".join(included_school_years)

    # Log training
    new_model = ModelTrainingHistory.objects.create(
        school_year=current_sy,
        dataset_count=dataset_size,
        included_school_years=included_school_years_text,
        model_filename=model_filename,
        accuracy=training_accuracy,
        trained_at=datetime.now(),
        is_active=True,
    )
    ModelTrainingHistory.objects.exclude(pk=new_model.pk).update(is_active=False)

    return f"Model trained successfully for {current_sy.school_year} (Accuracy: {training_accuracy:.2%}, {dataset_size} records)"
