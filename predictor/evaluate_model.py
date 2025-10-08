# predictor/evaluate_model.py

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from django.conf import settings
from predictor.models import Student, StudentGrade, SchoolYear, ModelTrainingHistory
from io import BytesIO
import base64

# Helper to convert matplotlib plot to base64
def plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def evaluate_active_model(school_year: SchoolYear):
    # Get active model
    active_model = ModelTrainingHistory.objects.filter(is_active=True).first()
    if not active_model:
        return {"error": "No active model found."}

    # Load model & scaler
    model_path = os.path.join(settings.BASE_DIR, "predictor", "ml_models", active_model.model_filename)
    scaler_path = f"{model_path}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"error": "Model files not found."}

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Get students with actual & predicted tracks in this school year
    students = Student.objects.filter(sy=school_year) \
        .exclude(actual_track__isnull=True).exclude(actual_track__exact="") \
        .exclude(predicted_track__isnull=True).exclude(predicted_track__exact="") \
        .values("student_id", "actual_track", "predicted_track", "age", "gender")

    if not students:
        return {"no_data": True}

    student_ids = [s["student_id"] for s in students]

    # Only grades for those students
    grades = StudentGrade.objects.filter(student_id__in=student_ids).values(
        "student_id_id",
        "g7_filipino", "g7_english", "g7_math", "g7_science", "g7_ap",
        "g7_tle", "g7_mapeh", "g7_esp",
        "g8_filipino", "g8_english", "g8_math", "g8_science", "g8_ap",
        "g8_tle", "g8_mapeh", "g8_esp",
        "g9_filipino", "g9_english", "g9_math", "g9_science", "g9_ap",
        "g9_tle", "g9_mapeh", "g9_esp",
        "g10_filipino", "g10_english", "g10_math", "g10_science", "g10_ap",
        "g10_tle", "g10_mapeh", "g10_esp"
    )

    df_students = pd.DataFrame(list(students))
    df_grades = pd.DataFrame(list(grades))
    df_grades.rename(columns={"student_id_id": "student_id"}, inplace=True)

    df = pd.merge(df_students, df_grades, on="student_id", how="inner")

    # Feature columns
    feature_cols = [
        "age", "gender",
        "g7_filipino", "g7_english", "g7_math", "g7_science", "g7_ap",
        "g7_tle", "g7_mapeh", "g7_esp",
        "g8_filipino", "g8_english", "g8_math", "g8_science", "g8_ap",
        "g8_tle", "g8_mapeh", "g8_esp",
        "g9_filipino", "g9_english", "g9_math", "g9_science", "g9_ap",
        "g9_tle", "g9_mapeh", "g9_esp",
        "g10_filipino", "g10_english", "g10_math", "g10_science", "g10_ap",
        "g10_tle", "g10_mapeh", "g10_esp"
    ]

    if df["gender"].dtype == "object":
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    X = df[feature_cols]
    y_true = df["actual_track"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Map predictions
    track_map = {0: "Academic", 1: "TVL"}
    y_pred_labels = [track_map[p] for p in y_pred]
    y_true_bin = y_true.map({"Academic": 0, "TVL": 1})

    # Metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    conf_matrix = confusion_matrix(y_true, y_pred_labels).tolist()
    report = classification_report(y_true, y_pred_labels, output_dict=True)
    roc_auc = roc_auc_score(y_true_bin, y_pred_proba)

    # Rename 'f1-score' -> 'f1_score' for all labels
    for label, metrics in report.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            metrics["f1_score"] = metrics.pop("f1-score")

    # Confusion matrix plot
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Academic", "TVL"], yticklabels=["Academic", "TVL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_base64 = plot_to_base64()

    # ROC plot
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba)
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_base64 = plot_to_base64()

    # Generate Analytical Insights
    def generate_model_analysis():
        insights = {}

        # --- Overall Model Performance ---
        if accuracy >= 0.85:
            perf_desc = "excellent performance with high predictive reliability"
        elif accuracy >= 0.75:
            perf_desc = "strong and consistent accuracy across tracks"
        elif accuracy >= 0.65:
            perf_desc = "moderate accuracy with potential room for feature refinement"
        else:
            perf_desc = "low performance; model may require retraining or additional predictors"

        insights["overall"] = (
            f"The model achieved an accuracy of <strong>{accuracy*100:.2f}%</strong>, "
            f"showing {perf_desc}. The ROC-AUC score of <strong>{roc_auc:.3f}</strong> "
            f"indicates the model’s ability to correctly distinguish between the <strong>Academic</strong> "
            f"and <strong>TVL</strong> tracks."
        )

        # --- Class-wise Analysis ---
        track_metrics = {k: v for k, v in report.items() if k in ["Academic", "TVL"]}
        if track_metrics:
            high_precision = max(track_metrics.items(), key=lambda x: x[1]["precision"])
            high_recall = max(track_metrics.items(), key=lambda x: x[1]["recall"])

            insights["classwise"] = (
                f"Among the two tracks, <strong>{high_precision[0]}</strong> achieved the highest precision "
                f"({high_precision[1]['precision']:.2f}), indicating fewer false positives, "
                f"while <strong>{high_recall[0]}</strong> has higher recall "
                f"({high_recall[1]['recall']:.2f}), suggesting better coverage of actual students "
                f"in that category."
            )

        # --- Confusion Matrix Analysis ---
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
        total = tn + fp + fn + tp
        academic_acc = (tn + fp) and (tn / (tn + fp)) or 0
        tvl_acc = (tp + fn) and (tp / (tp + fn)) or 0

        if abs(academic_acc - tvl_acc) > 0.15:
            stronger = "Academic" if academic_acc > tvl_acc else "TVL"
            insights["confusion"] = (
                f"The confusion matrix suggests that the model performs better at correctly "
                f"classifying <strong>{stronger}</strong> students, indicating a potential class imbalance or "
                f"bias in training data."
            )
        else:
            insights["confusion"] = (
                "The confusion matrix shows fairly balanced predictions between Academic and TVL tracks, "
                "implying the model generalizes well across both classes."
            )

        # --- ROC-AUC Analysis ---
        if roc_auc >= 0.90:
            auc_desc = "outstanding discriminatory capability between the two tracks"
        elif roc_auc >= 0.80:
            auc_desc = "very good class separation performance"
        elif roc_auc >= 0.70:
            auc_desc = "acceptable distinction between Academic and TVL tracks"
        elif roc_auc >= 0.60:
            auc_desc = "limited ability to distinguish between classes, suggesting overlap in features"
        else:
            auc_desc = "poor separation ability; predictions may rely on noisy or insufficient data"

        insights["roc_auc"] = ( 
            f"The ROC-AUC score of <strong>{roc_auc:.3f}</strong> reflects {auc_desc}. "
            f"This means the model can correctly rank a randomly chosen Academic student higher than "
            f"a TVL student about <strong>{roc_auc*100:.1f}%</strong> of the time."
        )

        # --- Recommendation ---
        if accuracy < 0.70 or roc_auc < 0.75:
            insights["recommendation"] = (
                "🔍 Consider retraining the model with more recent data or refining features such as "
                "subject-level averages, gender balance, or grade normalization to improve predictive accuracy."
            )
        else:
            insights["recommendation"] = (
                "✅ The model demonstrates reliable performance and can be used for predictive insights. "
                "However, continuous monitoring with new data is recommended to maintain accuracy."
            )

        return insights

    analysis = generate_model_analysis()

    return {
        "count": len(df),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix,
        "cm_base64": cm_base64,
        "roc_base64": roc_base64,
        "report": report,
        "students_df": df,
        "no_data": False,
        "school_year": school_year,
        "analysis": analysis,
    }
