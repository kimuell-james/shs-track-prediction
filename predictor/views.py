import os
from django.conf import settings
import joblib
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
import numpy as np
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from .predict_track import predict_track_for_student 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

from .models import *
from .forms import *
from .decorators import unauthenticated_user

def admin_required(user):
    return user.is_superuser  # or user.is_staff depending on your design

def registerPage(request):
    if not request.user.is_superuser:  # only superuser can register users
        messages.error(request, "You do not have permission to register users.")
        return redirect("login")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "User created successfully!")
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "predictor/register.html", {"form": form})

@unauthenticated_user
def loginPage(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome {username}!")
                return redirect("home")  # change to your homepage
            else:
                messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()
    return render(request, "predictor/login.html", {"form": form})

def logoutUser(request):
    logout(request)
    messages.info(request, "Logged out successfully.")
    return redirect("login")

@login_required(login_url="/login/")
def home(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/dashboard.html', context)

@login_required(login_url="/login/")
def studentsRecord(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/students_record.html', context)

@login_required(login_url="/login/")
def addStudentRecord(request):
# Get the next student ID
    last_student = Student.objects.order_by('-student_id').first()
    next_id = last_student.student_id + 1 if last_student else 1

    if request.method == "POST":
        student_form = StudentForm(request.POST)
        grades_form = StudentGradesForm(request.POST)

        if student_form.is_valid() and grades_form.is_valid():
            student = student_form.save(commit=False)
            student.student_id = next_id  # assign ID manually
            student.save()

            grades = grades_form.save(commit=False)
            grades.student_id = student
            grades.save()

            return redirect(f"{reverse('students_record')}?msg=success")
        else:
            print("Add student - student_form errors:", student_form.errors)
            print("Add student - grades_form errors:", grades_form.errors)

    else:
        student_form = StudentForm()
        grades_form = StudentGradesForm()

    context = {
        'student_form': student_form,
        'grades_form': grades_form,
        'readonly': False,
        "is_add": True,
        "display_id": next_id
    }
    return render(request, "predictor/student_form.html", context)

@login_required(login_url="/login/")
def viewStudentRecord(request, pk):
    student = get_object_or_404(Student, student_id=pk)
    grades = get_object_or_404(StudentGrade, student_id=student)

    student_form = StudentForm(instance=student)
    grades_form = StudentGradesForm(instance=grades)

    for field in student_form.fields.values():
        field.widget.attrs['readonly'] = True
        field.widget.attrs['disabled'] = True

    for field in grades_form.fields.values():
        field.widget.attrs['readonly'] = True
        field.widget.attrs['disabled'] = True

    context = {'student_form': student_form, 'grades_form': grades_form, 'readonly': True, 'mode': 'view', 'display_id': student.student_id,}
    return render(request, 'predictor/student_form.html', context)

@login_required(login_url="/login/")
def updateStudentRecord(request, pk):
    student = get_object_or_404(Student, student_id=pk)
    grades = get_object_or_404(StudentGrade, student_id=student)

    if request.method == 'POST':
        student_form = StudentForm(request.POST, instance=student)
        grades_form = StudentGradesForm(request.POST, instance=grades)

        if student_form.is_valid() and grades_form.is_valid():
            student_form.save()
            grades_form.save()
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True})
            return redirect(f"{reverse('students_record')}?msg=success")
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'errors': {
                    'student': student_form.errors,
                    'grades': grades_form.errors
                }})
            return redirect(f"{reverse('students_record')}?msg=error")
    
    # normal GET
    student_form = StudentForm(instance=student)
    grades_form = StudentGradesForm(instance=grades)
    context = {'student_form': student_form, 'grades_form': grades_form, 'readonly': False, "is_add": False, 'mode': 'edit', 'display_id': student.student_id,}
    return render(request, 'predictor/student_form.html', context)

@login_required(login_url="/login/")
def deleteStudentRecord(request, pk):
    student = get_object_or_404(Student, student_id=pk)

    if request.method == "POST":
        # Delete grades first (if cascade is not set in model)
        StudentGrade.objects.filter(student_id=student).delete()

        # Delete the student
        student.delete()

        messages.success(request, "Student record deleted successfully!")
        return redirect(f"{reverse('students_record')}?msg=deleted")

    else:
        return redirect(f"{reverse('students_record')}?msg=error")

@login_required(login_url="/login/")
def predictStudentTrack(request, pk):
    student = get_object_or_404(Student, student_id=pk)
    grades = get_object_or_404(StudentGrade, student_id=student)

    predicted_track, predicted_label, top_contributors = predict_track_for_student(student, grades)

    # Extract only subject names from tuples
    subject_names = [name for name, _ in top_contributors]

    student.predicted_track = predicted_label
    student.contributing_subjects = ", ".join(subject_names)  # safe now
    student.save()

    messages.success(request, f"Prediction complete: {predicted_label}")
    # return redirect(f"{reverse('students_record')}?msg=success")
    return redirect(f"{reverse('students_record')}?msg=success#student-{student.student_id}")

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64

@login_required(login_url="/login/")
def modelEvaluation(request):
    # Load model
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'logreg_balanced.pkl')
    model = joblib.load(model_path)

    # Students that have both actual and predicted tracks
    students = Student.objects.exclude(actual_track__isnull=True).exclude(actual_track__exact="") \
        .exclude(predicted_track__isnull=True).exclude(predicted_track__exact="") \
        .values("student_id", "actual_track", "predicted_track", "age", "gender")

    if not students:
        return render(request, 'predictor/model_evaluation.html', {"no_data": True})

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

    if not grades:
        return render(request, 'predictor/model_evaluation.html', {"no_data": True})

    # Convert to DataFrames
    df_students = pd.DataFrame(list(students))
    df_grades = pd.DataFrame(list(grades))
    df_grades.rename(columns={"student_id_id": "student_id"}, inplace=True)

    # Merge
    df = pd.merge(df_students, df_grades, on="student_id", how="inner")

    # Features
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

    # Encode gender
    if df["gender"].dtype == "object":
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    X = df[feature_cols]
    y_true = df["actual_track"]

    # Predict
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # probability of class 1 (TVL)

    # Map predictions to labels
    track_map = {0: "Academic", 1: "TVL"}
    y_pred_labels = [track_map[p] for p in y_pred]

    # Map true labels for ROC
    y_true_bin = y_true.map({"Academic": 0, "TVL": 1})

    # Metrics
    count = df.shape[0]
    accuracy = accuracy_score(y_true, y_pred_labels)
    conf_matrix = confusion_matrix(y_true, y_pred_labels).tolist()
    report = classification_report(y_true, y_pred_labels, output_dict=True)
    roc_auc = roc_auc_score(y_true_bin, y_pred_proba)

    # Confusion Matrix heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Academic", "TVL"], yticklabels=["Academic", "TVL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_base64 = plot_to_base64()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_base64 = plot_to_base64()

    # Precision/Recall/F1 bar chart
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            if "f1-score" in metrics:
                metrics["f1_score"] = metrics.pop("f1-score")

    metrics_labels = []
    precisions = []
    recalls = []
    f1s = []
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics_labels.append(label)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1_score"])

    x = range(len(metrics_labels))
    plt.figure(figsize=(5, 3))
    plt.bar(x, precisions, width=0.25, label="Precision")
    plt.bar([i + 0.25 for i in x], recalls, width=0.25, label="Recall")
    plt.bar([i + 0.5 for i in x], f1s, width=0.25, label="F1-Score")
    plt.xticks([i + 0.25 for i in x], metrics_labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Classification Metrics")
    plt.legend()
    metrics_base64 = plot_to_base64()

    context = {
        "count": count,
        "accuracy": round(accuracy * 100, 2),
        "roc_auc": round(roc_auc, 3),
        "records": df.to_dict(orient="records"),
        "cm_base64": cm_base64,
        "roc_base64": roc_base64,
        "metrics_base64": metrics_base64,
        "report": report,
        "no_data": False,
    }

    return render(request, "predictor/model_evaluation.html", context)

@user_passes_test(admin_required, login_url="/login/")
def adminPanel(request):
    admin = Admin.objects.all()

    context = {'admin': admin}

    return render(request, 'predictor/admin_panel.html', context)

@user_passes_test(admin_required, login_url="/login/")
@login_required(login_url="/login/")
def adminList(request):
    admin = Admin.objects.all()

    context = {'admin': admin}

    return render(request, 'predictor/admin_list.html', context)
