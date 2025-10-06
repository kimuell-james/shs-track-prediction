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
import json
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from django.core.paginator import Paginator
from django import forms
from django.db.models import Count

from .predict_track import predict_track_for_student 
from .models import *
from .forms import *
from .train_model import train_model
from .decorators import unauthenticated_user

def admin_required(user):
    return user.is_superuser

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
                return redirect("home")  
            else:
                return redirect("login")
    else:
        form = LoginForm()

    context = {'form': form}

    return render(request, "predictor/login.html", context)

@login_required(login_url="/login/")
def logoutUser(request):
    logout(request)
    return redirect("login")

@login_required(login_url="/login/")
def home(request):

    # Get current school year
    current_sy = SchoolYear.objects.filter(is_current=True).first()

    if not current_sy:
        messages.info(request, "No active school year set.")
        return render(request, 'predictor/dashboard.html', {})
    
    # Stats
    total_students = Student.objects.filter(sy=current_sy).count()
    predicted_count = Student.objects.filter(sy=current_sy, predicted_track__isnull=False).count()
    actual_count = Student.objects.filter(sy=current_sy, actual_track__isnull=False).count()

    # Distributions
    # Distributions
    predicted_distribution = (
        Student.objects.filter(sy=current_sy)
        .values("predicted_track")
        .annotate(count=Count("predicted_track"))
    )
    actual_distribution = (
        Student.objects.filter(sy=current_sy)
        .values("actual_track")
        .annotate(count=Count("actual_track"))
    )
    gender_distribution = (
        Student.objects.filter(sy=current_sy)
        .values("actual_track", "gender")
        .annotate(count=Count("gender"))
    )
    age_distribution = (
        Student.objects.filter(sy=current_sy)
        .values("actual_track", "age")
        .annotate(count=Count("age"))
        .order_by("age")
    )

    # # Load data for describe()
    # students = Student.objects.filter(sy=current_sy).values()
    # grades = StudentGrade.objects.filter(student_id__sy=current_sy).values()

    # df_students = pd.DataFrame(students)
    # df_grades = pd.DataFrame(grades)

    # desc_students = df_students[["age", "grade_level"]].describe().to_html(classes="table table-striped") if not df_students.empty else None
    # desc_grades = df_grades.drop(columns=["grade_id", "student_id"]).describe().to_html(classes="table table-striped") if not df_grades.empty else None

    context = {
        "total_students": total_students,
        "predicted_count": predicted_count,
        "actual_count": actual_count,
        "predicted_distribution": json.dumps(list(predicted_distribution)),
        "actual_distribution": json.dumps(list(actual_distribution)),
        "gender_distribution": json.dumps(list(gender_distribution)),
        "age_distribution": json.dumps(list(age_distribution)),
        # "desc_students": desc_students,
        # "desc_grades": desc_grades,
        "current_sy": current_sy.school_year,
    }

    return render(request, 'predictor/dashboard.html', context)

@login_required(login_url="/login/")
def studentsRecord(request):
    current_sy = SchoolYear.objects.filter(is_current=True).first()

    if current_sy:
        records = Student.objects.filter(sy_id=current_sy).order_by("student_id")
    else:
        messages.info(request, "No active school year set.")
        records = Student.objects.none()  # no records if no active SY set

    paginator = Paginator(records, 50)  # 50 records per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {"records": records, "page_obj": page_obj, "current_sy": current_sy}

    return render(request, 'predictor/student_record.html', context)

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

            messages.success(request, "Student record added successfully!")
            return redirect("student_record")
        else:
            for form in [student_form, grades_form]:
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f"{field.capitalize()}: {error}")

            # print("Add student - student_form errors:", student_form.errors)
            # print("Add student - grades_form errors:", grades_form.errors)

            return render(request, 'your_template.html', {
                'student_form': student_form,
                'grades_form': grades_form
            })
            

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
            messages.success(request, f"Student {pk} record updated successfully!")
            return redirect("student_record")
        else:
            for form in [student_form, grades_form]:
                for field, errors in form.errors.items():
                    for error in errors:
                        messages.error(request, f"{field.capitalize()}: {error}")

            return render(request, 'your_template.html', {
                'student_form': student_form,
                'grades_form': grades_form
            })
    
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
        return redirect("student_record")

    else:
        messages.error(request, "Failed to delete student record!")
        return redirect("student_record")

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

    # messages.success(request, f"Prediction complete: {predicted_label}")

    messages.success(request, f"Prediction complete: Student {pk} - {predicted_label}")
    return redirect("student_record")

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64

@login_required(login_url="/login/")
def modelEvaluation(request):
    # Active school year
    current_sy = SchoolYear.objects.filter(is_current=True).first()
    if not current_sy:
        messages.info(request, "No active school year set.")
        return render(request, 'predictor/model_evaluation.html', {"no_schoolyear": True})

    # Load model
    model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'logreg_balanced.pkl')
    model = joblib.load(model_path)

    # Students in active school year with actual & predicted tracks
    students = Student.objects.filter(sy=current_sy) \
        .exclude(actual_track__isnull=True).exclude(actual_track__exact="") \
        .exclude(predicted_track__isnull=True).exclude(predicted_track__exact="") \
        .values("student_id", "actual_track", "predicted_track", "age", "gender")

    if not students:
        return render(request, 'predictor/model_evaluation.html', {"no_data": True, "sy": current_sy})

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
        return render(request, 'predictor/model_evaluation.html', {"no_data": True, "sy": current_sy})

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
    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Academic", "TVL"], yticklabels=["Academic", "TVL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_base64 = plot_to_base64()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba)
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_base64 = plot_to_base64()

    for label, metrics in report.items(): 
        if isinstance(metrics, dict): 
            if "f1-score" in metrics: metrics["f1_score"] = metrics.pop("f1-score")

    context = {
        "count": count,
        "accuracy": round(accuracy * 100, 2),
        "roc_auc": round(roc_auc, 3),
        "cm_base64": cm_base64,
        "roc_base64": roc_base64,
        "report": report,
        "no_data": False,
        "current_sy": current_sy,  # pass school year to template
    }

    return render(request, "predictor/model_evaluation.html", context)


@user_passes_test(admin_required, login_url="/login/")
def adminPanel(request):
    users = User.objects.all().order_by("id")
    school_years = SchoolYear.objects.all().order_by("-school_year")
    models = ModelTrainingHistory.objects.all().order_by('-trained_at')

    context = {'users': users, 'school_years': school_years, 'models':models}

    return render(request, 'predictor/admin_panel.html', context)

class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "first_name", "last_name", "email", "is_active", "is_staff"]

    def __init__(self, *args, **kwargs):
        current_user = kwargs.pop("current_user", None)  # pass logged-in user
        super().__init__(*args, **kwargs)

        # If not superuser, hide staff/active fields
        if not current_user or not current_user.is_superuser:
            self.fields.pop("is_active")
            self.fields.pop("is_staff")

@user_passes_test(admin_required, login_url="/login/")
def registerUser(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "User added successfully!")
            return redirect("admin_panel")

    else:
        form = CustomUserCreationForm()

    return render(request, "predictor/register_user.html", {"form": form})

@login_required
def updateUser(request, user_id):
    user = get_object_or_404(User, id=user_id)

    # Restrict: Superuser can edit any user; normal users only themselves
    if not request.user.is_superuser and request.user != user:
        return redirect("admin_panel")


    # Pass current_user to form so it knows what to hide
    update_form = UserUpdateForm(request.POST or None, instance=user, current_user=request.user)

    password_form = None
    if request.user == user:
        password_form = PasswordChangeForm(user, request.POST or None, prefix="pw")

    if request.method == "POST":
        if "update_profile" in request.POST and update_form.is_valid():
            update_form.save()
            messages.success(request, f"User {user} profile updated successfully!")
            return redirect("admin_panel")

        if "change_password" in request.POST and password_form and password_form.is_valid():
            user = password_form.save()
            update_session_auth_hash(request, user)
            messages.success(request, f"User {user} password updated successfully!")
            return redirect("admin_panel")
        
    context = {'update_form': update_form, 'password_form': password_form, 'user_obj': user,}

    return render(request, "predictor/update_user.html", context)

@user_passes_test(admin_required, login_url="/login/")
def deleteUser(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == "POST":
        user.delete()
        messages.success(request, "User deleted successfully!")
        return redirect("admin_panel")
    
    context = {'user': user}

    return render(request, "predictor/delete_user.html", context)

@user_passes_test(admin_required, login_url="/login/")
def school_year_list(request):
    school_years = SchoolYear.objects.all().order_by("-sy_id")

    context = {'school_years': school_years}

    return render(request, "predictor/school_year_list.html", context)

@user_passes_test(admin_required, login_url="/login/")
def add_school_year(request):
    if request.method == "POST":
        form = SchoolYearForm(request.POST)
        if form.is_valid():
            if form.cleaned_data.get("is_current"):
                SchoolYear.objects.update(is_current=False)  # reset others
            form.save()
            messages.success(request, "School Year added successfully!")
            return redirect("admin_panel")
    else:
        form = SchoolYearForm()
    return render(request, "predictor/add_school_year.html", {"form": form})

@user_passes_test(admin_required, login_url="/login/")
def edit_school_year(request, pk):
    sy = get_object_or_404(SchoolYear, pk=pk)
    if request.method == "POST":
        form = SchoolYearForm(request.POST, instance=sy)
        if form.is_valid():
            if form.cleaned_data.get("is_current"):
                SchoolYear.objects.update(is_current=False)
            form.save()
            messages.success(request, "School Year detail updated successfully!")
            return redirect("admin_panel")
    else:
        form = SchoolYearForm(instance=sy)
    return render(request, "predictor/edit_school_year.html", {"form": form, "sy": sy})

@user_passes_test(admin_required, login_url="/login/")
def delete_school_year(request, pk):
    sy = get_object_or_404(SchoolYear, pk=pk)
    if request.method == "POST":
        sy.delete()
        messages.success(request, "School Year deleted successfully!")
        return redirect("admin_panel")
    context = {'sy':sy}
    return render(request, "predictor/delete_school_year.html", context)

@user_passes_test(admin_required, login_url="/login/")
def setCurrentYear(request, sy_id):
    # Reset all years
    SchoolYear.objects.update(is_current=False)
    
    # Set selected year as current 
    year = get_object_or_404(SchoolYear, pk=sy_id)
    year.is_current = True
    year.save()

    messages.success(request, f"School Year set to {year}")

    return redirect("admin_panel") 

@user_passes_test(admin_required, login_url="/login/")
def trainModel(request):
    if request.method == 'POST':
        try:
            result_message = train_model()
            messages.success(request, result_message)
        except Exception as e:
            messages.error(request, f"❌ Training failed: {e}")
        return redirect("admin_panel") 

    # Handle GET — show model info table
    models = ModelTrainingHistory.objects.all().order_by('-trained_at')

    context = {'models': models}

    return render(request, "predictor/train_model.html", context)