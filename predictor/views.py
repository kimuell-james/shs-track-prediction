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
from .evaluate_model import evaluate_active_model
from .decorators import unauthenticated_user
from .constants import SUBJECT_LABELS

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
    # --- 1Get current school year ---
    current_sy = SchoolYear.objects.filter(is_current=True).first()
    if not current_sy:
        messages.info(request, "No active school year set.")
        return render(request, 'predictor/dashboard.html', {})

    # --- Get selected grade from dropdown ---
    selected_grade = request.GET.get("grade_level", "g10")  # default to G10

    # --- Filter StudentGrades for students in current SY ---
    data = list(
        StudentGrade.objects.filter(student_id__sy=current_sy)
        .values("student_id__actual_track",
                *[f"g{i}_{subj}" for i in range(7, 11) for subj in [
                    "filipino", "english", "math", "science", "ap", "tle", "mapeh", "esp"
                ]])
    )

    df = pd.DataFrame(data)
    if df.empty:
        return render(request, "predictor/dashboard.html", {
            "error": f"No student grades found for school year {current_sy.school_year}."
        })

    # Rename track column for convenience
    df.rename(columns={"student_id__actual_track": "track"}, inplace=True)

    # --- Melt and preprocess ---
    grade_cols = [col for col in df.columns if col.startswith("g")]
    long_df = df.melt(id_vars=["track"], value_vars=grade_cols,
                      var_name="subject", value_name="grade")
    long_df["grade_level"] = long_df["subject"].str.extract(r"(g\d+)")
    long_df["subject_name"] = long_df["subject"].str.replace(r"g\d+_", "", regex=True)

    # ---  Filter by selected grade ---
    filtered_df = long_df[long_df["grade_level"] == selected_grade]

    SUBJECT_DISPLAY = {
        "filipino": "Filipino",
        "english": "English",
        "math": "Math",
        "science": "Science",
        "ap": "AP",
        "tle": "TLE",
        "mapeh": "MAPEH",
        "esp": "ESP"
    }

    # --- Compute average per subject per track ---
    avg_table = (
        filtered_df.groupby(["subject_name", "track"])["grade"]
        .mean()
        .unstack(fill_value=0)
        .round(2)
        .reset_index()
    )

    avg_table["subject_name"] = avg_table["subject_name"].map(SUBJECT_DISPLAY)

    # --- Prepare grade dropdown options ---
    grade_options = sorted(
        long_df["grade_level"].unique().tolist(),
        key=lambda x: int(x[1:])  # extract number after 'g' and sort
    )

    # --- Dashboard stats ---
    total_students = Student.objects.filter(sy=current_sy).count()
    predicted_count = Student.objects.filter(sy=current_sy, predicted_track__isnull=False).count()
    actual_count = Student.objects.filter(sy=current_sy, actual_track__isnull=False).count()

    # --- Distributions ---
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

    # --- Generate chart descriptions ---
    def generate_chart_descriptions():
        descriptions = {}
        if predicted_distribution:
            top_predicted = max(predicted_distribution, key=lambda x: x["count"])
            descriptions["predicted"] = (
                f"The most predicted track for {current_sy.school_year} is "
                f"{top_predicted['predicted_track']} with {top_predicted['count']} students."
            )
        else:
            descriptions["predicted"] = "No predicted track data available."

        if actual_distribution:
            top_actual = max(actual_distribution, key=lambda x: x["count"])
            descriptions["actual"] = (
                f"The most chosen actual track is {top_actual['actual_track']} "
                f"with {top_actual['count']} students enrolled."
            )
        else:
            descriptions["actual"] = "No actual track data available."

        if gender_distribution:
            descriptions["gender"] = (
                f"This chart shows the gender distribution per track for {current_sy.school_year}. "
                "It helps identify if a particular track is more popular among male or female students."
            )
        else:
            descriptions["gender"] = "No gender distribution data available."

        if age_distribution:
            descriptions["age"] = (
                f"This chart visualizes the age distribution of students across tracks for "
                f"{current_sy.school_year}, showing common age ranges per track."
            )
        else:
            descriptions["age"] = "No age distribution data available."

        return descriptions

    chart_descriptions = generate_chart_descriptions()

    # --- Prepare context ---
    context = {
        "avg_table": avg_table.to_dict(orient="records"),
        "columns": avg_table.columns.tolist(),
        "grade_options": grade_options,
        "selected_grade": selected_grade,
        "total_students": total_students,
        "predicted_count": predicted_count,
        "actual_count": actual_count,
        "predicted_distribution": json.dumps(list(predicted_distribution)),
        "actual_distribution": json.dumps(list(actual_distribution)),
        "gender_distribution": json.dumps(list(gender_distribution)),
        "age_distribution": json.dumps(list(age_distribution)),
        "chart_descriptions": chart_descriptions,
        "current_sy": current_sy.school_year,  # string for display in template
    }

    return render(request, "predictor/dashboard.html", context)


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

    # Apply subject labels mapping
    for field_name, label in SUBJECT_LABELS.items():
        if field_name in grades_form.fields:
            grades_form.fields[field_name].label = label

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

        # Apply subject labels mapping
        for field_name, label in SUBJECT_LABELS.items():
            if field_name in grades_form.fields:
                grades_form.fields[field_name].label = label

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

    # Apply subject labels mapping
    for field_name, label in SUBJECT_LABELS.items():
        if field_name in grades_form.fields:
            grades_form.fields[field_name].label = label

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

    page = request.GET.get("page", 1)

    messages.success(request, f"Prediction complete: Student {pk} - {predicted_label}")
    return redirect(f"{reverse('student_record')}?page={page}#student-{student.student_id}")

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
        return render(request, "predictor/model_evaluation.html", {"no_schoolyear": True})

    evaluation = evaluate_active_model(current_sy)

    if evaluation.get("error"):
        messages.error(request, evaluation["error"])
        return render(request, "predictor/model_evaluation.html", {"no_data": True})

    if evaluation.get("no_data"):
        messages.info(request, "No student data available for evaluation.")
        return render(request, "predictor/model_evaluation.html", {"no_data": True, "sy": current_sy})

    context = {
        "count": evaluation["count"],
        "accuracy": round(evaluation["accuracy"] * 100, 2),
        "roc_auc": round(evaluation["roc_auc"], 3),
        "cm_base64": evaluation["cm_base64"],
        "roc_base64": evaluation["roc_base64"],
        "report": evaluation["report"],
        "analysis": evaluation["analysis"],
        "no_data": evaluation["no_data"],
        "current_sy": current_sy,
    }

    return render(request, "predictor/model_evaluation.html", context)


@user_passes_test(admin_required, login_url="/login/")
def adminPanel(request):
    users = User.objects.all().order_by("id")
    school_years = SchoolYear.objects.all().order_by("-school_year")
    models = ModelTrainingHistory.objects.all().order_by('-trained_at')

    # ✅ Only include school years of students that qualify for training
    trainable_students = Student.objects.filter(actual_track__isnull=False)
    student_count = trainable_students.count()
    included_school_years = (
        SchoolYear.objects.filter(sy_id__in=trainable_students.values_list("sy_id", flat=True))
        .values_list("school_year", flat=True)
        .distinct()
    )
    included_school_years_text = ", ".join(included_school_years)

    # Paginate school years
    sy_paginator = Paginator(school_years, 5)
    sy_page_number = request.GET.get("sy_page")
    sy_page_obj = sy_paginator.get_page(sy_page_number)

    # Paginate users
    user_paginator = Paginator(users, 5)
    user_page_number = request.GET.get("user_page")
    user_page_obj = user_paginator.get_page(user_page_number)

    # Paginate model history (optional)
    model_paginator = Paginator(models, 5)
    model_page_number = request.GET.get("model_page")
    model_page_obj = model_paginator.get_page(model_page_number)

    context = {
        "school_years": sy_page_obj,
        "users": user_page_obj,
        "models": model_page_obj,
        "student_count": student_count,
        "included_school_years_text": included_school_years_text,
    }

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

# @user_passes_test(admin_required, login_url="/login/")
# def userList(request):
#     users = User.objects.all().order_by("id")

#     # Paginate users
#     user_paginator = Paginator(users, 5)
#     user_page_number = request.GET.get("user_page")
#     user_page_obj = user_paginator.get_page(user_page_number)


#     context = {'users': user_page_obj}

#     return render(request, "predictor/user_list.html", context)

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
    else:
        messages.error(request, "Failed to delete user profile.")
        return redirect("admin_panel")

# @user_passes_test(admin_required, login_url="/login/")
# def school_year_list(request):
#     school_years = SchoolYear.objects.all().order_by("-sy_id")

#     # Paginate school years
#     sy_paginator = Paginator(school_years, 5)
#     sy_page_number = request.GET.get("sy_page")
#     sy_page_obj = sy_paginator.get_page(sy_page_number)

#     context = {'school_years': sy_page_obj}

#     return render(request, "predictor/school_year_list.html", context)

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
    else:
        messages.error(request, "Failed to delete School Year.")
        return redirect("admin_panel")

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
    # models = ModelTrainingHistory.objects.all().order_by('-trained_at')

    # context = {'models': models}

    # return render(request, "predictor/train_model.html", context)