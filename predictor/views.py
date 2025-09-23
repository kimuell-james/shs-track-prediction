from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from .predict_track import predict_track_for_student 
from .models import *
from .forms import *

def home(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/dashboard.html', context)

def studentsRecord(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/students_record.html', context)

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


def modelEvaluation(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/model_evaluation.html', context)

def adminPanel(request):
    admin = Admin.objects.all()

    context = {'admin': admin}

    return render(request, 'predictor/admin_panel.html', context)

def adminList(request):
    admin = Admin.objects.all()

    context = {'admin': admin}

    return render(request, 'predictor/admin_list.html', context)
