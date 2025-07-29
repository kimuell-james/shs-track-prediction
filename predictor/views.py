from django.shortcuts import render
from .models import *

def home(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/dashboard.html', context)

def studentsRecord(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/students_record.html', context)

def predictTrack(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/predict_track.html', context)

def modelEvaluation(request):
    records = Student.objects.all()

    context = {'records': records}

    return render(request, 'predictor/model_evaluation.html', context)

def adminPanel(request):
    admin = Admin.objects.all()

    context = {'admin': admin}

    return render(request, 'predictor/admin_panel.html', context)
