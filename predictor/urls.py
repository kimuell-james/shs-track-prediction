from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('students_records/', views.studentsRecord, name='students_record'),

    path('predict_track/', views.predictTrack, name='predict_track'),

    path('model_evaluation/', views.modelEvaluation, name='model_evaluation'),

    path('admin_panel/', views.adminPanel, name='admin_panel'),


]