from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('students_records/', views.studentsRecord, name='students_record'),

    path('view_record/<str:pk>/', views.viewStudentRecord, name="view_record"),
    path('update_record/<str:pk>/', views.updateStudentRecord, name="update_record"),

    path('predict_track/', views.predictTrackList, name='predict_track'),
    path('predict_track/<int:student_id>/', views.predictStudentTrack, name='predict_student_track'),

    path('model_evaluation/', views.modelEvaluation, name='model_evaluation'),

    path('admin_panel/', views.adminPanel, name='admin_panel'),


]