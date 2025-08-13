from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('students_records/', views.studentsRecord, name='students_record'),

    path('add_record/add/', views.addStudentRecord, name="add_record"),
    path('view_record/<int:pk>/view/', views.viewStudentRecord, name="view_record"),
    path('update_record/<int:pk>/edit/', views.updateStudentRecord, name="update_record"),
    path('delete_record/<int:pk>/', views.deleteStudentRecord, name='delete_record'),

    # path('predict_track/', views.predictTrackList, name='predict_track'),
    path('predict_track/<int:pk>/', views.predictStudentTrack, name='predict_student_track'),

    path('model_evaluation/', views.modelEvaluation, name='model_evaluation'),

    path('admin_panel/', views.adminPanel, name='admin_panel'),


]