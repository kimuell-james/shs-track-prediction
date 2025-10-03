from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.loginPage, name='login'),
    path('logout/', views.logoutUser, name='logout'),

    path('', views.home, name='home'),
    
    path('student_record/', views.studentsRecord, name='student_record'),

    path('student_record/add/', views.addStudentRecord, name='add_record'),
    path('student_record/view/<int:pk>/', views.viewStudentRecord, name='view_record'),
    path('student_record/update/<int:pk>/', views.updateStudentRecord, name='update_record'),
    path('student_record/delete/<int:pk>/', views.deleteStudentRecord, name='delete_record'),
    path('student_record/predict/<int:pk>/', views.predictStudentTrack, name='predict_track'),

    path('model_evaluation/', views.modelEvaluation, name='model_evaluation'),

    path('admin_panel/', views.adminPanel, name='admin_panel'),
    path('admin_panel/register_user/', views.registerUser, name='register_user'),
    path('admin_panel/users/update/<int:user_id>/', views.updateUser, name='update_user'),
    path('admin_panel/users/delete/<int:user_id>/', views.deleteUser, name='delete_user'),
    # path('admin_panel/admin_list/', views.adminList, name='admin_list'),

    path('admin_panel/school_years/', views.school_year_list, name='school_year_list'),
    path('admin_panel/school_years/add/', views.add_school_year, name='add_school_year'),
    path('admin_panel/school_years/edit/<int:pk>/', views.edit_school_year, name='edit_school_year'),
    path('admin_panel/school_years/delete/<int:pk>/', views.delete_school_year, name='delete_school_year'),
    path('admin_panel/set_current_year/<int:sy_id>/', views.setCurrentYear, name='set_current_year'),

]
