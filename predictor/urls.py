from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.registerPage, name="register"),
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logoutUser, name="logout"),

    path('', views.home, name='home'),
    
    path('students_records/', views.studentsRecord, name='students_record'),

    path('add_record/add/', views.addStudentRecord, name="add_record"),
    path('view_record/<int:pk>/view/', views.viewStudentRecord, name="view_record"),
    path('update_record/<int:pk>/edit/', views.updateStudentRecord, name="update_record"),
    path('delete_record/<int:pk>/', views.deleteStudentRecord, name='delete_record'),

    path('predict_track/<int:pk>/predict', views.predictStudentTrack, name='predict_track'),

    path('model_evaluation/', views.modelEvaluation, name='model_evaluation'),

    path('admin_panel/', views.adminPanel, name='admin_panel'),
    path("users/update/<int:user_id>/", views.updateUser, name="update_user"),
    path("users/delete/<int:user_id>/", views.deleteUser, name="delete_user"),
    # path('admin_panel/admin_list/', views.adminList, name='admin_list'),


]