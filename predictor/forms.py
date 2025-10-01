from django import forms
from .models import *
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        exclude = ['student_id']
        fields = ["age", "gender", "sy", "grade_level", "predicted_track", "actual_track", "contributing_subjects"]  # include sy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            current_sy = SchoolYear.objects.get(is_current=True)
            self.fields["sy"].initial = current_sy
        except SchoolYear.DoesNotExist:
            pass


class StudentGradesForm(forms.ModelForm):
    class Meta:
        model = StudentGrade
        exclude = ['student_id']
        fields = '__all__'

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "password1", "password2")

class LoginForm(AuthenticationForm):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)