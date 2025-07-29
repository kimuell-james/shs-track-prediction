from django.db import models

# Create your models here.

class SchoolYear(models.Model):
    sy_id = models.AutoField(primary_key=True)
    school_year = models.CharField(max_length=50)

    def __str__(self):
        return self.school_year

class Student(models.Model):
    student_id = models.AutoField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=50)
    grade_level = models.IntegerField()
    sy = models.ForeignKey(SchoolYear, on_delete=models.CASCADE)
    predicted_track = models.CharField(max_length=50, blank=True, null=True)
    actual_track = models.CharField(max_length=50)
    important_subject = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField()
    predicted_at = models.DateTimeField(blank=True, null=True)

class Admin(models.Model):
    admin_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)  # Tip: Use Django's auth system instead
    email = models.CharField(max_length=50)

class StudentGrade(models.Model):
    grade_id = models.AutoField(primary_key=True)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    grade_level = models.IntegerField()
    subject_name = models.CharField(max_length=50)
    grade_value = models.FloatField()
