from django.db import models

# Create your models here.

class Admin(models.Model):
    admin_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)  # Tip: Use Django's auth system instead
    email = models.CharField(max_length=50)
    
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
    actual_track = models.CharField(max_length=50, blank=True, null=True)
    important_subject = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    predicted_at = models.DateTimeField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.student_id:
            last_student = Student.objects.order_by('-student_id').first()
            self.student_id = (last_student.student_id + 1) if last_student else 1
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['student_id']

    def __str__(self):
        return str(self.student_id)

class StudentGrade(models.Model):
    grade_id = models.AutoField(primary_key=True)
    student_id = models.ForeignKey(Student, on_delete=models.CASCADE)
    g7_filipino = models.FloatField()
    g7_english = models.FloatField()
    g7_math = models.FloatField()
    g7_science = models.FloatField()
    g7_ap = models.FloatField()
    g7_tle = models.FloatField()
    g7_mapeh = models.FloatField()
    g7_esp = models.FloatField()
    g8_filipino = models.FloatField()
    g8_english = models.FloatField()
    g8_math = models.FloatField()
    g8_science = models.FloatField()
    g8_ap = models.FloatField()
    g8_tle = models.FloatField()
    g8_mapeh = models.FloatField()
    g8_esp = models.FloatField()
    g9_filipino = models.FloatField()
    g9_english = models.FloatField()
    g9_math = models.FloatField()
    g9_science = models.FloatField()
    g9_ap = models.FloatField()
    g9_tle = models.FloatField()
    g9_mapeh = models.FloatField()
    g9_esp = models.FloatField()
    g10_filipino = models.FloatField()
    g10_english = models.FloatField()
    g10_math = models.FloatField()
    g10_science = models.FloatField()
    g10_ap = models.FloatField()
    g10_tle = models.FloatField()
    g10_mapeh = models.FloatField()
    g10_esp = models.FloatField()
