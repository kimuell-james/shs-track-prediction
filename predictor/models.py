from django.db import models

# Create your models here.
    
class SchoolYear(models.Model):
    sy_id = models.AutoField(primary_key=True)
    school_year = models.CharField(max_length=50, unique=True)
    is_current = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        if self.is_current:
            # Reset all other years to False
            SchoolYear.objects.exclude(pk=self.pk).update(is_current=False)
        super().save(*args, **kwargs)

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
    contributing_subjects = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    predicted_at = models.DateTimeField(blank=True, null=True)

    def save(self, *args, **kwargs):
        # Assign the current school year if not set
        if not self.sy_id:
            current_sy = SchoolYear.objects.filter(is_current=True).first()
            if current_sy:
                self.sy = current_sy

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
