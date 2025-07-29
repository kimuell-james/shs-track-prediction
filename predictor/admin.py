from django.contrib import admin

# Register your models here.

from .models import *

admin.site.register(SchoolYear)
admin.site.register(Student)
admin.site.register(Admin)
admin.site.register(StudentGrade)