from django.contrib import admin

# Register your models here.

from .models import *

@admin.register(SchoolYear)
class SchoolYearAdmin(admin.ModelAdmin):
    list_display = ("sy_id", "school_year", "is_current")
    list_editable = ("is_current",)
    ordering = ("-sy_id",)

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("student_id", "age", "gender", "grade_level", "predicted_track", "actual_track", "sy")
    search_fields = ("student_id", "actual_track", "predicted_track")
    list_filter = ("grade_level", "sy")
    list_per_page = 50   # 🔹 Show only 50 students per page
    # list_max_show_all = 100  # 🔹 Max when "Show all" is clicked

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        try:
            current_sy = SchoolYear.objects.get(is_current=True)
            form.base_fields["sy"].initial = current_sy  # pre-fill
        except SchoolYear.DoesNotExist:
            pass
        return form
    
    def save_model(self, request, obj, form, change):
        if not change:  # only when adding
            try:
                obj.sy = SchoolYear.objects.get(is_current=True)
            except SchoolYear.DoesNotExist:
                pass
        super().save_model(request, obj, form, change)

@admin.register(StudentGrade)
class StudentGradeAdmin(admin.ModelAdmin):
    list_display = (
    "grade_id",
    "student_id",
    # Grade 7
    "g7_filipino", "g7_english", "g7_math", "g7_science",
    "g7_ap", "g7_tle", "g7_mapeh", "g7_esp",
    # Grade 8
    "g8_filipino", "g8_english", "g8_math", "g8_science",
    "g8_ap", "g8_tle", "g8_mapeh", "g8_esp",
    # Grade 9
    "g9_filipino", "g9_english", "g9_math", "g9_science",
    "g9_ap", "g9_tle", "g9_mapeh", "g9_esp",
    # Grade 10
    "g10_filipino", "g10_english", "g10_math", "g10_science",
    "g10_ap", "g10_tle", "g10_mapeh", "g10_esp",
    )
    search_fields = ("student_id__student_id",)  # allows search by student_id
    list_filter = ("student_id__sy", "student_id__grade_level")  # filter by school year or grade level

# admin.site.register(SchoolYear)
# admin.site.register(Student, StudentAdmin)
# admin.site.register(StudentGrade)