import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from predictor.models import Student, StudentGrade, SchoolYear
import os

class Command(BaseCommand):
    help = 'Import student data and grades from CSV'

    def handle(self, *args, **kwargs):
        csv_path = os.path.join('predictor', 'data', 'shs_prediction_testset.csv')  # adjust if needed

        # Make sure school year exists
        sy_obj, _ = SchoolYear.objects.get_or_create(school_year="2023-2024")

        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')  # use ',' if needed

            for row in reader:
                student = Student.objects.create(
                age=int(row['age']),
                gender=row['gender'],
                grade_level=11,
                sy=sy_obj,
                actual_track=row['track'],        # Keep this; it's the only available value
                predicted_track=None,             # Set as None for now
                important_subject=None,
                created_at=datetime.now(),
                predicted_at=None
            )

                for key, value in row.items():
                    if key.startswith("g") and "_" in key:
                        parts = key.split('_')
                        if len(parts) == 2:
                            grade_level = int(parts[0][1:])  # g7 → 7
                            subject_name = parts[1].strip()
                            try:
                                grade_value = float(value)
                            except ValueError:
                                continue

                            StudentGrade.objects.create(
                                student=student,
                                grade_level=grade_level,
                                subject_name=subject_name.capitalize(),
                                grade_value=grade_value
                            )

        self.stdout.write(self.style.SUCCESS('Student and grade data imported successfully.'))
