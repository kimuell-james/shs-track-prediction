import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from predictor.models import Student, StudentGrade, SchoolYear
from pathlib import Path
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Import student data and grades from CSV'

    def handle(self, *args, **kwargs):
        # csv_path = os.path.join('shs_track_prediction', 'data', 'csv file here')
        csv_path = Path(settings.BASE_DIR) / 'data' / 'shs-data-trainset.csv'

        # Get the current active school year
        sy_obj = SchoolYear.objects.filter(is_current=True).first()

        with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')  
            for row in reader:
                print(row.keys())  # shows all column names from CSV
                break
            for row in reader:
                student = Student.objects.create(
                    age=int(row['age']),
                    gender=row['gender'],
                    grade_level=11,
                    sy=sy_obj,
                    actual_track=row['track'],        # Keep this; it's the only available value
                    predicted_track=None,             # Set as None for now
                    contributing_subjects=None,
                    created_at=datetime.now(),
                    predicted_at=None
                )

                # Create student grade
                StudentGrade.objects.create(
                    student_id=student,
                    g7_filipino=float(row['g7_filipino']),
                    g7_english=float(row['g7_english']),
                    g7_math=float(row['g7_math']),
                    g7_science=float(row['g7_science']),
                    g7_ap=float(row['g7_ap']),
                    g7_tle=float(row['g7_tle']),
                    g7_mapeh=float(row['g7_mapeh']),
                    g7_esp=float(row['g7_esp']),
                    g8_filipino=float(row['g8_filipino']),
                    g8_english=float(row['g8_english']),
                    g8_math=float(row['g8_math']),
                    g8_science=float(row['g8_science']),
                    g8_ap=float(row['g8_ap']),
                    g8_tle=float(row['g8_tle']),
                    g8_mapeh=float(row['g8_mapeh']),
                    g8_esp=float(row['g8_esp']),
                    g9_filipino=float(row['g9_filipino']),
                    g9_english=float(row['g9_english']),
                    g9_math=float(row['g9_math']),
                    g9_science=float(row['g9_science']),
                    g9_ap=float(row['g9_ap']),
                    g9_tle=float(row['g9_tle']),
                    g9_mapeh=float(row['g9_mapeh']),
                    g9_esp=float(row['g9_esp']),
                    g10_filipino=float(row['g10_filipino']),
                    g10_english=float(row['g10_english']),
                    g10_math=float(row['g10_math']),
                    g10_science=float(row['g10_science']),
                    g10_ap=float(row['g10_ap']),
                    g10_tle=float(row['g10_tle']),
                    g10_mapeh=float(row['g10_mapeh']),
                    g10_esp=float(row['g10_esp']),
                )

        self.stdout.write(self.style.SUCCESS('Student and grade data imported successfully.'))
