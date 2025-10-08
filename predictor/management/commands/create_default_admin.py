# predictor/management/commands/create_default_admin.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = "Creates a default admin if it does not exist"

    def handle(self, *args, **kwargs):
        username = "admin"
        email = "admin@example.com"
        password = "admin123"

        if not User.objects.filter(username=username).exists():
            User.objects.create_superuser(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f"✅ Default admin '{username}' created."))
        else:
            self.stdout.write(self.style.WARNING(f"⚠️ Default admin '{username}' already exists."))
