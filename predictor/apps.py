from django.apps import AppConfig
from django.contrib.auth import get_user_model
from django.db.utils import OperationalError


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        from django.conf import settings
        User = get_user_model()

        # Default admin credentials
        default_username = "admin"
        default_email = "admin@example.com"
        default_password = "admin123"

        try:
            if not User.objects.filter(username=default_username).exists():
                User.objects.create_superuser(
                    username=default_username,
                    email=default_email,
                    password=default_password
                )
                print(f"✅ Default superuser '{default_username}' created.")
            else:
                print(f"ℹ️ Default superuser '{default_username}' already exists.")
        except OperationalError:
            # This handles cases where the database isn't ready yet (e.g., during migrate)
            pass