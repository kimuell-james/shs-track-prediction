from django.apps import AppConfig
from django.contrib.auth import get_user_model
from django.db.utils import OperationalError, ProgrammingError


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        from django.db.models.signals import post_migrate
        from django.contrib.auth.models import User

        def create_default_admin(sender, **kwargs):
            if not User.objects.filter(username='admin').exists():
                User.objects.create_superuser('admin', 'admin@example.com', 'admin123')

        post_migrate.connect(create_default_admin, sender=self)