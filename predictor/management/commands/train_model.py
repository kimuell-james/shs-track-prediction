from django.core.management.base import BaseCommand
import joblib
import os
import pandas as pd
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class Command(BaseCommand):
    help = 'Train and save unbalanced and balanced Logistic Regression models using the full dataset'

    def handle(self, *args, **kwargs):
        # Load dataset
        data_path = settings.BASE_DIR / 'data' / 'shs_prediction_trainset.csv'
        df = pd.read_csv(data_path)

        # Encode gender
        if 'gender' in df.columns:
            le_gender = LabelEncoder()
            df['gender'] = le_gender.fit_transform(df['gender'])

        # Encode target 
        if df['track'].dtype == object:
            le_track = LabelEncoder()
            df['track'] = le_track.fit_transform(df['track'])

        # Separate features and target
        X = df.drop(columns=['track'])
        y = df['track']

        # Directory to save the models
        model_dir = settings.BASE_DIR / 'predictor' / 'ml_models'
        os.makedirs(model_dir, exist_ok=True)

        # === UNBALANCED MODEL ===
        unbalanced_model = LogisticRegression(max_iter=200)
        unbalanced_model.fit(X, y)

        unbalanced_model_path = model_dir / 'logreg_unbalanced.pkl'
        joblib.dump(unbalanced_model, unbalanced_model_path)
        self.stdout.write(self.style.SUCCESS(f"Unbalanced model saved to {unbalanced_model_path}"))

        # === BALANCED MODEL ===
        balanced_model = LogisticRegression(max_iter=200, class_weight='balanced')
        balanced_model.fit(X, y)

        balanced_model_path = model_dir / 'logreg_balanced.pkl'
        joblib.dump(balanced_model, balanced_model_path)
        self.stdout.write(self.style.SUCCESS(f"Balanced model saved to {balanced_model_path}"))
