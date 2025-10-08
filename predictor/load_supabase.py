import os
import joblib
import requests
import io
from django.conf import settings
from predictor.models import ModelTrainingHistory

# Utility function for URL-based loading
def load_model_from_url(url: str):
    """Load a joblib model directly from a public URL (Supabase)."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)
        print(f"✅ Loaded model from URL: {url}")
        return joblib.load(file_bytes)
    except Exception as e:
        raise ValueError(f"❌ Failed to load model from URL ({url}): {e}")

# --- MAIN LOAD LOGIC ---
def load_active_model():
    # Get active model record
    active_models = ModelTrainingHistory.objects.filter(is_active=True)
    if not active_models.exists():
        raise ValueError("⚠️ No active model found. Train one first.")
    elif active_models.count() > 1:
        raise ValueError("⚠️ Multiple active models found. Please keep only one active model active.")

    active_model = active_models.first()

    # ✅ If Supabase URLs exist — load directly from the web
    if active_model.model_url and active_model.scaler_url and active_model.columns_url:
        print("🔗 Loading model from Supabase URLs...")
        model = load_model_from_url(active_model.model_url)
        scaler = load_model_from_url(active_model.scaler_url)
        training_columns = load_model_from_url(active_model.columns_url)
        return model, scaler, training_columns

    # ⚙️ Otherwise, fallback to local directory (for local dev)
    print("📁 Loading model from local storage...")
    model_dir = os.path.join(settings.BASE_DIR, "predictor", "ml_models")
    model_path = os.path.join(model_dir, active_model.model_filename)
    scaler_path = os.path.join(model_dir, f"{active_model.model_filename}_scaler.pkl")
    columns_path = os.path.join(model_dir, f"{active_model.model_filename}_columns.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"⚠️ Model file not found at {model_path}. Please retrain the model.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    training_columns = joblib.load(columns_path)

    return model, scaler, training_columns
