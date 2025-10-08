import os
import io
import joblib
from supabase import create_client, Client
from django.conf import settings

# --- Initialize Supabase Client ---
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL") or getattr(settings, "SUPABASE_URL", None)
    key = os.getenv("SUPABASE_KEY") or getattr(settings, "SUPABASE_KEY", None)
    if not url or not key:
        raise ValueError("❌ Missing Supabase credentials. Please check your environment variables or settings.py.")
    return create_client(url, key)


# --- Upload a file to Supabase Storage ---
def upload_to_supabase(local_path: str, file_name: str) -> str:
    """
    Uploads a file to the Supabase 'ml_models' bucket and returns its public URL.
    """
    bucket_name = "ml_models"
    supabase = get_supabase_client()

    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload(file_name, f, {"upsert": True})

        # ✅ Get the public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
        print(f"✅ Uploaded {file_name} to Supabase. Public URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Failed to upload {file_name}: {e}")
        return None
