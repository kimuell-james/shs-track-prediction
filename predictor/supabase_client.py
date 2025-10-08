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
    supabase_url = os.getenv("SUPABASE_URL") or getattr(settings, "SUPABASE_URL", "")

    try:
        with open(local_path, "rb") as f:
            response = supabase.storage.from_(bucket_name).upload(
                file=f,
                path=file_name,
                file_options={"cache-control": "3600", "upsert": "true"}
            )

        if hasattr(response, "error") and response.error:
            print(f"❌ Upload failed for {file_name}: {response.error}")
            return None

        # ✅ Construct full public URL manually
        public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{file_name}"
        print(f"✅ Uploaded '{file_name}' successfully!")
        print(f"🌐 Public URL: {public_url}")

        # Optional: Verify upload success
        check = supabase.storage.from_(bucket_name).list()
        uploaded_files = [f['name'] for f in check]
        if file_name not in uploaded_files:
            print(f"⚠️ File '{file_name}' not found in Supabase bucket listing — upload may have failed silently.")

        return public_url

    except Exception as e:
        print(f"❌ Exception while uploading {file_name}: {e}")
        return None
