import os
import openai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to your training file
training_file_path = "openai_finetune_ready.jsonl"

# === Upload file ===
print("📤 Uploading training file...")
upload_response = openai.File.create(
    file=open(training_file_path, "rb"),
    purpose="fine-tune"
)

file_id = upload_response["id"]
print("✅ File uploaded. File ID:", file_id)

# === Start fine-tune job ===
print("🚀 Starting fine-tune job...")
fine_tune_response = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)

job_id = fine_tune_response["id"]
print("🛠 Fine-tune Job ID:", job_id)
print("⏳ Monitor progress with:\nopenai api fine_tunes.follow -i", job_id)
