import os
from dotenv import load_dotenv


AWS_S3_BUCKET_NAME = "wafer-sensor-fault"
MONGO_DATABASE_NAME = "ML"

MONGO_COLLECTION_NAME = "Wafer_fault"
TARGET_COLUMN = "quality"

load_dotenv()
MONGODB_URL=os.getenv("MONGODB_URL")
MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"