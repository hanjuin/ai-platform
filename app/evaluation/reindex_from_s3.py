import os
import sys
import boto3
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.db.session import SessionLocal
from app.models.db_models import Document, User
from app.routers.documents import generate_and_store_embedding

BUCKET = os.getenv("S3_BUCKET_NAME", "").replace("s3://", "")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client("s3", region_name=AWS_REGION)

def list_s3_documents():
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix="document/")
    
    if "Contents" not in response:
        print("No files found in S3 under document/ prefix.")
        return []
    
    return response["Contents"]

def extract_filename(s3_key:str) -> str:
    basename = s3_key.split("/")[-1]
    filename = basename.split("_", 1)[-1]
    return filename

def reindex_from_s3():
    db = SessionLocal()
    
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("No admin found")
            return

        files = list_s3_documents()
        if not files:
            return

        print(f"Found {len(files)} files in s3. Indexing...")

        for obj in files:
            s3_key = obj["Key"]
            filename = extract_filename(s3_key)
            
            document = Document(
                filename=filename,
                s3_key=s3_key,
                owner_id=admin.user_id
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            print(f"    [{document.document_id}] {filename} - embedding...")
            
            generate_and_store_embedding(document.document_id)
            
            print(f"    [{document.document_id}] {filename} - done.")

        print("\nReindex complete.")
    
    finally:
        db.close()
    
if __name__ == "__main__":
    reindex_from_s3()
