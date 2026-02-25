from fastapi import FastAPI

app = FastAPI(title="AI Document Intelligence API")

@app.get("/health")
def health_check():
    return {"status" : "ok"}