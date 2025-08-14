from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# === Import routers from your endpoints folder ===
from api.endpoints import chat, user, mood

# === Create the FastAPI app ===
app = FastAPI(
    title="Mental Wellness AI API",
    description="Unified backend API for the Mental Wellness AI system",
    version="1.0.0"
)

# === Enable CORS (for frontend connections) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Include endpoint routers ===
app.include_router(chat.router, tags=["Chat"])
app.include_router(user.router, tags=["User"])
app.include_router(mood.router, tags=["Mood"])

# === Health check endpoint ===
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Mental Wellness AI API is running"}

# === Run the server (when executed directly) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.fastapi_server:app", host="0.0.0.0", port=5000, reload=True)
