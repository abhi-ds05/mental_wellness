from fastapi import APIRouter
from pydantic import BaseModel

# Try to import your profile-related functions
try:
    from user_profile.mood_history import get_user_mood_history
except ImportError:
    get_user_mood_history = None

try:
    from user_profile.user_embeddings import get_user_profile_vector
except ImportError:
    get_user_profile_vector = None

router = APIRouter()

# ===== Request Models =====
class UserIDRequest(BaseModel):
    user_id: str

# ===== Endpoints =====
@router.post("/user/history")
def get_history(req: UserIDRequest):
    """
    Returns the mood history for the specified user.
    """
    if not get_user_mood_history:
        return {"error": "get_user_mood_history function not found in user_profile.mood_history"}

    try:
        history = get_user_mood_history(req.user_id)
    except Exception as e:
        return {"error": str(e)}

    return {"user_id": req.user_id, "mood_history": history}


@router.post("/user/profile_vector")
def get_profile_vector(req: UserIDRequest):
    """
    Returns the profile embedding vector for the user.
    """
    if not get_user_profile_vector:
        return {"error": "get_user_profile_vector function not found in user_profile.user_embeddings"}

    try:
        vector = get_user_profile_vector(req.user_id)
    except Exception as e:
        return {"error": str(e)}

    return {"user_id": req.user_id, "embedding_vector": vector}
