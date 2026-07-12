"""Authenticated self-service profile endpoints."""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.models.models import User
from app.utils.database import get_db

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads" / "profiles"
ALLOWED_IMAGES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_IMAGE_BYTES = 3 * 1024 * 1024


class ProfileUpdate(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    address: str | None = Field(default=None, max_length=1000)
    phone: str | None = Field(default=None, max_length=50)


def _profile(user: User) -> dict:
    return {
        "id": user.id,
        "full_name": user.name,
        "email": user.email,
        "role": user.role,
        "address": user.address,
        "phone": user.phone,
        "profile_image_url": user.profile_image_url,
    }


@router.get("/me")
def get_my_profile(current_user: User = Depends(get_current_user)):
    return _profile(current_user)


@router.patch("/me")
def update_my_profile(
    body: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.name = body.full_name.strip()
    current_user.address = body.address.strip() if body.address else None
    current_user.phone = body.phone.strip() if body.phone else None
    db.commit()
    db.refresh(current_user)
    return _profile(current_user)


@router.post("/me/avatar")
async def upload_my_avatar(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    extension = ALLOWED_IMAGES.get(file.content_type or "")
    if not extension:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WebP image.")
    content = await file.read()
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Profile image must be between 1 byte and 3MB.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{current_user.id}_{uuid4().hex}{extension}"
    (UPLOAD_DIR / filename).write_bytes(content)
    current_user.profile_image_url = f"/uploads/profiles/{filename}"
    db.commit()
    return _profile(current_user)
