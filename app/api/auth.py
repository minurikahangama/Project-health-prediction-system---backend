"""
Authentication router.

Endpoints:
  POST /auth/login             — returns JWT access token
  POST /auth/change-password   — forced password change on first login

Shared dependencies exported for use by all other routers:
  get_current_user(token, db)  — validates JWT, returns User object
  require_role(*roles)         — factory that returns a role-checking dependency
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.utils.database import get_db
from app.models.models import User
from app.utils.time import utcnow
from datetime import timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address
import os

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# bcrypt cost factor 12 — deliberately slow to resist brute-force
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

SECRET_KEY = os.getenv("SECRET_KEY", "changeme_in_env")
ALGORITHM  = os.getenv("ALGORITHM", "HS256")
EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ── Token helpers ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    payload = data.copy()
    expire  = utcnow() + (expires_delta or timedelta(minutes=EXPIRE_MIN))
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Shared dependencies (imported by other routers) ───────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Validates the Bearer JWT, looks up the user in the database,
    and returns the User object.

    Raises HTTP 401 for any invalid / expired token.
    """
    credentials_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = db.query(User).filter(User.email == email).first()
    if user is None or not user.is_active:
        raise credentials_error
    return user


def require_role(*roles: str):
    """
    Returns a FastAPI dependency that enforces role-based access control.

    Usage:
        @router.get("/admin-only")
        def admin_endpoint(user: User = Depends(require_role("super_admin"))):
            ...
    """
    def _checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {list(roles)}",
            )
        return user
    return _checker


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login")
@limiter.limit("5/15minutes")   # NFR-12: rate-limit brute-force attempts
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate a user and return a JWT access token.

    Accepts standard OAuth2 form body (username = email address).
    Rate-limited to 5 attempts per 15 minutes per IP.
    """
    user = db.query(User).filter(User.email == form_data.username).first()

    # Verify password — use constant-time comparison to prevent timing attacks
    if not user or not pwd_ctx.verify(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    # Update last login timestamp
    user.last_login = utcnow()
    db.commit()

    token = create_access_token({
        "sub":                  user.email,
        "role":                 user.role,
        "org_id":               user.org_id,
        "force_password_change": user.force_password_change,
    })

    return {
        "access_token":          token,
        "token_type":            "bearer",
        "role":                  user.role,
        "name":                  user.name,
        "force_password_change": user.force_password_change,
    }


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Allows a user to change their password.
    Called on first login when force_password_change is True.
    """
    # Verify the current/temporary password
    if not pwd_ctx.verify(body.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Enforce new password strength
    pwd = body.new_password
    if len(pwd) < 8:
        raise HTTPException(status_code=400,
                            detail="New password must be at least 8 characters")
    if not any(c.isupper() for c in pwd):
        raise HTTPException(status_code=400,
                            detail="New password must include an uppercase letter")
    if not any(c.isdigit() for c in pwd):
        raise HTTPException(status_code=400,
                            detail="New password must include a number")
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in pwd):
        raise HTTPException(status_code=400,
                            detail="New password must include a special character")
    if pwd == body.current_password:
        raise HTTPException(status_code=400,
                            detail="New password must differ from the current one")

    # Save new hash
    current_user.password_hash        = pwd_ctx.hash(pwd)
    current_user.force_password_change = False
    db.commit()

    # Issue a new token without the force_password_change flag
    token = create_access_token({
        "sub":                  current_user.email,
        "role":                 current_user.role,
        "org_id":               current_user.org_id,
        "force_password_change": False,
    })

    return {"access_token": token, "token_type": "bearer"}
