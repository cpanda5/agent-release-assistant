"""FastAPI dependency: resolve the current user from a Bearer token."""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.security import decode_token

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Return ``{"user_uuid": str, "username": str}`` or raise 401."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_uuid": payload["sub"], "username": payload["username"]}
