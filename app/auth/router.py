"""Auth endpoints: register / login / me."""
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pymysql.err import IntegrityError

from app.auth.deps import get_current_user
from app.auth.security import create_token, hash_password, verify_password
from app.db import get_db_pool
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_USER_COLS = "uuid, username, display_name, email, created_at"


def _require_pool():
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


def _user_out(row) -> UserOut:
    return UserOut(
        uuid=row[0], username=row[1], display_name=row[2], email=row[3], created_at=row[4]
    )


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest) -> TokenResponse:
    """Create a user and return a Bearer token (auto-login after register)."""
    pool = _require_pool()
    user_uuid = str(uuid4())
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"INSERT INTO `user` (uuid, username, display_name, email, password_hash) "
                    f"VALUES (%s, %s, %s, %s, %s)",
                    (
                        user_uuid,
                        request.username,
                        request.display_name,
                        request.email,
                        hash_password(request.password),
                    ),
                )
                await conn.commit()
                await cur.execute(
                    f"SELECT {_USER_COLS} FROM `user` WHERE uuid = %s", (user_uuid,)
                )
                row = await cur.fetchone()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Username or email already exists")

    token, expires_at = create_token(row[0], row[1])
    return TokenResponse(
        access_token=token, expires_at=expires_at, user=_user_out(row)
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_USER_COLS}, password_hash, status "
                f"FROM `user` WHERE username = %s AND deleted_at IS NULL",
                (request.username,),
            )
            row = await cur.fetchone()

    if row is None or not verify_password(request.password, row[5]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if row[6] != 1:
        raise HTTPException(status_code=403, detail="Account disabled")

    token, expires_at = create_token(row[0], row[1])
    return TokenResponse(
        access_token=token, expires_at=expires_at, user=_user_out(row[:5])
    )


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)) -> UserOut:
    pool = _require_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_USER_COLS} FROM `user` WHERE uuid = %s AND deleted_at IS NULL",
                (user["user_uuid"],),
            )
            row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return _user_out(row)
