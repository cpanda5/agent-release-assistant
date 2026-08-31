"""
Auth primitives — password hashing + signed tokens, stdlib only.

Passwords use PBKDF2-SHA256 with a random salt (``hashlib.pbkdf2_hmac``).
Tokens are a compact HMAC-signed format (``<payload-b64>.<sig-b64>``) carrying
``{sub, username, exp}`` — enough for a simple login, no JWT dependency needed.
"""
import base64
import hashlib
import hmac
import json
import secrets
import time

from app.config import settings

_PBKDF2_ROUNDS = 260_000


# ── Password hashing ───────────────────────────────────────────────────────── #

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    )
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, salt, hex_hash = stored.split("$", 2)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    )
    return hmac.compare_digest(dk.hex(), hex_hash)


# ── Tokens ─────────────────────────────────────────────────────────────────── #

def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(body: str) -> str:
    return _b64e(
        hmac.new(settings.auth_secret_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    )


def create_token(user_uuid: str, username: str) -> tuple[str, float]:
    """Return ``(token, expires_at_unix_ts)``."""
    expires_at = time.time() + settings.token_expire_hours * 3600
    payload = {"sub": user_uuid, "username": username, "exp": int(expires_at)}
    body = _b64e(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body)}", expires_at


def decode_token(token: str) -> dict | None:
    """Return the payload dict, or None when the token is malformed / tampered / expired."""
    try:
        body, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(body), sig):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, KeyError):
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None
    return payload
