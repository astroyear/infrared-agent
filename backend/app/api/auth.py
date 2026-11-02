import hashlib
import hmac
import os
from datetime import timedelta
import jwt
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from ..core import Session, fail, now
from ..models import User

router = APIRouter(prefix="/api/auth")

def session():
    with Session() as db:
        yield db

class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=512)

def secret():
    value = os.getenv("JWT_SECRET", "")
    if len(value.encode()) < 32:
        raise RuntimeError("请在 .env 中设置至少 32 字节的 JWT_SECRET")
    return value

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600000).hex()

def public(user):
    return {"id": user.id, "username": user.username}

@router.post("/register", status_code=201)
def register(body: Credentials, db=Depends(session)):
    name = body.username.strip().casefold()
    if not name:
        fail(422, "用户名不能为空")
    salt = os.urandom(16).hex()
    user = User(username=name, password=salt + ":" + hash_password(body.password, salt))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        fail(409, "用户名已存在")
    return public(user)

@router.post("/login")
def login(body: Credentials, db=Depends(session)):
    user = db.scalar(select(User).where(User.username == body.username.strip().casefold()))
    if not user:
        fail(401, "用户名或密码错误")
    salt, expected = user.password.split(":")
    if not hmac.compare_digest(hash_password(body.password, salt), expected):
        fail(401, "用户名或密码错误")
    ttl = int(os.getenv("JWT_EXPIRE_MINUTES", "1440")) * 60
    issued = now()
    token = jwt.encode({"sub": user.id, "iat": issued, "exp": issued + timedelta(seconds=ttl)},
                       secret(), algorithm="HS256")
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl, "user": public(user)}

def current(credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
            db=Depends(session)):
    try:
        claims = jwt.decode(credentials.credentials if credentials else "", secret(), algorithms=["HS256"],
                            options={"require": ["sub", "iat", "exp"]})
    except jwt.InvalidTokenError:
        fail(401, "请重新登录")
    user = db.get(User, claims["sub"])
    if not user:
        fail(401, "请重新登录")
    return user

@router.get("/me")
def me(user=Depends(current)):
    return public(user)
