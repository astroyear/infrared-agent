import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
STORAGE = Path(os.getenv("STORAGE_ROOT", ROOT / "storage")).resolve()
URL = os.getenv("DATABASE_URL", "sqlite:///" + str(STORAGE / "app.sqlite3"))
if URL.startswith("postgresql://"):
    URL = URL.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(URL, pool_pre_ping=True)
POSTGRES = engine.dialect.name == "postgresql"
if engine.dialect.name == "sqlite":
    @event.listens_for(engine, "connect")
    def configure_sqlite(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
Session = sessionmaker(engine, expire_on_commit=False)

def uid():
    return str(uuid4())

def now():
    return datetime.now(timezone.utc)

def fail(code, message):
    raise HTTPException(code, message)

def owned(db, model, resource_id, owner):
    row = db.get(model, resource_id)
    if row is None or row.owner_id != owner:
        fail(404, "资源不存在")
    return row
