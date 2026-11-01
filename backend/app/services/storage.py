import hashlib
import io
import os
from pathlib import Path
from sqlalchemy import select
from PIL import Image, UnidentifiedImageError
from ..core import STORAGE, fail, uid
from ..models import File

def upload(db, owner, file, purpose):
    limit = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 ** 2)))
    data = file.file.read(limit + 1)
    if len(data) > limit:
        fail(413, "文件过大")
    name = Path((file.filename or "").replace("\\", "/")).name
    ext = Path(name).suffix.lower()
    if purpose == "image" and ext in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != ("PNG" if ext == ".png" else "JPEG"):
                    fail(422, "图片扩展名与格式不符")
                if image.width * image.height > int(os.getenv("MAX_IMAGE_PIXELS", "20000000")):
                    fail(413, "图片像素过多")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            fail(422, "无法解码图片")
    elif purpose == "knowledge" and ext == ".md":
        try:
            if not data.decode("utf-8-sig").strip() or b"\x00" in data:
                fail(422, "Markdown 内容无效")
        except UnicodeError:
            fail(422, "Markdown 须为 UTF-8")
    else:
        fail(422, "图片使用 JPEG/PNG，知识文件使用 Markdown")
    digest = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(File).where(File.owner_id == owner, File.purpose == purpose,
                                            File.name == name, File.sha256 == digest))
    if existing:
        return existing
    key = f"uploads/{uid()}{ext}"
    path = STORAGE / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    row = File(owner_id=owner, purpose=purpose, name=name, key=key,
               sha256=digest)
    db.add(row)
    try:
        db.commit()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return row
