import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import BoundedSemaphore, Lock
from sqlalchemy import select, update
from ..core import STORAGE, Session, fail, uid
from ..models import Task
from .detector import Detector

detector = Detector()
executor = ThreadPoolExecutor(max_workers=1)
capacity = BoundedSemaphore(2)
submission = Lock()

def info(task):
    return {"id": task.id, "task_id": task.id, "status": task.status,
            "original_url": f"/api/detections/{task.id}/original",
            "image_url": f"/api/detections/{task.id}/image", **(task.result or {})}

def process(task_id, upload_key):
    try:
        with Session.begin() as db:
            task = db.get(Task, task_id)
            task.status = "running"
            conf = task.conf
        folder = STORAGE / "detections" / task_id
        folder.mkdir(parents=True)
        original = folder / ("input" + Path(upload_key).suffix)
        shutil.copyfile(STORAGE / upload_key, original)
        result = detector.predict(original, folder / "result.jpg", conf)
        with Session.begin() as db:
            task = db.get(Task, task_id)
            task.status, task.result = "succeeded", result
            temp = folder / "result.tmp"
            temp.write_text(json.dumps(info(task), ensure_ascii=False), encoding="utf8")
            temp.replace(folder / "result.json")
    except Exception:
        logging.exception("检测任务失败 %s", task_id)
        with Session.begin() as db:
            db.execute(update(Task).where(Task.id == task_id).values(status="failed"))
    finally:
        capacity.release()

def detect(db, owner, file, conf, key):
    if file.purpose != "image":
        fail(422, "需要图片附件")
    fingerprint = f"{file.sha256}:{conf}:{detector.digest}"
    with submission:
        if key:
            old = db.scalar(select(Task).where(Task.owner_id == owner, Task.key == key))
            if old:
                if old.request_hash != fingerprint:
                    fail(409, "幂等键已用于不同的请求")
                return info(old)
        if not capacity.acquire(blocking=False):
            fail(503, "检测服务繁忙，请稍后重试")
        try:
            task = Task(id=uid(), owner_id=owner, file_id=file.id, key=key, conf=conf,
                        request_hash=fingerprint, status="pending")
            db.add(task)
            db.commit()
            future = executor.submit(process, task.id, file.key)
        except Exception:
            capacity.release()
            raise
    try:
        future.result(timeout=15)
    except TimeoutError:
        pass
    db.refresh(task)
    return info(task)
