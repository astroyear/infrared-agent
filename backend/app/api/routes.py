from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, File as FileField, Form, Header, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from .. import approval
from ..core import STORAGE, fail, owned
from ..models import Document, File, Operation, Task, Thread, Turn, Version
from ..services import detections, knowledge, storage
from .auth import current, session

router = APIRouter(prefix="/api")
thread_locks = [Lock() for _ in range(128)]

@router.post("/files", status_code=201)
def files(file: UploadFile, purpose: Annotated[Literal["image", "knowledge"], Form()],
          user=Depends(current), db=Depends(session)):
    row = storage.upload(db, user.id, file, purpose)
    return {"file_id": row.id, "name": row.name, "purpose": row.purpose}

@router.post("/detections")
def run(response: Response, file: UploadFile | None = FileField(None), file_id: str | None = Form(None),
        conf: float = Form(0.25, ge=0, le=1), idempotency_key: str | None = Header(None, max_length=180),
        user=Depends(current), db=Depends(session)):
    if bool(file) == bool(file_id):
        fail(422, "file 与 file_id 须二选一")
    row = storage.upload(db, user.id, file, "image") if file else owned(db, File, file_id, user.id)
    task = detections.detect(db, user.id, row, conf, idempotency_key)
    response.status_code = 202 if task["status"] in ("pending", "running") else 201
    return task

@router.get("/detections/{task_id}")
def task(task_id: str, user=Depends(current), db=Depends(session)):
    return detections.info(owned(db, Task, task_id, user.id))

@router.get("/detections/{task_id}/{kind}")
def image(task_id: str, kind: Literal["original", "image"], user=Depends(current), db=Depends(session)):
    task = owned(db, Task, task_id, user.id)
    source = db.get(File, task.file_id)
    name = "input" + Path(source.key).suffix if kind == "original" else "result.jpg"
    path = STORAGE / "detections" / task.id / name
    if not path.is_file():
        fail(409, "图片尚未就绪")
    return FileResponse(path)

@router.post("/conversations", status_code=201)
def create_thread(user=Depends(current), db=Depends(session)):
    thread = Thread(owner_id=user.id, title="新会话")
    db.add(thread)
    db.commit()
    return {"thread_id": thread.id, "title": thread.title}

@router.get("/conversations")
def threads(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200),
            user=Depends(current), db=Depends(session)):
    rows = db.scalars(select(Thread).where(Thread.owner_id == user.id).order_by(Thread.created_at.desc())
                      .offset(offset).limit(limit))
    return [{"thread_id": r.id, "title": r.title} for r in rows]

@router.get("/conversations/{thread_id}/messages")
def messages(thread_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200),
             user=Depends(current), db=Depends(session)):
    owned(db, Thread, thread_id, user.id)
    rows = db.scalars(select(Turn).where(Turn.thread_id == thread_id).order_by(Turn.created_at)
                      .offset(offset).limit(limit))
    return [{"client_message_id": r.client_id, "message": r.question, **r.reply} for r in rows]

class Chat(BaseModel):
    thread_id: str
    client_message_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=16000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=1)
    detection_task_id: str | None = None

@router.post("/chat")
def chat(body: Chat, request: Request, user=Depends(current), db=Depends(session)):
    with thread_locks[hash(body.thread_id) % len(thread_locks)]:
        thread = owned(db, Thread, body.thread_id, user.id)
        attachments = list(body.attachment_ids)
        if body.detection_task_id:
            owned(db, Task, body.detection_task_id, user.id)
        for key in attachments:
            owned(db, File, key, user.id)
        refs = [*attachments, body.detection_task_id]
        old = db.scalar(select(Turn).where(Turn.thread_id == thread.id, Turn.client_id == body.client_message_id))
        if old:
            if old.question != body.message or old.attachments != refs:
                fail(409, "消息标识已用于其他内容")
            return old.reply
        state = {"thread": thread.id, "owner": user.id, "client_id": body.client_message_id,
                 "question": body.message, "files": attachments, "linked_task": body.detection_task_id, "reply": {}}
        recent = db.scalars(select(Turn).where(Turn.thread_id == thread.id).order_by(Turn.created_at.desc()))
        state["last_task"] = body.detection_task_id or next((r.reply.get("task_id") for r in recent if r.reply.get("task_id")), None)
        if not state["last_task"]:
            state["last_task"] = db.scalar(select(Task.id).where(Task.owner_id == user.id)
                                           .order_by(Task.created_at.desc(), Task.id.desc()).limit(1))
        db.commit()
        graph = request.app.state.graph
        config = {"configurable": {"thread_id": thread.id}}
        saved = graph.get_state(config)
        if saved.values.get("client_id") == body.client_message_id:
            if any(saved.values.get(k) != state[k] for k in ("question", "files", "linked_task")):
                fail(409, "消息标识已用于其他内容")
            result = graph.invoke(None, config) if saved.next else saved.values
        else:
            result = graph.invoke(state, config)
        reply = dict(result["reply"])
        if reply.get("operation_id"):
            approval.pause(request.app.state.review, reply["operation_id"])
        if body.detection_task_id:
            reply.setdefault("task_id", body.detection_task_id)
        if thread.title == "新会话":
            thread.title = body.message[:40]
        db.add(Turn(thread_id=thread.id, client_id=body.client_message_id, question=body.message,
                    attachments=refs, reply=reply))
        db.commit()
        return reply

@router.get("/documents")
def documents(user=Depends(current), db=Depends(session)):
    return [{"id": d.id, "title": d.title, "version": d.active}
            for d in db.scalars(select(Document).order_by(Document.title))]

@router.get("/documents/{doc_id}/versions")
def versions(doc_id: str, user=Depends(current), db=Depends(session)):
    if not db.get(Document, doc_id):
        fail(404, "文档不存在")
    return [{"id": v.id, "number": v.number, "created_at": v.created_at} for v in db.scalars(
        select(Version).where(Version.doc_id == doc_id).order_by(Version.number.desc()))]

@router.get("/documents/{doc_id}/versions/{version_id}")
def version(doc_id: str, version_id: str, user=Depends(current), db=Depends(session)):
    v = db.get(Version, version_id)
    if not v or v.doc_id != doc_id:
        fail(404, "版本不存在")
    return {"title": db.get(Document, doc_id).title, "number": v.number, "content": v.body}

@router.get("/knowledge/operations/{op_id}")
def operation(op_id: str, user=Depends(current), db=Depends(session)):
    return knowledge.preview(db, owned(db, Operation, op_id, user.id))

@router.post("/knowledge/operations/{op_id}/{action}")
def resolve(op_id: str, action: Literal["confirm", "cancel"], request: Request,
            user=Depends(current), db=Depends(session)):
    with thread_locks[hash(op_id) % len(thread_locks)]:
        op = owned(db, Operation, op_id, user.id)
        if op.status == "pending":
            action = approval.decision(request.app.state.review, op_id, action)
        return knowledge.resolve(db, user.id, op_id, action == "confirm")
