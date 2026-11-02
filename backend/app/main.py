import logging
from contextlib import ExitStack, asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from openai import APIError
from sqlalchemy import text, update
from uvicorn.logging import AccessFormatter, DefaultFormatter
from . import agents, approval
from .api import auth, routes
from .core import POSTGRES, STORAGE, URL, Session, engine, uid
from .graph import build
from .models import Base, Task
from .services import detections

def configure_log_timestamps():
    prefix = "%(asctime)s.%(msecs)03d %(levelprefix)s "
    for name, cls, message in (
        ("uvicorn", DefaultFormatter, "%(message)s"),
        ("uvicorn.access", AccessFormatter, '%(client_addr)s - "%(request_line)s" %(status_code)s'),
    ):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler.formatter, cls):
                handler.setFormatter(cls(prefix + message, datefmt="%Y-%m-%d %H:%M:%S",
                                         use_colors=handler.formatter.use_colors))

# Uvicorn imports the app before emitting its server startup messages.
configure_log_timestamps()

@asynccontextmanager
async def lifespan(app):
    auth.secret()
    STORAGE.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        with engine.begin() as conn:
            if POSTGRES:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(conn)
            conn.execute(update(Task).where(Task.status.in_(["running", "pending"])).values(status="interrupted"))
        if POSTGRES:
            from langgraph.checkpoint.postgres import PostgresSaver
            from langgraph.store.postgres import PostgresStore
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row
            pool = stack.enter_context(ConnectionPool(URL.replace("postgresql+psycopg://", "postgresql://"),
                                       kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}))
            saver, store = PostgresSaver(pool), PostgresStore(pool)
            saver.setup()
            store.setup()
        else:
            saver, store = InMemorySaver(), InMemoryStore()
        detections.detector.load()
        app.state.graph = build(agents.client(), saver, store)
        app.state.review = approval.build_review(saver)
        yield
        detections.executor.shutdown(wait=True)
        engine.dispose()

app = FastAPI(title="无人机红外检测助手", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(routes.router)

@app.middleware("http")
async def correlation(request, call_next):
    request.state.request_id = uid()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

def error(request, code, message):
    return JSONResponse(status_code=code, content={"code": code, "message": message,
                        "request_id": getattr(request.state, "request_id", None)})

@app.exception_handler(HTTPException)
async def api_error(request, exc):
    return error(request, exc.status_code, exc.detail)

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return error(request, 422, "请求字段或类型有误：" + ", ".join(str(e["loc"][-1]) for e in exc.errors()))

@app.exception_handler(APIError)
async def provider_error(request, exc):
    return error(request, 502, "模型 API 调用失败，请检查服务端密钥、模型名称、余额和网络")

@app.exception_handler(Exception)
async def unexpected(request, exc):
    logging.exception("请求失败 %s", request.state.request_id)
    return error(request, 500, "服务端处理失败，请按 request_id 检查后端日志")

@app.get("/api/health/ready")
def ready():
    with Session() as db:
        db.execute(text("SELECT 1"))
    return {"ready": True, "model_hash": detections.detector.digest}
