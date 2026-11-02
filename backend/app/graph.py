from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from . import agents
from .core import Session, fail, owned
from .models import File, Turn
from .services.detections import detect

class State(TypedDict, total=False):
    owner: str
    thread: str
    client_id: str
    question: str
    files: list[str]
    linked_task: str | None
    last_task: str | None
    intent: str
    reply: dict

def build(model, saver, store):
    def route(s):
        with Session() as db:
            kinds = [owned(db, File, i, s["owner"]).purpose for i in s["files"]]
        if s["question"].startswith("记住偏好："):
            return {"intent": "preference"}
        choice = agents.ask_json(model, "分类当前请求，返回 intent：detect 为运行新的图片检测；"
                                 "manage 为新增或编辑公共知识；qa 为领域问题或追问已有结果。", {
                                     "请求": s["question"], "附件类型": kinds, "上次任务": s.get("last_task")})
        return {"intent": choice.get("intent") if choice.get("intent") in ("detect", "manage") else "qa"}

    def detection(s):
        with Session() as db:
            files = [owned(db, File, i, s["owner"]) for i in s["files"]]
            image = next((f for f in files if f.purpose == "image"), None)
            if not image:
                return {"reply": {"answer": "请附上一张 JPEG 或 PNG 图片后请求检测。"}}
            task = detect(db, s["owner"], image, 0.25, s["thread"] + ":" + s["client_id"])
            return {"last_task": task["id"], "reply": {"answer": "检测状态：" + task["status"],
                                                       "task_id": task["id"]}}

    def query(s):
        with Session() as db:
            recent = db.scalars(select(Turn).where(Turn.thread_id == s["thread"])
                                .order_by(Turn.created_at.desc()).limit(6)).all()
            history = [(t.question, t.reply["answer"]) for t in reversed(recent)]
            pref = store.get(("preferences", s["owner"]), "style")
            return {"reply": agents.answer(model, db, s["owner"], s["question"], history,
                                           s.get("last_task"), pref.value if pref else None)}

    def maintenance(s):
        from .models import Operation
        with Session.begin() as db:
            # Retrying a turn after interruption reuses its proposal.
            op = db.scalar(select(Operation).where(Operation.thread_id == s["thread"],
                                                    Operation.request_id == s["client_id"]))
            if op:
                return {"reply": {"answer": "变更提议已保存。", "operation_id": op.id}}
            files = [owned(db, File, i, s["owner"]) for i in s["files"]]
            file = next((f for f in files if f.purpose == "knowledge"), None)
            reply = agents.manage(model, db, s["owner"], s["thread"], s["question"], file)
            if "operation_id" in reply:
                db.get(Operation, reply["operation_id"]).request_id = s["client_id"]
            return {"reply": reply}

    def preference(s):
        value = s["question"].partition("：")[2].strip()
        if not value:
            fail(422, "偏好不能为空")
        store.put(("preferences", s["owner"]), "style", {"style": value[:500]})
        return {"reply": {"answer": "已保存回答偏好。"}}

    graph = StateGraph(State)
    nodes = {"detect": detection, "qa": query, "manage": maintenance, "preference": preference}
    graph.add_node("route", route)
    graph.add_edge(START, "route")
    for name, fn in nodes.items():
        graph.add_node(name, fn)
        graph.add_edge(name, END)
    graph.add_conditional_edges("route", lambda s: s["intent"], {k: k for k in nodes})
    return graph.compile(checkpointer=saver, store=store)
