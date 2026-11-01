"""Run from the repository root: python -m backend.tests.smoke"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

class ExampleModel:
    """Scripted DeepSeek replies. Database, tools, graph and detector run normally."""
    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        import json
        prompt = messages[0].content
        payload = messages[-1].content
        if "分类当前请求" in prompt:
            q = json.loads(payload)["请求"]
            intent = "manage" if "导入" in q else "detect" if q == "请检测" else "qa"
            return AIMessage(content=json.dumps({"intent": intent}))
        if "选择要维护" in prompt:
            return AIMessage(content='{"title":"试验文档","mode":"import"}')
        if isinstance(messages[-1], ToolMessage):
            if payload == "没有可读取的检测任务" or '"original_name"' in payload:
                return AIMessage(content=payload)
            assert "回归测试资料" in payload
            return AIMessage(content="试验文档 L1 包含回归测试资料。")
        if "刚才的图" in json.loads(payload)["当前问题"]:
            return AIMessage(content="", tool_calls=[{"id": "d1", "name": "get_detection_result",
                             "args": {}, "type": "tool_call"}])
        return AIMessage(content="", tool_calls=[{"id": "t1", "name": "search_knowledge",
                         "args": {"query": "回归测试资料"}, "type": "tool_call"}])

def run(client):
    from uuid import uuid4
    from backend.app.core import ROOT
    def req(method, url, expected=200, **kwargs):
        r = client.request(method, "/api" + url, **kwargs)
        assert r.status_code == expected, (url, r.status_code, r.text[:500])
        return r

    def account(name):
        data = {"username": name, "password": "local-testing"}
        req("POST", "/auth/register", 201, json=data)
        token = req("POST", "/auth/login", json=data).json()["access_token"]
        return {"Authorization": "Bearer " + token}

    a, b = account("a"), account("b")
    req("GET", "/auth/me", 401)
    thread = req("POST", "/conversations", 201, headers=a).json()["thread_id"]
    other = req("POST", "/conversations", 201, headers=b).json()["thread_id"]

    def message(text, headers=a, conversation=thread, attachments=(), client_id=None, expected=200):
        return req("POST", "/chat", expected, headers=headers, json={
            "thread_id": conversation, "message": text, "client_message_id": client_id or str(uuid4()),
            "attachment_ids": attachments}).json()

    def md(content, headers=a):
        return req("POST", "/files", 201, headers=headers, data={"purpose": "knowledge"},
                   files={"file": ("test.md", content.encode(), "text/markdown")}).json()["file_id"]

    def proposal(content, headers=a, conversation=thread):
        return message("完整导入附件", headers, conversation, [md(content, headers)])["operation_id"]

    def confirm(op, headers=a, expected=200):
        return req("POST", "/knowledge/operations/" + op + "/confirm", expected, headers=headers).json()

    p = proposal("# 回归测试资料\n样图和检测输出。")
    confirm(p, b, 404)
    v = confirm(p)["version_id"]
    assert confirm(p)["version_id"] == v
    docs = req("GET", "/documents", headers=b).json()
    assert len(docs) == 1 and docs[0]["version"] == 1
    answer = message("资料里有什么？", b, other)
    assert answer["citations"] and req("GET", answer["citations"][0]["url"][4:], headers=b).json()["number"] == 1
    old = proposal("# 回归测试资料\n内容二。")
    newer = proposal("# 回归测试资料\n内容三。", b, other)
    confirm(newer, b)
    confirm(old, expected=409)
    req("POST", "/knowledge/operations/" + old + "/cancel", headers=a)
    confirm(old, expected=409)
    req("GET", "/conversations/" + thread + "/messages", 404, headers=b)
    message("请检测", b, other, [md("# 未发布")], expected=404)
    assert "附上" in message("请检测")["answer"]
    req("POST", "/files", 422, headers=a, data={"purpose": "image"}, files={"file": ("bad.jpg", b"bad")})
    req("POST", "/files", 422, headers=a, data={"purpose": "knowledge"}, files={"file": ("bad.md", b"\xff")})

    samples = ROOT / "RTDETR-main-code/test_images/hituav"
    counts = {}
    for image in sorted(samples.glob("*.jpg")):
        f = req("POST", "/files", 201, headers=a, data={"purpose": "image"},
                files={"file": (image.name, image.read_bytes())}).json()["file_id"]
        def submit(conf, expected=201):
            return req("POST", "/detections", expected, headers={**a, "Idempotency-Key": image.name},
                       data={"file_id": f, "conf": conf}).json()
        t = submit(0.25)
        assert t["status"] == "succeeded" and submit(0.25)["id"] == t["id"]
        submit(0.5, 409)
        req("GET", "/detections/" + t["id"] + "/image", 404, headers=b)
        assert req("GET", "/detections/" + t["id"] + "/image", headers=a).content.startswith(b"\xff\xd8")
        counts[image.stem] = len(t["detections"])
        if image.stem == "1_60_50_0_07426":
            baseline = ROOT / "RTDETR-main-code/runs/predict/pwl_test8/labels/1_60_50_0_07426.txt"
            rows = baseline.read_text().splitlines()
            assert len(rows) == len(t["detections"])
            w, h = t["width"], t["height"]
            for box, line in zip(t["detections"], rows):
                x1, y1, x2, y2 = box["bbox_xyxy"]
                normalized = [box["class_id"], (x1+x2)/(2*w), (y1+y2)/(2*h),
                              (x2-x1)/w, (y2-y1)/h, box["confidence"]]
                assert max(abs(x-float(y)) for x,y in zip(normalized, line.split())) < 1e-5
    print("Six-image CPU results:", counts)
    assert len(counts) == 6
    import json
    followup = message("刚才的图叫什么？")
    details = json.loads(followup["answer"])
    assert followup["task_id"] == t["id"] and details["original_name"] == image.name
    assert sum(details["class_counts"].values()) == counts[image.stem]
    assert message("刚才的图叫什么？", b, other)["answer"] == "没有可读取的检测任务"
    reply = message("请检测", attachments=[f], client_id="retry")
    assert message("请检测", attachments=[f], client_id="retry") == reply
    message("不同内容", client_id="retry", expected=409)
    print("PASS: authentication, ownership, image regression, retries, chat tools, citations, publication and version conflicts")

if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        os.environ.update(STORAGE_ROOT=directory, DATABASE_URL="sqlite:///" + str(Path(directory) / "test.sqlite3"),
                          JWT_SECRET="test-only-" * 5, EMBEDDING_MODEL="")
        from backend.app.main import app
        with patch("backend.app.agents.client", return_value=ExampleModel()), TestClient(app) as client:
            run(client)
