import difflib
import hashlib
import os
import re
from threading import Lock
import jieba
import numpy as np
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from ..core import POSTGRES, fail, owned
from ..models import Chunk, DIM, Document, Operation, Version

publish_lock = Lock()

def embedding_model():
    return os.getenv("EMBEDDING_MODEL", "").strip()

def embeddings():
    return OpenAIEmbeddings(model=embedding_model(), base_url=os.environ["EMBEDDING_BASE_URL"],
                            api_key=os.environ["EMBEDDING_API_KEY"], request_timeout=40,
                            max_retries=1, check_embedding_ctx_length=False)

def split(body):
    size = int(os.getenv("CHUNK_SIZE", "800"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    if not 0 <= overlap < size:
        raise ValueError("CHUNK_OVERLAP 须小于 CHUNK_SIZE")
    lines = body.splitlines(keepends=True)
    starts = sorted({0, *(i for i, line in enumerate(lines) if re.match(r"^#{1,6}\s", line))})
    parts = []
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        section = "".join(lines[start:end])
        heading = lines[start].lstrip("# ").strip() if lines[start].startswith("#") else "正文"
        for offset in range(0, len(section), size - overlap):
            excerpt = section[offset:offset + size]
            first = start + section.count("\n", 0, offset) + 1
            last = first + excerpt.rstrip("\n").count("\n")
            if excerpt.strip():
                parts.append((f"{heading} · L{first}–{last}", excerpt.strip()))
            if offset + size >= len(section):
                break
    return parts

def terms(s):
    return [t for t in jieba.lcut(s.lower()) if re.search(r"\w", t)]

def search(db, query, limit=5):
    active = select(Chunk).join(Version, Chunk.version_id == Version.id).join(
        Document, Document.id == Version.doc_id).where(Version.number == Document.active)
    chunks = db.scalars(active).all()
    if not chunks:
        return []
    tokens = [terms(c.body) for c in chunks]
    query_tokens = terms(query)
    scores = BM25Okapi([row or [""] for row in tokens]).get_scores(query_tokens)
    matched = [i for i, row in enumerate(tokens) if set(query_tokens).intersection(row)]
    keyword = sorted(matched, key=lambda i: scores[i], reverse=True)
    rankings = [[chunks[i].id for i in keyword]]
    mode = os.getenv("RETRIEVAL_MODE", "hybrid" if embedding_model() else "keyword")
    if mode not in ("keyword", "vector", "hybrid"):
        fail(503, "RETRIEVAL_MODE 应为 keyword、vector 或 hybrid")
    if mode != "keyword":
        if not embedding_model():
            fail(503, "向量检索需要先配置 Embedding")
        if mode == "vector":
            rankings.clear()
        v = embeddings().embed_query(query)
        if len(v) != DIM:
            fail(503, "Embedding 维度与配置不符")
        stmt = active.where(Version.embedding_model == embedding_model(), Chunk.vector.is_not(None))
        if POSTGRES:
            from pgvector.sqlalchemy import Vector
            from sqlalchemy import cast
            nearest = db.scalars(stmt.order_by(cast(Chunk.vector, Vector(DIM)).cosine_distance(v)).limit(20)).all()
        else:
            def cosine(c):
                return np.dot(v, c.vector) / max(1e-12, np.linalg.norm(v) * np.linalg.norm(c.vector))
            nearest = sorted(db.scalars(stmt).all(), key=cosine, reverse=True)[:20]
        rankings.append([c.id for c in nearest])
    fused = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking[:20], 1):
            fused[key] = fused.get(key, 0) + 1 / (60 + rank)
    by_id = {c.id: c for c in chunks}
    hits = []
    for key in sorted(fused, key=fused.get, reverse=True)[:limit]:
        c = by_id[key]
        version = db.get(Version, c.version_id)
        hits.append({"chunk_id": key, "title": db.get(Document, version.doc_id).title,
                     "location": c.location, "text": c.body, "version": version.number,
                     "url": f"/api/documents/{version.doc_id}/versions/{version.id}"})
    return hits

def current(db, doc):
    return db.scalar(select(Version).where(Version.doc_id == doc.id, Version.number == doc.active))

def digest(body):
    return hashlib.sha256(body.encode()).hexdigest()

def propose(db, owner, thread, title, body, expected):
    body = body.replace("\r\n", "\n").strip()
    if not title.strip() or not body or len(body) > 250000:
        fail(422, "知识内容须为 1 至 250000 字符且标题不能为空")
    existing = db.scalar(select(Version).join(Document, Version.doc_id == Document.id)
                         .where(Version.number == Document.active, Version.digest == digest(body)))
    if existing:
        return {"answer": "公共知识库已有相同内容。", "version_id": existing.id}
    doc = db.scalar(select(Document).where(Document.title == title))
    if (doc.active if doc else 0) != expected:
        fail(409, "编辑期间文档版本已变化，请重试")
    op = Operation(owner_id=owner, thread_id=thread, title=title, candidate=body,
                   doc_id=doc.id if doc else None, expected=doc.active if doc else 0,
                   status="pending")
    db.add(op)
    db.flush()
    return {"answer": "请核对候选内容及差异，然后确认发布。", "operation_id": op.id}

def preview(db, op):
    old = db.scalar(select(Version).where(Version.doc_id == op.doc_id, Version.number == op.expected))
    diff = difflib.unified_diff((old.body if old else "").splitlines(), op.candidate.splitlines(),
                                fromfile=f"v{op.expected}", tofile="候选版本", lineterm="")
    return {"id": op.id, "title": op.title, "expected_version": op.expected, "status": op.status,
            "version_id": op.version_id, "diff": "\n".join(diff)}

def resolve(db, owner, op_id, confirm):
    with publish_lock:
        op = owned(db, Operation, op_id, owner)
        if op.status == ("published" if confirm else "cancelled"):
            return preview(db, op)
        if op.status != "pending":
            fail(409, "该操作已结束")
        if confirm:
            doc = db.scalar(select(Document).where(Document.title == op.title))
            if (doc.active if doc else 0) != op.expected:
                fail(409, "版本已变化，请重新生成提议")
            parts = split(op.candidate)
            # Close the read transaction before the external embedding call.
            db.commit()
            vectors = embeddings().embed_documents([s for _, s in parts]) if embedding_model() else [None] * len(parts)
            if len(vectors) != len(parts) or any(v is not None and len(v) != DIM for v in vectors):
                fail(503, "Embedding 数量或维度与配置不符")
            if not doc:
                doc = Document(title=op.title, active=0, created_by=owner)
                db.add(doc)
                db.flush()
            doc.active += 1
            version = Version(doc_id=doc.id, number=doc.active, body=op.candidate, author=owner,
                              digest=digest(op.candidate), embedding_model=embedding_model())
            db.add(version)
            db.flush()
            for (loc, body), vec in zip(parts, vectors):
                db.add(Chunk(version_id=version.id, location=loc, body=body, vector=vec))
            op.doc_id, op.version_id, op.status = doc.id, version.id, "published"
        else:
            op.status = "cancelled"
        db.commit()
        return preview(db, op)
