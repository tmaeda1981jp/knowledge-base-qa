"""FastAPI エンドポイントのテスト（ベクトルDB・LLM は差し替える）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main
from app.ingest import IngestResult
from app.schemas import Source, Step


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main.vectorstore, "count_chunks", lambda: 6)
    monkeypatch.setattr(main.vectorstore, "list_documents", lambda: [{"file": "a.md", "chunks": 2}])
    monkeypatch.setattr(main.settings, "docs_dir", tmp_path / "docs")
    return TestClient(main.app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ナレッジベース" in r.text


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 6
    assert body["judge"] in {"jev", "claude"}


def test_documents(client):
    assert client.get("/api/documents").json() == {"documents": [{"file": "a.md", "chunks": 2}]}


def test_chat_returns_answer_sources_and_steps(client, monkeypatch):
    def fake_ask(question):
        return {
            "answer": f"回答: {question} [1]",
            "found": True,
            "sources": [Source(file="a.md", page=None, excerpt="抜粋", relevance=0.9)],
            "rewritten_question": None,
            "groundedness": 0.93,
            "steps": [Step(node="retrieve", detail="3件"), Step(node="grade", detail="1件採用")],
            "judge": "jev",
        }

    monkeypatch.setattr(main.graph_module, "ask", fake_ask)
    r = client.post("/api/chat", json={"question": "見学は何分？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "回答: 見学は何分？ [1]"
    assert body["found"] is True
    assert body["sources"][0]["file"] == "a.md"
    assert [s["node"] for s in body["steps"]] == ["retrieve", "grade"]
    assert body["judge"] == "jev"


def test_chat_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_chat_without_documents_is_409(client, monkeypatch):
    monkeypatch.setattr(main.vectorstore, "count_chunks", lambda: 0)
    r = client.post("/api/chat", json={"question": "x"})
    assert r.status_code == 409


def test_chat_hides_internal_errors(client, monkeypatch):
    def boom(question):
        raise RuntimeError("secret sk-ant-xxx leaked?")

    monkeypatch.setattr(main.graph_module, "ask", boom)
    r = client.post("/api/chat", json={"question": "x"})
    assert r.status_code == 502
    assert "sk-ant" not in r.text


def test_ingest(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.ingest_module, "resolve_source_dir", lambda: tmp_path)
    monkeypatch.setattr(
        main.ingest_module,
        "ingest_directory",
        lambda d: IngestResult(
            source_dir=str(d),
            files=[{"file": "a.md", "chunks": 2}],
            warnings=["b.pdf: 文字を取り出せませんでした"],
        ),
    )
    body = client.post("/api/ingest").json()
    assert body["files"] == [{"file": "a.md", "chunks": 2}]
    assert body["chunks"] == 2
    assert body["warnings"] == ["b.pdf: 文字を取り出せませんでした"]


def test_upload_saves_to_docs_and_ingests(client, monkeypatch, tmp_path):
    seen = {}

    def fake_ingest_file(path):
        seen["path"] = path
        return 3

    monkeypatch.setattr(main.ingest_module, "ingest_file", fake_ingest_file)
    r = client.post(
        "/api/upload", files={"file": ("memo.md", "# メモ\n本文".encode(), "text/markdown")}
    )
    assert r.status_code == 200
    assert r.json() == {"file": "memo.md", "chunks": 3}
    assert seen["path"] == tmp_path / "docs" / "memo.md"
    assert (tmp_path / "docs" / "memo.md").read_text() == "# メモ\n本文"


def test_upload_rejects_unsupported_type(client):
    r = client.post("/api/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
