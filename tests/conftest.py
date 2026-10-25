"""テスト用のモック。LLM・判定器・検索を外部サービスなしで差し替える。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.graph import build_graph, initial_state

DOCS = [
    Document(
        page_content="黒滝蒸留所の見学は事前予約制で、所要時間は約60分です。",
        metadata={"source": "distillery.md", "chunk": 0},
    ),
    Document(
        page_content="バーボン樽はバニラやココナッツの香りをつけやすい。",
        metadata={"source": "cask.pdf", "page": 2, "chunk": 1},
    ),
    Document(
        page_content="フィニッシュとは飲み込んだ後に残る余韻のこと。",
        metadata={"source": "terms.txt", "chunk": 0},
    ),
]


class FakeJudge:
    """grade / verify の結果を呼び出し順に返す。"""

    name = "fake"

    def __init__(self, grade_results: list[list[float]], verify_results: list[float] | None = None):
        self._grade = list(grade_results)
        self._verify = list(verify_results or [])
        self.grade_calls: list[tuple[str, list[str]]] = []
        self.verify_calls: list[tuple[str, str, list[str]]] = []

    def grade(self, question, chunks):
        self.grade_calls.append((question, chunks))
        return self._grade.pop(0)

    def verify(self, question, answer, chunks):
        self.verify_calls.append((question, answer, chunks))
        return self._verify.pop(0)


class RecordingRetriever:
    def __init__(self, docs=DOCS):
        self.docs = docs
        self.queries: list[str] = []

    def __call__(self, query, k):
        self.queries.append(query)
        return self.docs[:k]


@pytest.fixture
def retriever():
    return RecordingRetriever()


@pytest.fixture
def make_graph(retriever):
    """build_graph の薄いラッパー。LLM の応答は responses で順に与える。"""

    def _make(judge, responses: list[str], **kwargs):
        llm = FakeListChatModel(responses=responses)
        graph = build_graph(
            judge=judge,
            llm=llm,
            retriever=retriever,
            retrieve_k=3,
            grade_threshold=0.6,
            verify_threshold=0.7,
            **kwargs,
        )
        return graph, llm

    return _make


@pytest.fixture
def run():
    def _run(graph, question="黒滝蒸留所の見学は何分？"):
        return graph.invoke(initial_state(question))

    return _run
