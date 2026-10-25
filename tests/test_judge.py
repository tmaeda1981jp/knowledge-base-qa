"""判定レイヤーのテスト（Jev / Claude の API は呼ばない）。"""

from types import SimpleNamespace

import pytest

from app.judge import FallbackJudge, JevJudge, _clamp


class FakeTypeSafeClient:
    """typesafe_sdk.TypeSafeClient の system_one だけを模倣する。"""

    def __init__(self, nouls: dict[str, float]):
        self.nouls = nouls
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {k: SimpleNamespace(type="noul", noul=self.nouls[k]) for k in questions}
        return SimpleNamespace(model="jev-test", answers=answers, usage=None)


def make_jev(nouls):
    judge = JevJudge.__new__(JevJudge)  # __init__ は本物のクライアントを作るので回避
    judge._client = FakeTypeSafeClient(nouls)
    return judge


def test_jev_grade_asks_one_noul_per_chunk_in_one_request():
    judge = make_jev({"chunk_0": 0.91, "chunk_1": 0.12})

    scores = judge.grade("質問", ["チャンクA", "チャンクB"])

    assert scores == [0.91, 0.12]
    assert len(judge._client.calls) == 1  # k 問を1リクエストに同梱
    state, questions = judge._client.calls[0]
    assert state["question"] == "質問"
    assert [c["text"] for c in state["chunks"]] == ["チャンクA", "チャンクB"]
    assert set(questions) == {"chunk_0", "chunk_1"}
    assert "0" in str(questions["chunk_0"].instructions)


def test_jev_verify_returns_groundedness():
    judge = make_jev({"grounded": 0.83})

    g = judge.verify("質問", "回答", ["根拠1", "根拠2"])

    assert g == 0.83
    state, _ = judge._client.calls[0]
    assert state == {"question": "質問", "answer": "回答", "evidence": ["根拠1", "根拠2"]}


def test_jev_grade_with_no_chunks_does_not_call_api():
    judge = make_jev({})
    assert judge.grade("質問", []) == []
    assert judge._client.calls == []


class BrokenJudge:
    name = "broken"

    def grade(self, q, chunks):
        raise ConnectionError("down")

    def verify(self, q, a, chunks):
        raise ConnectionError("down")


class ConstJudge:
    name = "const"

    def grade(self, q, chunks):
        return [0.5] * len(chunks)

    def verify(self, q, a, chunks):
        return 0.5


def test_fallback_switches_to_secondary_and_records_name():
    judge = FallbackJudge(BrokenJudge(), ConstJudge())
    assert judge.name == "broken"

    assert judge.grade("q", ["a", "b"]) == [0.5, 0.5]
    assert judge.name == "const (fallback)"
    assert judge.verify("q", "a", ["x"]) == 0.5


def test_fallback_uses_primary_when_it_works():
    judge = FallbackJudge(ConstJudge(), BrokenJudge())
    assert judge.grade("q", ["a"]) == [0.5]
    assert judge.name == "const"


@pytest.mark.parametrize("value,expected", [(-0.2, 0.0), (0.4, 0.4), (1.7, 1.0)])
def test_clamp(value, expected):
    assert _clamp(value) == expected
