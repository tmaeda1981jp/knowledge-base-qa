"""LangGraph の分岐テスト（LLM・判定器はモック）。

1. 関係あり → generate → verify 合格
2. 関係なし → rewrite → retrieve → 関係あり → 回答
3. 関係なし → rewrite → 再び関係なし → no_answer（再試行は1回まで）
4. 関係あり → generate → verify 不合格 → no_answer（根拠候補は残す）
5. verify 無効なら素通り
"""

from app import prompts
from tests.conftest import FakeJudge


def nodes(result):
    return [s.node for s in result["steps"]]


def test_relevant_then_grounded_answer(make_graph, run, retriever):
    judge = FakeJudge(grade_results=[[0.9, 0.2, 0.7]], verify_results=[0.95])
    graph, _ = make_graph(judge, responses=["見学は約60分です[1]。"])

    result = run(graph)

    assert nodes(result) == ["retrieve", "grade", "generate", "verify"]
    assert result["found"] is True
    assert result["answer"] == "見学は約60分です[1]。"
    assert result["rewritten_question"] is None
    assert result["groundedness"] == 0.95
    # 閾値 0.6 以上の 1件目・3件目だけが根拠になる
    assert [s.file for s in result["sources"]] == ["distillery.md", "terms.txt"]
    assert [s.relevance for s in result["sources"]] == [0.9, 0.7]
    assert result["sources"][0].page is None
    assert retriever.queries == ["黒滝蒸留所の見学は何分？"]
    # generate には関係ありのチャンクだけが渡る
    assert judge.verify_calls[0][2] == [
        "黒滝蒸留所の見学は事前予約制で、所要時間は約60分です。",
        "フィニッシュとは飲み込んだ後に残る余韻のこと。",
    ]


def test_irrelevant_then_rewrite_then_answer(make_graph, run, retriever):
    judge = FakeJudge(grade_results=[[0.1, 0.2, 0.3], [0.8, 0.1, 0.1]], verify_results=[0.9])
    graph, _ = make_graph(judge, responses=["黒滝蒸留所 見学 所要時間", "約60分です[1]。"])

    result = run(graph)

    assert nodes(result) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "verify",
    ]
    assert result["found"] is True
    assert result["rewritten_question"] == "黒滝蒸留所 見学 所要時間"
    assert result["retry_count"] == 1
    # 2回目の検索は言い換え後の質問で行う
    assert retriever.queries == ["黒滝蒸留所の見学は何分？", "黒滝蒸留所 見学 所要時間"]
    # 回答生成は元の質問に対して行う
    assert judge.verify_calls[0][0] == "黒滝蒸留所の見学は何分？"


def test_irrelevant_twice_gives_no_answer(make_graph, run):
    judge = FakeJudge(grade_results=[[0.1, 0.1, 0.1], [0.2, 0.0, 0.1]])
    graph, llm = make_graph(judge, responses=["言い換えた質問"])

    result = run(graph, "東京の明日の天気は？")

    assert nodes(result) == ["retrieve", "grade", "rewrite", "retrieve", "grade", "no_answer"]
    assert result["found"] is False
    assert result["answer"] == prompts.NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["retry_count"] == 1  # 再試行は1回で打ち切り
    assert judge.verify_calls == []  # 回答を生成していないので verify も呼ばれない
    assert len(judge.grade_calls) == 2


def test_not_grounded_answer_is_discarded(make_graph, run):
    judge = FakeJudge(grade_results=[[0.9, 0.1, 0.1]], verify_results=[0.2])
    graph, _ = make_graph(judge, responses=["見学は3時間で、昼食つきです。"])

    result = run(graph)

    assert nodes(result) == ["retrieve", "grade", "generate", "verify", "no_answer"]
    assert result["found"] is False
    assert result["answer"] == prompts.NOT_GROUNDED_MESSAGE
    assert result["groundedness"] == 0.2
    # 根拠候補は利用者が自分で確認できるように残す
    assert [s.file for s in result["sources"]] == ["distillery.md"]


def test_verify_can_be_disabled(make_graph, run):
    judge = FakeJudge(grade_results=[[0.9, 0.1, 0.1]])  # verify_results なし
    graph, _ = make_graph(judge, responses=["約60分です。"], verify_enabled=False)

    result = run(graph)

    assert nodes(result) == ["retrieve", "grade", "generate", "verify"]
    assert result["found"] is True
    assert result["groundedness"] is None
    assert judge.verify_calls == []
