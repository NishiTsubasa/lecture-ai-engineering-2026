"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, confirm_deferred_tool_requests

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        pending = agent.run_sync("今日学んだ内容を記録に保存して")
        assert pending.output.approvals[0].tool_name == "save_learning_log"
        assert not list(tmp_path.iterdir())

        result = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=confirm_deferred_tool_requests(pending.output),
        )

    assert result.output
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_save_learning_log_denial_does_not_create_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        pending = agent.run_sync("今日学んだ内容を記録に保存して")
        agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=confirm_deferred_tool_requests(pending.output),
        )

    assert not list(tmp_path.iterdir())
