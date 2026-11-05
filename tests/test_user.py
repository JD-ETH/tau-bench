"""The user simulator over an injected chat function, inside a real retail environment: no network."""
from typing import Any, Dict, List, Optional

import litellm
import pytest

from tau_bench import chat as chat_module
from tau_bench.chat import ChatResult, litellm_chat
from tau_bench.envs import get_env
from tau_bench.envs.user import LLMUserSimulationEnv, ReactUserSimulationEnv, load_user
from tau_bench.types import RESPOND_ACTION_NAME, Action


class ScriptedChat:
    def __init__(self, replies: List[str]):
        self.replies = replies
        self.requests: List[List[Dict[str, Any]]] = []

    def __call__(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> ChatResult:
        self.requests.append(list(messages))
        return ChatResult(message={"role": "assistant", "content": self.replies[len(self.requests) - 1]}, cost=0.5)


def test_an_injected_simulator_drives_a_real_environment_without_a_request_at_construction():
    chat = ScriptedChat(["Hi, I want to cancel my order.", "###STOP###"])
    env = get_env("retail", user_strategy="llm", user_model=None, task_split="test", task_index=0, user=LLMUserSimulationEnv(chat=chat))
    assert chat.requests == []

    reset = env.reset(task_index=0)
    assert reset.observation == "Hi, I want to cancel my order."
    assert env.task.instruction in chat.requests[0][0]["content"]

    response = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Sure, which order?"}))
    assert response.done and response.observation == "###STOP###"
    assert chat.requests[1][-1] == {"role": "user", "content": "Sure, which order?"}
    assert env.user.get_total_cost() == 1.0


def test_load_user_builds_llm_strategies_from_a_chat_function_alone():
    chat = ScriptedChat(["User Response:\nHello there"])
    user = load_user("react", chat=chat)
    assert isinstance(user, ReactUserSimulationEnv)
    assert user.reset(instruction="say hello") == "Hello there"
    with pytest.raises(ValueError):
        load_user("llm")


def test_litellm_chat_retries_transient_provider_failures(monkeypatch: pytest.MonkeyPatch):
    attempts: List[int] = []

    class Message:
        def model_dump(self, exclude_none: bool = False):
            return {"role": "assistant", "content": "ok"}

    class Response:
        choices = [type("Choice", (), {"message": Message(), "finish_reason": "stop"})()]
        _hidden_params = {"response_cost": None}

    def flaky_completion(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise litellm.ServiceUnavailableError(message="down", llm_provider="p", model="m")
        assert kwargs["tools"] == [{"type": "function"}] and kwargs["temperature"] == 0.7
        return Response()

    monkeypatch.setattr(chat_module, "completion", flaky_completion)
    monkeypatch.setattr(chat_module.time, "sleep", lambda seconds: None)
    result = litellm_chat("m", "p", temperature=0.7)([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert len(attempts) == 2
    assert result == ChatResult(message={"role": "assistant", "content": "ok"}, finish_reason="stop", cost=0.0)
