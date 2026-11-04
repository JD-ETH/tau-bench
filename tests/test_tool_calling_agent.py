"""The tool-calling agent over an injected chat function, with a fake environment: no network."""
import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.tool_calling_agent import ToolCallingAgent, tool_call_to_action
from tau_bench.chat import ChatResult
from tau_bench.types import RESPOND_ACTION_NAME, Action, EnvInfo, EnvResetResponse, EnvResponse, Task

TASK = Task(user_id="u1", actions=[], instruction="cancel order 7", outputs=[])
TOOLS = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]


def tool_call(call_id: str, name: str, arguments: Any) -> Dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptedChat:
    """Returns the scripted results in order and records every request."""

    def __init__(self, results: List[ChatResult]):
        self.results = results
        self.requests: List[Dict[str, Any]] = []

    def __call__(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> ChatResult:
        self.requests.append({"messages": list(messages), "tools": tools})
        return self.results[len(self.requests) - 1]


class FakeEnv:
    """tau-bench's env contract: tools answer with their arguments, replies end the episode on 'bye'."""

    tools_info = TOOLS
    wiki = "Be helpful."

    def __init__(self):
        self.actions: List[Action] = []

    def reset(self, task_index=None):
        return EnvResetResponse(observation="Hi, cancel order 7 please.", info=EnvInfo(task=TASK))

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        if action.name == RESPOND_ACTION_NAME:
            done = "bye" in action.kwargs["content"]
            return EnvResponse(observation="###STOP###" if done else "ok", reward=float(done), done=done, info=EnvInfo(task=TASK))
        return EnvResponse(observation=json.dumps(action.kwargs), reward=0.0, done=False, info=EnvInfo(task=TASK))


def solve(results: List[ChatResult]):
    chat, env = ScriptedChat(results), FakeEnv()
    agent = ToolCallingAgent(tools_info=TOOLS, wiki="Be helpful.", chat=chat)
    return agent.solve(env, task_index=0, max_num_steps=5), chat, env


def test_every_tool_call_of_a_turn_runs_and_the_turn_is_replayed_as_returned():
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Check both orders.",
        "tool_calls": [tool_call("c1", "get_order", '{"order_id": 7}'), tool_call("c2", "get_order", '{"order_id": 8}')],
    }
    result, chat, env = solve(
        [ChatResult(message=assistant, finish_reason="tool_calls"), ChatResult(message={"role": "assistant", "content": "bye"})]
    )
    assert [(a.name, a.kwargs) for a in env.actions] == [
        ("get_order", {"order_id": 7}),
        ("get_order", {"order_id": 8}),
        (RESPOND_ACTION_NAME, {"content": "bye"}),
    ]
    replayed = chat.requests[1]["messages"]
    assert replayed[2] is assistant
    assert replayed[3] == {"role": "tool", "tool_call_id": "c1", "name": "get_order", "content": '{"order_id": 7}'}
    assert replayed[4] == {"role": "tool", "tool_call_id": "c2", "name": "get_order", "content": '{"order_id": 8}'}
    assert chat.requests[0]["tools"] is TOOLS
    assert result.reward == 1.0
    assert result.messages[-1] == {"role": "user", "content": "###STOP###"}


def test_a_turn_cut_by_the_token_limit_ends_the_episode_without_acting():
    truncated = {"role": "assistant", "content": "Let me think about"}
    result, chat, env = solve([ChatResult(message=truncated, finish_reason="length")])
    assert env.actions == []
    assert len(chat.requests) == 1
    assert result.messages[-1] == truncated
    assert result.reward == 0.0


def test_invalid_tool_arguments_become_a_tool_error_and_do_not_reach_the_env():
    assistant = {"role": "assistant", "content": None, "tool_calls": [tool_call("c1", "get_order", "{not json")]}
    result, chat, env = solve([ChatResult(message=assistant), ChatResult(message={"role": "assistant", "content": "bye"})])
    assert [a.name for a in env.actions] == [RESPOND_ACTION_NAME]
    error = chat.requests[1]["messages"][3]
    assert error["tool_call_id"] == "c1" and error["content"].startswith("Error: tool call arguments are not valid JSON")


def test_tool_call_to_action_accepts_dict_arguments_and_rejects_non_objects():
    assert tool_call_to_action(tool_call("c1", "get_order", {"order_id": 7})).kwargs == {"order_id": 7}
    try:
        tool_call_to_action(tool_call("c1", "get_order", "[1, 2]"))
    except ValueError as e:
        assert "JSON object" in str(e)
    else:
        raise AssertionError("a JSON list must be rejected")
