# Copyright Sierra

import json
from typing import Any, Dict, List, Optional

from tau_bench.agents.base import Agent
from tau_bench.chat import ChatFn, litellm_chat
from tau_bench.envs.base import Env
from tau_bench.types import RESPOND_ACTION_FIELD_NAME, RESPOND_ACTION_NAME, Action, SolveResult


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        chat: Optional[ChatFn] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.chat = chat if chat is not None else litellm_chat(model, provider, temperature=temperature)

    def system_prompt(self) -> str:
        return self.wiki

    def solve(self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": env_reset_res.observation},
        ]
        for _ in range(max_num_steps):
            result = self.chat(messages, self.tools_info)
            total_cost += result.cost
            # The turn is kept as the model produced it, so that replaying it extends the same tokens.
            messages.append(result.message)
            if result.finish_reason == "length":
                # Cut off before the model finished deciding: nothing to act on, and the turn cannot be resumed.
                break
            done = False
            tool_calls = result.message.get("tool_calls") or []
            if not tool_calls:
                env_response = env.step(
                    Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: result.message.get("content") or ""})
                )
                reward, info, done = env_response.reward, {**info, **env_response.info.model_dump()}, env_response.done
                messages.append({"role": "user", "content": env_response.observation})
            # Every call of the turn runs, in order, each answered by its own tool message.
            for tool_call in tool_calls:
                if done:
                    observation = "Error: the conversation has ended"
                else:
                    try:
                        action = tool_call_to_action(tool_call)
                    except ValueError as e:
                        observation = f"Error: {e}"
                    else:
                        env_response = env.step(action)
                        reward, info, done = env_response.reward, {**info, **env_response.info.model_dump()}, env_response.done
                        observation = env_response.observation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": observation,
                    }
                )
            if done:
                break
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)


def tool_call_to_action(tool_call: Dict[str, Any]) -> Action:
    """The environment action of one OpenAI tool call; ``ValueError`` when its arguments are not valid JSON."""
    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"tool call arguments are not valid JSON ({e}): {arguments}") from e
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call arguments must be a JSON object, got {type(arguments).__name__}")
    return Action(name=tool_call["function"]["name"], kwargs=arguments)
