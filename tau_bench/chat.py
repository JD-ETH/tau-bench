# Copyright Sierra
"""The model call behind every agent and LLM user simulator.

A ``ChatFn`` takes a conversation (and the tool schemas, for agents) and returns the assistant
message exactly as the model produced it, so a caller that replays the message later sends what
was generated: content, tool calls and reasoning alike. The default talks to litellm; pass another
``ChatFn`` to reach a different endpoint, with different sampling or request options, without
touching the agent or the simulator.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import litellm
from litellm import completion

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    # An OpenAI assistant message: role, content, and tool_calls / reasoning_content when present.
    message: Dict[str, Any]
    finish_reason: Optional[str] = None
    cost: float = 0.0


ChatFn = Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], ChatResult]


def litellm_chat(
    model: str,
    provider: str,
    temperature: float = 0.0,
    max_retries: int = 10,
    retry_delay_seconds: float = 1.0,
    **completion_kwargs: Any,
) -> ChatFn:
    """A ``ChatFn`` over ``litellm.completion`` that retries the provider's transient failures."""

    def chat(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ChatResult:
        kwargs = dict(model=model, custom_llm_provider=provider, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
        kwargs.update(completion_kwargs)
        for attempt in range(1, max_retries + 1):
            try:
                res = completion(**kwargs)
                break
            except (litellm.ServiceUnavailableError, litellm.InternalServerError) as e:
                if attempt == max_retries:
                    raise RuntimeError(f"LLM service unavailable after {max_retries} attempts: {e}") from e
                logger.error(f"LLM call failed (attempt {attempt}/{max_retries}): {e.__class__.__name__}; retrying")
                time.sleep(retry_delay_seconds)
        choice = res.choices[0]
        return ChatResult(
            message=choice.message.model_dump(exclude_none=True),
            finish_reason=choice.finish_reason,
            cost=res._hidden_params.get("response_cost") or 0.0,
        )

    return chat
