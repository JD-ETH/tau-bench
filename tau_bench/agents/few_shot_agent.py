# Copyright Sierra

import random
from typing import Any, Dict, List, Optional

from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.chat import ChatFn


class FewShotToolCallingAgent(ToolCallingAgent):
    """The tool-calling agent with sampled example conversations appended to its system prompt."""

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        few_shot_displays: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_few_shots: int = 5,
        chat: Optional[ChatFn] = None,
    ):
        super().__init__(tools_info=tools_info, wiki=wiki, model=model, provider=provider, temperature=temperature, chat=chat)
        few_shot_displays = few_shot_displays or []
        if len(few_shot_displays) == 0:
            raise ValueError("Few shot displays are empty")
        elif len(few_shot_displays) < num_few_shots:
            raise ValueError(
                f"Few shot displays are less than num_few_shots requested: {len(few_shot_displays)} < {num_few_shots}"
            )
        self.few_shot_displays = few_shot_displays
        self.num_few_shots = num_few_shots

    def system_prompt(self) -> str:
        sampled = random.sample(self.few_shot_displays, self.num_few_shots)
        few_shots = "\n\n".join(f"Example {i + 1}:\n{display}" for i, display in enumerate(sampled))
        return f"{self.wiki}\n\n{few_shots}"
