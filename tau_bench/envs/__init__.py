# Copyright Sierra

from typing import Optional, Union
from tau_bench.envs.base import Env
from tau_bench.envs.user import BaseUserSimulationEnv, UserStrategy


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
    user: Optional[BaseUserSimulationEnv] = None,
) -> Env:
    """The environment of one domain; ``user`` replaces the simulator ``user_strategy`` would build."""
    if env_name == "retail":
        from tau_bench.envs.retail import MockRetailDomainEnv

        env_cls = MockRetailDomainEnv
    elif env_name == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv

        env_cls = MockAirlineDomainEnv
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env_cls(
        user_strategy=user_strategy,
        user_model=user_model,
        task_split=task_split,
        user_provider=user_provider,
        task_index=task_index,
        user=user,
    )
