from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

@dataclass
class State:
    """Structured state before encoding (maps to Xt in the paper)."""
    t: int
    data: Dict[str, Any]  # order book levels, inventories, processes, etc.

@dataclass
class Action:
    """Structured action (maps to At in the paper)."""
    t: int
    data: Dict[str, Any]  # different fields for CLOB vs Auction

@dataclass
class StepResult:
    next_state: State
    reward: float
    done: bool
    info: Dict[str, Any]

Obs = Any           # encoded tensor/array
ActionIndex = int   # discrete action id for DQN
