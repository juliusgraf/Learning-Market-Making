from dataclasses import dataclass

@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    length: int
    terminal_inventory: float
    pnl: float
