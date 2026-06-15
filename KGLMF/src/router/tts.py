import torch
import torch.nn.functional as F

class AdaptiveTTSController:
    def __init__(self, budget_min: int = 4, budget_max: int = 64,
                 scaling_factor: float = 2.0, temperature: float = 0.1):
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.scaling_factor = scaling_factor
        self.temperature = temperature

    def estimate_complexity(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return entropy.item()

    def compute_budget(self, complexity: float) -> int:
        raw = self.budget_min + int(self.scaling_factor * complexity)
        return min(raw, self.budget_max)

    def compute_tts_total(self, budget: int, n_agents: int,
                         graph_depth: int) -> float:
        return budget + 0.5 * n_agents + 0.3 * graph_depth
