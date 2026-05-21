"""Token reward calculation for node operators."""
from dataclasses import dataclass


@dataclass
class RewardCalculator:
    """Calculate token rewards based on node performance."""
    base_reward_per_request: float = 0.001  # $GBLOCK per request
    reputation_multiplier: bool = True
    min_reputation: float = 30.0
    bonus_uptime_hours: float = 100.0  # Bonus multiplier threshold

    def calculate(
        self,
        requests_served: int,
        reputation_score: float,
        uptime_hours: float,
    ) -> float:
        """Calculate total reward for a period.

        Args:
            requests_served: Number of requests completed
            reputation_score: Current reputation score (0-100)
            uptime_hours: Total uptime in hours

        Returns:
            Token reward amount in $GBLOCK
        """
        if reputation_score < self.min_reputation:
            return 0.0

        # Base reward
        base = requests_served * self.base_reward_per_request

        # Reputation multiplier: 0.5x at score 30, 1.0x at 50, 2.0x at 100
        if self.reputation_multiplier:
            rep_mult = 0.5 + (reputation_score - 30) / 70 * 1.5
            rep_mult = max(0.5, min(2.0, rep_mult))
        else:
            rep_mult = 1.0

        # Uptime bonus: 10% extra for every 100 hours
        uptime_bonus = 1.0 + (uptime_hours / self.bonus_uptime_hours) * 0.1

        return base * rep_mult * uptime_bonus

    def estimate_daily(self, requests_per_hour: float, reputation: float) -> float:
        """Estimate daily rewards."""
        daily_requests = requests_per_hour * 24
        return self.calculate(daily_requests, reputation, 24.0)
