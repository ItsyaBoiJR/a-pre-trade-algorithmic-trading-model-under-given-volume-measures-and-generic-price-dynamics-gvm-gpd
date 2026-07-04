import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class PreTradeExecutionModel:
    def __init__(self, volume_profile, price_covariance, participation_cap, risk_aversion):
        """
        Initialize the pre-trade execution model.

        Args:
            volume_profile (torch.Tensor): The expected volume profile (discretized over time).
            price_covariance (torch.Tensor): The covariance matrix of price dynamics.
            participation_cap (float): The maximum allowable participation rate.
            risk_aversion (float): The risk aversion parameter.
        """
        self.volume_profile = volume_profile
        self.price_covariance = price_covariance
        self.participation_cap = participation_cap
        self.risk_aversion = risk_aversion
        self.num_intervals = len(volume_profile)

    def compute_impact_cost(self, pov):
        """
        Compute the impact cost based on the Participation of Volume (PoV) function.

        Args:
            pov (torch.Tensor): The PoV function (participation rate over time).

        Returns:
            torch.Tensor: The impact cost.
        """
        return torch.sum(pov ** 2)

    def compute_opportunistic_cost(self, pov):
        """
        Compute the opportunistic cost based on price covariance and PoV.

        Args:
            pov (torch.Tensor): The PoV function (participation rate over time).

        Returns:
            torch.Tensor: The opportunistic cost.
        """
        return torch.matmul(pov, torch.matmul(self.price_covariance, pov))

    def compute_total_cost(self, pov):
        """
        Compute the total cost as a combination of impact and opportunistic costs.

        Args:
            pov (torch.Tensor): The PoV function (participation rate over time).

        Returns:
            torch.Tensor: The total cost.
        """
        impact_cost = self.compute_impact_cost(pov)
        opportunistic_cost = self.compute_opportunistic_cost(pov)
        return impact_cost + self.risk_aversion * opportunistic_cost

    def optimize_execution(self):
        """
        Optimize the PoV function to minimize the total cost under constraints.

        Returns:
            torch.Tensor: The optimized PoV function.
        """
        # Initialize PoV with uniform participation
        pov = torch.full((self.num_intervals,), 1.0 / self.num_intervals, requires_grad=True)

        # Define optimizer
        optimizer = Adam([pov], lr=0.01)

        # Optimization loop
        for _ in range(1000):
            optimizer.zero_grad()

            # Compute total cost
            total_cost = self.compute_total_cost(pov)

            # Add participation cap constraint as a penalty
            penalty = torch.sum(torch.clamp(pov - self.participation_cap, min=0) ** 2)
            total_cost += penalty

            # Backpropagation
            total_cost.backward()
            optimizer.step()

            # Ensure PoV is non-negative and sums to 1 (normalization)
            with torch.no_grad():
                pov.clamp_(min=0)
                pov /= torch.sum(pov)

        return pov.detach()

if __name__ == '__main__':
    # Dummy data for testing
    num_intervals = 10
    volume_profile = torch.tensor([0.1] * num_intervals)  # Uniform volume profile
    price_covariance = torch.eye(num_intervals) * 0.01  # Simplified covariance matrix
    participation_cap = 0.2  # Max 20% participation
    risk_aversion = 0.5  # Risk aversion parameter

    # Initialize the model
    model = PreTradeExecutionModel(volume_profile, price_covariance, participation_cap, risk_aversion)

    # Optimize execution
    optimized_pov = model.optimize_execution()

    # Print results
    print("Optimized Participation of Volume (PoV):", optimized_pov.numpy())
    print("Sum of PoV:", optimized_pov.sum().item())