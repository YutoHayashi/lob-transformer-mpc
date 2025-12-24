from typing import TypedDict

import pandas as pd
import torch
import pulp

from lob_transformer.module import LOBDatasetConfig, LOBDataset, LOBTransformer


class ProbabilityDistribution(TypedDict):
    down: float
    stay: float
    up: float

class Probability(TypedDict):
    horizon: int
    threshold: float
    distribution: ProbabilityDistribution


class MPC:
    """
    Model Predictive Control (MPC) for Optimal Trading
        Based on Classification Results from Multiple LOB-Transformer Models.
        Using Mixed Integer Programming for Maximizing Expected Value
    
    Objective Function:
    $$
    J = \sum_{h=0}^{H-1} ( P_h \cdot S \cdot R_h - C \cdot S \cdot I_h )
    $$
    Where:
    - $P_h$: Position at horizon $h$
    - $S$: Transaction size
    - $R_h$: Interval return at horizon $h$
    - $C$: Cost ratio (transaction fee / 100)
    - $I_h$: Binary variable indicating if a trade occurred at horizon $h$
    """
    
    def solve_trading_mip(self,
                          probabilities: list[Probability],
                          current_open_positions: int) -> tuple[int, int, float]:
        """Solve the Mixed Integer Programming problem to determine the optimal trading action.
        
        Args:
            probabilities (list[Probability]): A list of Probability dictionaries containing 
                probability distribution over 3 classes for each lob-transformer model.
            current_open_positions (int): The current number of open positions.
        Returns:
            tuple[int, int, float]: The optimal trading action, the next number of open positions, and the objective value.
        """
        num_horizons = len(probabilities)
        
        # Represets the expected value at horizon steps ahead from the current point
        cumulative_returns = [
            (prob["distribution"]["up"] - prob["distribution"]["down"]) * prob["threshold"] for prob in probabilities
        ]
        # Represents the expected value between the previous horizon and the next horizon
        interval_returns = [cumulative_returns[0]] + [
            cumulative_returns[i] - cumulative_returns[i - 1] for i in range(1, num_horizons)
        ]
        
        # Apply decay to interval returns to account for lower prediction accuracy at longer horizons
        interval_returns = [
            ret * (self.prediction_decay ** h) for h, ret in enumerate(interval_returns)
        ]
        
        problem = pulp.LpProblem("TradingOptimization", pulp.LpMaximize)
        
        # a[horizon] : action at horizon (SELL=-1, HOLD=0, BUY=1)
        action = pulp.LpVariable.dicts("action", range(num_horizons), lowBound=-1, upBound=1, cat=pulp.LpInteger)
        
        # p[horizon] : position at horizon
        position = pulp.LpVariable.dicts("pos", range(num_horizons), lowBound=-self.max_positions, upBound=self.max_positions, cat=pulp.LpInteger)
        
        # is_traded[horizon] : binary variable indicating if a trade is made at horizon
        is_traded = pulp.LpVariable.dicts("is_traded", range(num_horizons), cat=pulp.LpBinary)
        
        cost_ratio = self.transaction_fee / 100
        spread_ratio = self.spread / 100
        cost = (cost_ratio + (spread_ratio / 2)) * self.transaction_size
        
        j = pulp.lpSum([
            (position[h] * self.transaction_size * interval_returns[h]) 
                - (cost * is_traded[h]) 
            for h in range(num_horizons)
        ])
        problem += j
        
        for h in range(num_horizons):
            previous_open_positions = current_open_positions if h == 0 else position[h - 1]
            problem += position[h] == previous_open_positions + action[h]
            problem += is_traded[h] >= action[h]
            problem += is_traded[h] >= -action[h]
        
        problem.solve(pulp.PULP_CBC_CMD(msg=0))
        
        optimal_action = int(pulp.value(action[0]))
        next_open_positions = current_open_positions + optimal_action
        
        return (
            optimal_action,
            next_open_positions,
            pulp.value(problem.objective)
        )
    
    def predict_probabilities(self,
                              lob_snapshot: pd.DataFrame) -> list[Probability]:
        """Predict probability distributions for price movements using LOB-Transformer models.
        
        Args:
            lob_snapshot (pd.DataFrame): A snapshot of the limit order book data.\n
                Expected columns:
                - timestamp
                - mid_price
                - ask_price_1, ask_size_1, bid_price_1, bid_size_1
                - ...
                - ask_price_10, ask_size_10, bid_price_10, bid_size_10
        
        Returns:
            list[Probability]: A list of Probability dictionaries containing 
                probability distribution over 3 classes for each lob-transformer model.
        """
        probabilities: list[Probability] = []
        
        for lob_transformer in self.lob_transformers:
            dataset_config: LOBDatasetConfig = lob_transformer.hparams.dataset_config
            dataset = LOBDataset(lob_snapshot.copy(), **{
                **dataset_config.__dict__,
                "target_cols": [],
            })
            
            x, _ = dataset[-1]
            x = (x
                .unsqueeze(0)
                .to(self.device))
            
            with torch.no_grad():
                logits = lob_transformer(x)
                _probabilities = torch.softmax(logits, dim=1)[0]
            
            probability: Probability = {
                "horizon": dataset_config.horizon,
                "threshold": dataset_config.threshold,
                "distribution": {
                    "down": _probabilities[0].item(),
                    "stay": _probabilities[1].item(),
                    "up": _probabilities[2].item(),
                }
            }
            
            probabilities.append(probability)
        
        return probabilities
    
    def __init__(self,
                 lob_transformers: list[LOBTransformer],
                 transaction_fee: float,
                 transaction_size: float,
                 max_positions: int,
                 spread: float,
                 prediction_decay: float,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.lob_transformers = lob_transformers
        self.transaction_fee = transaction_fee
        self.transaction_size = transaction_size
        self.max_positions = max_positions
        self.spread = spread
        self.prediction_decay = prediction_decay
        self.device = device


if __name__ == "__main__":
    # Example to use
    
    lob_transformers = []
    transaction_fee = 0.01
    transaction_size = 1
    max_positions = 1
    spread = 0.00
    prediction_decay = 1.0
    
    mpc = MPC(
        lob_transformers,
        transaction_fee=transaction_fee,
        transaction_size=transaction_size,
        max_positions=max_positions,
        spread=spread,
        prediction_decay=prediction_decay
    )
    
    dummy_probabilities = [
        {
            "horizon": h,
            "threshold": h * 0.5e-4,
            "distribution": {
                "down": probability[0],
                "stay": probability[1],
                "up": probability[2]
            },
        } for h, probability in enumerate([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.3, 0.4, 0.3]], start=1)
    ]
    
    optimal_action, next_open_positions, objective_value = mpc.solve_trading_mip(
        dummy_probabilities,
        current_open_positions=0
    )
    
    print(f"Optimal action: {optimal_action}, Next open positions: {next_open_positions}, Objective value: {objective_value}")