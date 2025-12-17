import enum

import torch


class Action(enum.Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class MPC:
    def optimize_trade_action(self,
                              probabilities: list[torch.Tensor]) -> Action:
        """Determine optimal trading action (buy, sell, hold) based on predicted probabilities.
        
        Args:
            probabilities (list[torch.Tensor]): A list of tensors containing 
                probability distribution over 3 classes for each lob-transformer model.
        Returns:
            Action: The optimal trading action.
        """
        pass
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Example to use
    mpc = MPC()