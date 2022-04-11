import nntplib
import torch
import torch.nn as nn
import torch.nn.functional as F

def ib_loss(input_values, ib):
    """
    Atgs :
        input_values : 
        ib : 
    """
    loss = input_values * ib
    return loss.mean()


class IBLoss(nn.Module):
    def __init__(self, weight: float = None, num_classes: int = 2, alpha: float = 10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.num_classes = num_classes
        self.epsilon = 1e-3
        self.weight = weight
    
    def forward(self, input, target, features):
        """
        Args: 
            input : モデルへの入力
            target : ラベル
            features : FC層への入力
        """
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1) # N * 1
        ib = grads * features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction="none", weight=self.weight), ib)
        