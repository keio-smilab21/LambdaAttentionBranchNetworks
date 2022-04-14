import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import calculate_loss, criterion_with_cast_targets

class MaskBCE(nn.Module):
    def __init__(self, pos_weight):
        super(MaskBCE, self).__init__()
        
        self.pos_weight = pos_weight
        
        self.BCE_orig = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.BCE_mask = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, mode, outputs, targets, model=None, lambdas=None):
        if mode == "origin":
            loss_orig = calculate_loss(self.BCE_orig, outputs, targets, model, lambdas)
            return loss_orig

        elif mode == "MaskBCE":
            loss_mask = criterion_with_cast_targets(self.BCE_mask, outputs, targets)
            return loss_mask