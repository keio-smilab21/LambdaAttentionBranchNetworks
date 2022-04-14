import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import calculate_loss

class MaskKL(nn.Module):
    def __init__(self, pos_weight):
        super(MaskKL, self).__init__()

        self.pos_weight = pos_weight

        self.BCE_orig = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.BCE_mask = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.KL = nn.KLDivLoss(reduction="batchmean")

    def forward(self, mode, outputs, targets, model=None, lambdas=None):
        if mode == "origin":
            loss_BCE_orig = calculate_loss(self.BCE_orig, outputs, targets, model, lambdas)
            return loss_BCE_orig
        elif mode == "maskBCE":
            loss_BCE_mask = self.BCE_mask(outputs, targets)
            return loss_BCE_mask
        elif mode == "KL":
            loss_KL = self.KL(outputs, targets)
            return loss_KL
