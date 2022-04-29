from tkinter.tix import CELL
from cv2 import CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import calculate_loss, criterion_with_cast_targets

class SingleBCE(nn.Module):
    def __init__(self, pos_weight):
        super(SingleBCE, self).__init__()

        self.pos_weight = pos_weight
        self.BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.data_num = 0
    
    def forward(self, outputs, targets, model=None, lambdas=None):
        self.data_num = targets.size()[0]
        self.loss = calculate_loss(self.BCE, outputs, targets, model, lambdas)

        return self.loss

class DoubleBCE(nn.Module):
    def __init__(self, pos_weight, alpha=1.0):
        super(DoubleBCE, self).__init__()
        
        self.pos_weight = pos_weight
        self.alpha = alpha
        
        self.BCE_orig = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.BCE_mask = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, outputs_mask, targets, mask_targets, model=None, lambdas=None):
        self.loss_orig = calculate_loss(self.BCE_orig, outputs, targets, model, lambdas)
        self.loss_mask = criterion_with_cast_targets(self.BCE_mask, outputs_mask, mask_targets)

        return self.loss_orig + self.alpha * self.loss_mask

class BCEWithKL(nn.Module):
    def __init__(self, pos_weight, alpha=1.0):
        super(BCEWithKL, self).__init__()
        
        self.pos_weight = pos_weight
        self.alpha = alpha

        self.BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.KL = nn.KLDivLoss(log_target=True, reduction="batchmean")
    
    def forward(self, outputs, outputs_KL, targets, model=None, lambdas=None):
        self.loss_BCE = calculate_loss(self.BCE, outputs, targets, model, lambdas)
        self.loss_KL = criterion_with_cast_targets(self.KL, outputs_KL, outputs)

        return self.loss_BCE + self.alpha * self.loss_KL

class VillaKL(nn.Module):
    def __init__(self, pos_weight, alpha=1.0, beta=1.0):
        super(VillaKL, self).__init__()

        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta

        self.BCE_orig = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.BCE_mask = nn.BCEWithLogitsLoss()
        self.KL = nn.KLDivLoss(log_target=True, reduction="batchmean")

    def forward(self, outputs, outputs_mask_KL, outputs_mask_Villa, targets, mask_targets_Villa, model=None, lambdas=None):
        self.loss_BCE_orig = calculate_loss(self.BCE_orig, outputs, targets, model, lambdas)
        self.loss_BCE_mask = criterion_with_cast_targets(self.BCE_mask, outputs_mask_Villa, mask_targets_Villa)
        self.loss_KL = criterion_with_cast_targets(self.KL, outputs_mask_KL, outputs)

        return self.loss_BCE_orig + self.alpha * self.loss_BCE_mask + self.beta * self.loss_KL