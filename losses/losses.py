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
        
        self.BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.Villa = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, outputs, outputs_mask, targets, model=None, lambdas=None):
        self.loss_BCE = calculate_loss(self.BCE, outputs, targets, model, lambdas)
        villa = criterion_with_cast_targets(self.Villa, outputs_mask, targets)
        self.loss_Villa = (F.one_hot(targets, num_classes=2) * villa).mean()

        return self.loss_BCE + self.alpha * self.loss_Villa

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

        self.BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.Villa = nn.BCEWithLogitsLoss(reduction="none")
        self.KL = nn.KLDivLoss(log_target=True, reduction="batchmean")

    def forward(self, outputs, outputs_mask_KL, outputs_mask_Villa, targets, model=None, lambdas=None):
        self.loss_BCE = calculate_loss(self.BCE, outputs, targets, model, lambdas)
        villa = criterion_with_cast_targets(self.Villa, outputs_mask_Villa, targets)
        self.loss_Villa = (F.one_hot(targets, num_classes=2) * villa).mean()
        self.loss_KL = criterion_with_cast_targets(self.KL, outputs_mask_KL, outputs)

        return self.loss_BCE + self.alpha * self.loss_Villa + self.beta * self.loss_KL