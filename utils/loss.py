from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from models.attention_branch import AttentionBranchModel
from torch import nn


def criterion_with_cast_targets(
    criterion: nn.modules.loss._Loss, preds: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    型を変更してから誤差関数の計算を行う

    Args:
        criterion(Loss): 誤差関数
        preds(Tensor)  : 予測
        targets(Tensor): ラベル

    Returns:
        torch.Tensor: 誤差関数の値

    Note:
        誤差関数によって要求される型が異なるため変換を行う
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        targets = F.one_hot(targets, num_classes=2)
        targets = targets.long()

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        targets = F.one_hot(targets, num_classes=2)
        targets = targets.to(preds.dtype)
    
    if isinstance(criterion, nn.KLDivLoss):
        targets = targets.to(preds.dtype)

    return criterion(preds, targets)


def calculate_loss(
    criterion: nn.modules.loss._Loss,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    lambdas: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    ロスの計算を行う
    AttentionBranchModelのときattention lossを加える

    Args:
        criterion(Loss)  : 誤差関数
        preds(Tensor)    : 予測
        targets(Tensor)  : ラベル
        model(nn.Module) : 予測を行ったモデル
        lambdas(Dict[str, float]): lossの各項の重み

    Returns:
        torch.Tensor: 誤差関数の値
    """
    loss = criterion_with_cast_targets(criterion, outputs, targets)

    # Attention Loss
    if isinstance(model, AttentionBranchModel):
        keys = ["att", "var"]
        if lambdas is None:
            lambdas = {key: 1 for key in keys}
        for key in keys:
            if key not in lambdas:
                lambdas[key] = 1

        attention_loss = criterion_with_cast_targets(
            criterion, model.attention_pred, targets
        )
        # loss = loss + attention_loss
        # attention = model.attention_branch.attention
        # _, _, W, H = attention.size()

        # att_sum = torch.sum(attention, dim=(-1, -2))
        # attention_loss = torch.mean(att_sum / (W * H))
        loss = loss + lambdas["att"] * attention_loss
        # attention = model.attention_branch.attention
        # attention_varmean = attention.var(dim=(1, 2, 3)).mean()
        # loss = loss - lambdas["var"] * attention_varmean

    return loss
