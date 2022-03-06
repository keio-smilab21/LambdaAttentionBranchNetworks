import json
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from models.attention_branch import AttentionBranchModel
from torch import nn


def fix_seed(seed: int, deterministic: bool = False) -> None:
    """
    ランダムシードの固定

    Args:
        seed(int)          : 固定するシード値
        deterministic(bool): GPUに決定的動作させるか
                             Falseだと速いが学習結果が異なる
    """
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


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

    return criterion(preds, targets)


def calclurate_loss(
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


def reverse_normalize(
    x: np.ndarray,
    mean: Union[Tuple[float], Tuple[float, float, float]],
    std: Union[Tuple[float], Tuple[float, float, float]],
):
    """
    Normalizeを戻す

    Args:
        x(ndarray) : Normalizeした行列
        mean(Tuple): Normalize時に指定した平均
        std(Tuple) : Normalize時に指定した標準偏差
    """
    if x.shape[0] == 1:
        x = x * std + mean
        return x
    x[0, :, :] = x[0, :, :] * std[0] + mean[0]
    x[1, :, :] = x[1, :, :] * std[1] + mean[1]
    x[2, :, :] = x[2, :, :] * std[2] + mean[2]

    return x


def module_generator(model: nn.Module, reverse: bool = False):
    """
    入れ子可能なModuleのgenerator，入れ子Sequentialを1つで扱える
    インデックスによる層の取得ができないことに注意

    Args:
        model (nn.Module): モデル
        reverse(bool)    : 反転するか

    Yields:
        モデルの各層
    """
    modules = list(model.children())
    if reverse:
        modules = modules[::-1]

    for module in modules:
        if isinstance(module, nn.Sequential):
            yield from module_generator(module, reverse)
            continue
        yield module


def save_json(data: Union[List, Dict], save_path: str) -> None:
    """
    list/dictをjsonに保存

    Args:
        data (List/Dict): 保存するデータ
        save_path(str)  : 保存するパス（拡張子込み）
    """
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def softmax_image(image: torch.Tensor) -> torch.Tensor:
    image_size = image.size()
    if len(image_size) == 4:
        B, C, W, H = image_size
    elif len(image_size) == 3:
        B = 1
        C, W, H = image_size
    else:
        raise ValueError

    image = image.view(B, C, W * H)
    image = torch.softmax(image, dim=-1)

    image = image.view(B, C, W, H)
    if len(image_size) == 3:
        image = image[0]

    return image


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        result: np.ndarray = tensor.cpu().detach().numpy()
    else:
        result = tensor

    return result
