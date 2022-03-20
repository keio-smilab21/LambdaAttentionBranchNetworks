import argparse
import json
import random
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
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


def reverse_normalize(
    x: np.ndarray,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> np.ndarray:
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
        if list(module.children()):
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


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    argparseとjsonファイルのconfigを共存させる

    Args:
        parser(ArgumentParser)

    Returns:
        Namespace: argparserと同様に使用可能

    Note:
        configより引数に指定した値が優先
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args
