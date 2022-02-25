import math
import os
from typing import Dict, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_dataset_params
from utils.visualize import save_data_as_plot, save_image

from metrics.base import Metric


class InsertionDeletion(Metric):
    def __init__(self) -> None:
        self.total = 0
        self.total_insertion = 0
        self.total_deletion = 0
        self.class_insertion = {}
        self.num_by_classes = {}
        self.class_deletion = {}

    def evaluate(
        self,
        model: nn.Module,
        image: np.ndarray,
        attention: np.ndarray,
        label: Union[np.ndarray, torch.Tensor],
        batch_size: int,
        step: int,
        device: torch.device,
        block_size: int = 32,
    ) -> None:
        self.ins_preds = insertion_deletion(
            model,
            image,
            attention,
            batch_size,
            label,
            mode="insertion",
            step=step,
            device=device,
        )
        self.ins_auc = auc(self.ins_preds)
        self.total_insertion += self.ins_auc

        label_int = label.item()
        if not label_int in self.class_insertion:
            self.class_insertion[label_int] = self.ins_auc
            self.num_by_classes[label_int] = 1
        else:
            self.class_insertion[label_int] += self.ins_auc
            self.num_by_classes[label_int] += 1

        self.del_preds = insertion_deletion(
            model,
            image,
            attention,
            batch_size,
            label,
            mode="deletion",
            step=step,
            device=device,
        )
        self.del_auc = auc(self.del_preds)
        self.total_deletion += self.del_auc

        if not label_int in self.class_deletion:
            self.class_deletion[label_int] = self.del_auc
        else:
            self.class_deletion[label_int] += self.del_auc

        self.total += 1

    def score(self) -> Dict[str, float]:
        return {"Insertion": self.insertion(), "Deletion": self.deletion()}

    def log(self) -> str:
        result = ""
        scores = self.score()
        for name, score in scores.items():
            result += f"{name}: {score:.3f} "

        result += "\n Insertion\t"
        for class_id, insertion_score in self.class_insertion.items():
            class_total = self.num_by_classes[class_id]
            insertion_score /= class_total
            result += f"{class_id} ({class_total}): {insertion_score:.3f} "

        result += "\n Deletion\t"
        for class_id, deletion_score in self.class_deletion.items():
            class_total = self.num_by_classes[class_id]
            deletion_score /= class_total
            result += f"{class_id} ({class_total}): {deletion_score:.3f} "

        # 最後の空白を削る
        return result[:-1]

    def insertion(self) -> float:
        return self.total_insertion / self.total

    def deletion(self) -> float:
        return self.total_deletion / self.total

    def clear(self) -> None:
        self.total = 0
        self.ins_preds = None
        self.del_preds = None

    def save_roc_curve(self, save_dir: str) -> None:
        scores = self.score()
        ins_fname = os.path.join(save_dir, f"{self.total}_insertion.png")
        save_data_as_plot(self.ins_preds, ins_fname, label=f"AUC = {self.ins_auc:.4f}")

        del_fname = os.path.join(save_dir, f"{self.total}_deletion.png")
        save_data_as_plot(self.del_preds, del_fname, label=f"AUC = {self.del_auc:.4f}")

    def log_gspread(self):
        num_normal = self.num_by_classes[0]
        num_abnormal = self.num_by_classes[1]

        ins_normal = self.class_insertion[0] / num_normal
        ins_abnormal = self.class_insertion[1] / num_abnormal

        del_normal = self.class_deletion[0] / num_normal
        del_abnormal = self.class_deletion[1] / num_abnormal

        return f"{ins_normal}\t{ins_abnormal}\t{self.insertion()}\t{del_normal}\t{del_abnormal}\t{self.deletion()}"


def generate_insdel_images(
    image: Union[np.ndarray, torch.Tensor],
    attention: Union[np.ndarray, torch.Tensor],
    mode: str,
    step: int = 10,
) -> np.ndarray:
    """
    insertion / deletion用の画像を作成

    Args:
        image (array)   : 画像
        attention(array): アテンションマップ
        mode(str)       : insertion / deletion
        step(int)       : 高い順にいくつずつ足す/消していくか

    Returns:
        np.ndarray: (N, C, H, W)
                    N = ceil(W * H / step)
                    バッチが各段階に対応
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().detach().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    if not (image.shape[1:] == attention.shape):
        attention = cv2.resize(attention[0], dsize=(image.shape[1], image.shape[2]))

    W, H = attention.shape

    attention_flat = np.ravel(attention)
    # argsortにはreverseがないが降順にしたいためマイナス付与
    order = np.argsort(-attention_flat)
    order = np.apply_along_axis(lambda x: map_2d_indices(x, W), axis=0, arr=order)

    num_insertion = math.ceil(W * H / step)

    # insertionはresultに足していく
    # deletionはresult <- image <- zero
    result = np.zeros((num_insertion, 3, W, H))
    params = get_dataset_params("magnetogram")
    zero_image = np.zeros((3, W, H))

    for i in range(num_insertion):
        w_indices = order[0, : step * (i + 1)]
        h_indices = order[1, : step * (i + 1)]

        if mode == "insertion":
            result[i, :, w_indices, h_indices] = np.transpose(
                image[:, w_indices, h_indices]
            )
        elif mode == "deletion":
            result[i] = image
            result[i, :, w_indices, h_indices] = np.transpose(
                zero_image[:, w_indices, h_indices]
            )

        save_image(result[i], f"insdel/{mode}_{i}.png", params["mean"], params["std"])

    return result


def insertion_deletion(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    attention: np.ndarray,
    batch_size: int,
    label: Union[np.ndarray, torch.Tensor],
    mode: str,
    step: int,
    device: torch.device,
) -> np.ndarray:
    """
    insertion / deletionを実行

    Args:
        model (Module)  : 推論を行うモデル
        image (array)   : 画像
        attention(array): アテンションマップ
        batch_size(int) : バッチサイズ
        label(array)    : ラベル
        mode(str)       : insertion / deletion
        step(int)       : 高い順にいくつずつ足す/消していくか
        device(device)  : CPU / GPU

    Returns:
        np.ndarray: 各stepでの予測確率
    """
    insertion_images = generate_insdel_images(image, attention, mode, step=step)
    insertion_images = torch.Tensor(insertion_images)

    num_iter = math.ceil(insertion_images.size(0) / batch_size)
    result = torch.zeros(0)

    for iter in range(num_iter):
        start = batch_size * iter
        inputs = insertion_images[start : start + batch_size].to(device)

        outputs = model(inputs)

        outputs = F.softmax(outputs, 1)
        outputs = outputs[:, label]
        # result = torch.hstack((result, outputs.cpu().detach()))
        result = torch.cat([result, outputs.cpu().detach()], dim=0)

    return np.nan_to_num(result)


def map_2d_indices(indices_1d: int, width: int):
    """
    1次元のflattenされたindexを元に戻す
    index配列自体を二次元にするわけではなく、index番号のみ変換

    Args:
        indices_1d(array): インデックス配列
        width(int)       : 横幅

    Examples:
        [[0, 1, 2], [3, 4, 5]]
        -> [0, 1, 2, 3, 4, 5]

        map_2d_indices(1, 3)
        >>> [0, 1]
        map_ed_indices(5, 3)
        >>> [1, 2]

        flattenする前の配列のindexを返す
    """
    return [indices_1d // width, indices_1d % width]


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
