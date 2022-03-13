import math
import os
from typing import Dict, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_parameter_depend_in_data_set
from utils.utils import reverse_normalize
from utils.visualize import save_data_as_plot

from metrics.base import Metric
import skimage.measure


class BlockInsertionDeletion(Metric):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        step: int,
        device: torch.device,
        dataset: str,
        block_size: int = 32,
    ) -> None:
        self.total = 0
        self.total_insertion = 0
        self.total_deletion = 0
        self.class_insertion = {}
        self.num_by_classes = {}
        self.class_deletion = {}

        self.model = model
        self.batch_size = batch_size
        self.step = step
        self.device = device
        self.block_size = block_size
        self.dataset = dataset

    def evaluate(
        self,
        image: np.ndarray,
        attention: np.ndarray,
        label: Union[np.ndarray, torch.Tensor],
    ) -> None:
        self.image = image.copy()
        self.label = int(label.item())

        # image (C, W, H), attention (1, W', H') -> attention (W, H)
        self.attention = attention
        if not (self.image.shape[1:] == attention.shape):
            self.attention = cv2.resize(
                attention[0], dsize=(self.image.shape[1], self.image.shape[2])
            )

        # attentionをパッチにしてパッチの順位を計算
        self.divide_attention_map_into_patch()
        self.calculate_attention_order()

        # insertion用の入力を作成してinference
        self.generate_insdel_images(mode="insertion")
        self.ins_preds = self.inference()  # for plot
        self.ins_auc = auc(self.ins_preds)
        self.total_insertion += self.ins_auc

        # deletion
        self.generate_insdel_images(mode="deletion")
        self.del_preds = self.inference()
        self.del_auc = auc(self.del_preds)
        self.total_deletion += self.del_auc

        # クラスごとのサンプル数 / insdelスコアの集計
        self.num_by_classes.setdefault(self.label, 0)
        self.class_insertion.setdefault(self.label, 0)
        self.class_deletion.setdefault(self.label, 0)

        self.num_by_classes[self.label] += 1
        self.class_insertion[self.label] += self.ins_auc
        self.class_deletion[self.label] += self.del_auc

        self.total += 1

    def divide_attention_map_into_patch(self):
        assert self.attention is not None

        self.block_attention = skimage.measure.block_reduce(
            self.attention, (self.block_size, self.block_size), np.max
        )

    def calculate_attention_order(self):
        attention_flat = np.ravel(self.block_attention)
        # 降順にするためマイナス （reverse）
        order = np.argsort(-attention_flat)

        W, H = self.attention.shape
        block_w, block_h = W // self.block_size, H // self.block_size
        self.order = np.apply_along_axis(
            lambda x: map_2d_indices(x, block_w), axis=0, arr=order
        )

    def generate_insdel_images(self, mode: str):
        W, H = self.attention.shape
        block_w, block_h = W // self.block_size, H // self.block_size
        num_insertion = math.ceil(block_w * block_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        self.input = np.zeros((num_insertion, 3, W, H))
        image = reverse_normalize(self.image.copy(), params["mean"], params["std"])

        for i in range(num_insertion):
            step_index = self.step * (i + 1)
            w_indices = self.order[0, step_index]
            h_indices = self.order[1, step_index]
            threthold = self.block_attention[w_indices, h_indices]

            if mode == "insertion":
                mask = np.where(threthold <= self.block_attention, 1, 0)
            elif mode == "deletion":
                mask = np.where(threthold <= self.block_attention, 0, 1)

            mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            self.input[i, 0] = (image[0] * mask - params["mean"][0]) / params["std"][0]
            self.input[i, 1] = (image[1] * mask - params["mean"][1]) / params["std"][1]
            self.input[i, 2] = (image[2] * mask - params["mean"][2]) / params["std"][2]

    def inference(self):
        inputs = torch.Tensor(self.input)

        num_iter = math.ceil(inputs.size(0) / self.batch_size)
        result = torch.zeros(0)

        for iter in range(num_iter):
            start = self.batch_size * iter
            inputs = inputs[start : start + self.batch_size].to(self.device)

            outputs = self.model(inputs)

            outputs = F.softmax(outputs, 1)
            outputs = outputs[:, self.label]
            result = torch.cat([result, outputs.cpu().detach()], dim=0)

        return np.nan_to_num(result)

    def score(self) -> Dict[str, float]:
        return {
            "Insertion": self.insertion(),
            "Deletion": self.deletion(),
            "PID": self.insertion() - self.deletion(),
        }

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


def map_2d_indices(indices_1d: int, width: int):
    """
    1次元のflattenされたindexを元に戻す
    index配列自体を二次元にするわけではなく、index番号のみ変換

    Args:
        indices_1d(array): インデックス
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
