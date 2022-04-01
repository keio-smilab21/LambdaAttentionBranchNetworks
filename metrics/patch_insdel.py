import math
import os
from typing import Dict, Union

import cv2
from cv2 import blur
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_parameter_depend_in_data_set
from utils.utils import reverse_normalize
from utils.visualize import save_data_as_plot, save_image

from metrics.base import Metric
import skimage.measure


class PatchInsertionDeletion(Metric):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        patch_size: int,
        step: int,
        dataset: str,
        device: torch.device,
        mask_mode: str,
    ) -> None:
        self.total = 0
        self.total_insertion = 0
        self.total_deletion = 0
        self.class_insertion: Dict[int, float] = {}
        self.num_by_classes: Dict[int, int] = {}
        self.class_deletion: Dict[int, float] = {}

        self.model = model
        self.batch_size = batch_size
        self.step = step
        self.device = device
        self.patch_size = patch_size
        self.dataset = dataset

        self.mask_mode = mask_mode

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
        self.generate_insdel_images(mode="insertion", mask_mode=self.mask_mode)
        self.ins_preds = self.inference()  # for plot
        self.ins_auc = auc(self.ins_preds)
        self.total_insertion += self.ins_auc

        # deletion
        self.generate_insdel_images(mode="deletion", mask_mode=self.mask_mode)
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

        self.patch_attention = skimage.measure.block_reduce(
            self.attention, (self.patch_size, self.patch_size), np.max
        )

    def calculate_attention_order(self):
        attention_flat = np.ravel(self.patch_attention)
        # 降順にするためマイナス （reverse）
        order = np.argsort(-attention_flat)

        W, H = self.attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        self.order = np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )

    def generate_insdel_images(self, mode: str, mask_mode: str="black"):
        C, W, H = self.image.shape
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        self.input = np.zeros((num_insertion, C, W, H))
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(self.image.copy(), mean, std)

        for i in range(num_insertion):
            # self.order.shape[1] = (2, N)
            step_index = min(self.step * (i + 1), self.order.shape[1] - 1)
            w_indices = self.order[0, step_index]
            h_indices = self.order[1, step_index]
            threthold = self.patch_attention[w_indices, h_indices]

            if mask_mode == "blur":
                src_image = image # 元画像(3, 224, 224) -> RGB
                blur_image = cv2.blur(image.transpose(1,2,0), (self.patch_size, self.patch_size)) # (224, 224, 3)
                if len(blur_image.shape) == 3:
                    blur_image = blur_image.transpose(2,0,1) # (3, 224, 224)
                
                if mode == "insertion":
                    mask_src = np.where(threthold <= self.patch_attention, 1, 0)
                    mask_blur = np.where(threthold <= self.patch_attention, 0, 1)
                elif mode == "deletion":
                    mask_src = np.where(threthold <= self.patch_attention, 0, 1)
                    mask_blur = np.where(threthold <= self.patch_attention, 1, 0)

                mask_src = cv2.resize(mask_src, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
                mask_blur = cv2.resize(mask_blur, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

                for c in range(C):
                    self.input[i, c] = src_image[c] * mask_src +  blur_image[c] * mask_blur
                
                if i%10 == 0:
                    # img = self.input[i].transpose(1,2,0)
                    img = (src_image*mask_src + blur_image*mask_blur).transpose(1,2,0)
                    # img = blur_image.transpose(1,2,0)
                    fig, ax = plt.subplots()
                    im = ax.imshow(img, vmin=0, vmax=1)
                    fig.colorbar(im)
                    plt.savefig(f"{mode}/self.input[{i}].png")
                    # plt.savefig(f"{mode}/blur_image[{i}].png")
                    plt.clf()
                    plt.close()

            else:
                if mode == "insertion":
                    mask = np.where(threthold <= self.patch_attention, 1, 0)
                elif mode == "deletion":
                    mask = np.where(threthold <= self.patch_attention, 0, 1)

                mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

                for c in range(C):
                    if mask_mode == "black":
                        self.input[i, c] = (image[c] * mask - mean[c]) / std[c]
                    elif mask_mode == "mean":
                        self.input[i, c] = mask * ((image[c] - mean[c]) / std[c])
                    
                    if i%10 == 0:
                        img = self.input[i].transpose(1,2,0)
                        # img = (src_image*mask_src + blur_image*mask_blur).transpose(1,2,0)
                        # img = blur_image.transpose(1,2,0)
                        fig, ax = plt.subplots()
                        im = ax.imshow(img, vmin=0, vmax=1)
                        fig.colorbar(im)
                        plt.savefig(f"{mode}/self.input[{i}].png")
                        # plt.savefig(f"{mode}/blur_image[{i}].png")
                        plt.clf()
                        plt.close()
                    
                    

    def inference(self):
        inputs = torch.Tensor(self.input)

        num_iter = math.ceil(inputs.size(0) / self.batch_size)
        result = torch.zeros(0)

        for iter in range(num_iter):
            start = self.batch_size * iter
            batch_inputs = inputs[start : start + self.batch_size].to(self.device)

            outputs = self.model(batch_inputs)

            outputs = F.softmax(outputs, 1)
            outputs = outputs[:, self.label]
            result = torch.cat([result, outputs.cpu().detach()], dim=0)

        return np.nan_to_num(result)

    def save_images(self):
        params = get_parameter_depend_in_data_set(self.dataset)
        for i, image in enumerate(self.input):
            save_image(image, f"tmp/{self.total}_{i}", params["mean"], params["std"])

    def score(self) -> Dict[str, float]:
        result = {
            "Insertion": self.insertion(),
            "Deletion": self.deletion(),
            "PID": self.insertion() - self.deletion(),
        }

        for class_idx in self.class_insertion.keys():
            class_ins = self.class_insertion_score(class_idx)
            class_del = self.class_deletion_score(class_idx)
            class_result = {
                f"Insertion_{class_idx}": class_ins,
                f"Deletion_{class_idx}": class_del,
                f"PID_{class_idx}": class_ins - class_del,
            }
            result.update(class_result)

        return result

    def log(self) -> str:
        result = "Class\tPID\tIns\tDel\n"

        scores = self.score()
        result += f"All\t{scores['PID']:.3f}\t{scores['Insertion']:.3f}\t{scores['Deletion']:.3f}\n"

        for class_idx in self.class_insertion.keys():
            pid = scores[f"PID_{class_idx}"]
            insertion = scores[f"Insertion_{class_idx}"]
            deletion = scores[f"Deletion_{class_idx}"]
            result += f"{class_idx}\t{pid:.3f}\t{insertion:.3f}\t{deletion:.3f}\n"

        return result

    def insertion(self) -> float:
        return self.total_insertion / self.total

    def deletion(self) -> float:
        return self.total_deletion / self.total

    def class_insertion_score(self, class_idx: int) -> float:
        num_samples = self.num_by_classes[class_idx]
        inserton_score = self.class_insertion[class_idx]

        return inserton_score / num_samples

    def class_deletion_score(self, class_idx: int) -> float:
        num_samples = self.num_by_classes[class_idx]
        deletion_score = self.class_deletion[class_idx]

        return deletion_score / num_samples

    def clear(self) -> None:
        self.total = 0
        self.ins_preds = None
        self.del_preds = None

    def save_roc_curve(self, save_dir: str) -> None:
        ins_fname = os.path.join(save_dir, f"{self.total}_insertion.png")
        save_data_as_plot(self.ins_preds, ins_fname, label=f"AUC = {self.ins_auc:.4f}")

        del_fname = os.path.join(save_dir, f"{self.total}_deletion.png")
        save_data_as_plot(self.del_preds, del_fname, label=f"AUC = {self.del_auc:.4f}")


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

def inverse_channel(image):
    """
    チャンネルの順序を変換する関数
    (r, g, b) -> (b, g, r)
    (b, g, r) -> (r, g, b)

    Input : (3, XX, XX)を想定
    """
    return image[::-1, ...]

def save_mask_image(img,  mode):
        # img = self.input[i].transpose(1,2,0)
        # img = (src_image*mask_src + blur_image*mask_blur).transpose(1,2,0)
        # img = blur_image.transpose(1,2,0)
        fig, ax = plt.subplots()
        im = ax.imshow(img, vmin=0, vmax=1)
        fig.colorbar(im)
        plt.savefig(f"{mode}/self.input[{i}].png")
        # plt.savefig(f"{mode}/blur_image[{i}].png")
        plt.clf()
        plt.close()


def save_pid_image(i, image, mask=None, mode="temp"):
    """
    Insertion / Deletionの画像を保存するようの関数
    """
    if mask is None:
        fig, ax = plt.subplots()
        im = ax.imshow(img, vmin=0, vmax=1)
        fig.colorbar(im)
        plt.savefig(f"{mode}/self.input[{i}].png")
        plt.clf()
        plt.close()
    else:
        # img = self.input[i].transpose(1,2,0)
        img = (image*mask).transpose(1,2,0)
        fig, ax = plt.subplots()
        im = ax.imshow(img, vmin=0, vmax=1)
        fig.colorbar(im)
        plt.savefig(f"{mode}/self.input[{i}].png")
        plt.clf()
        plt.close()

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
