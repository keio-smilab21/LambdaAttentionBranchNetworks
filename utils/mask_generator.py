import math
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from data import get_parameter_depend_in_data_set
from models.attention_branch import AttentionBranchModel
from utils.utils import reverse_normalize

class Mask_Generator():
    def __init__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        patch_size: int,
        step: int,
        dataset: Dataset,
        mask_mode: str,
        ratio_src_image: float = 0.1,
        save_mask_input: bool = False,
    ) -> None:

        self.model = model
        self.images = images
        self.patch_size = patch_size
        self.step = step
        self.dataset = dataset
        self.mask_mode = mask_mode
        self.ratio = ratio_src_image
        self.save_mask_input = save_mask_input
        self.attentions = [] # len 64
        self.orders = [] # len 64

    def create_mask_inputs(self):
        mask_inputs = np.zeros(shape=self.images.shape)

        for i in range(self.images.shape[0]):
            input = self.images[i]
            input = torch.unsqueeze(input, 0)
                
            # attentionの値の計算
            attention = return_attention(self.model, i)
            if not (self.images[i].shape[1:] == attention.shape):
                if attention[0].dtype == np.float32:
                    attention = cv2.resize(
                        attention[0], dsize=(self.images[i].shape[1], self.images[i].shape[2])
                    )
                elif attention[0].dtype == np.float16:
                    attention = cv2.resize(
                        attention[0].astype(np.float32), dsize=(self.images[i].shape[1], self.images[i].shape[2])
                    )
            self.attentions.append(attention) # (512, 512)

            # attentionの順位を計算
            self.orders.append(self.calculate_attention_order(self.attentions[i]))

            # mask_inputを作成 : (1, 512 ,512)
            mask_input = self.create_mask_image(idx=i, mask_mode=self.mask_mode)
            mask_inputs[i] = mask_input
        
        if self.save_mask_input:
            # mask 画像の保存
            img = mask_inputs[-1].reshape(self.images.shape[2], self.images.shape[3]) # (512, 512)
            fig, ax = plt.subplots()
            if img.shape[-1] == 1:
                im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            else:
                im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            fig.colorbar(im)
            plt.savefig(f"mask_image/deletion/mask_input_KL.png")
            plt.clf()
            plt.close()

        return mask_inputs # (64, 1, 512, 512)


    def calculate_attention_order(self, attention):
        """
        attentionの順番を計算する
        path_insdelとはdeletion / insertion を逆にする
        """
        attention_flat = np.ravel(attention)
        order = np.argsort(-attention_flat)

        W, H = attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        return np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )


    def create_mask_image(self, idx, mask_mode: str="base"):
        image = self.images[idx].cpu().numpy()
        attention = self.attentions[idx]
        attention_order = self.orders[idx]
        
        if not (image.shape[1:] == attention.shape):
            attention = cv2.resize(
                attention[0], dsize=(image.shape[1], image.shape[2])
            )
        C, W, H = image.shape # (1, 512, 512)
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(image.copy(), mean, std)

        step_index = min(self.step * (int(num_insertion*self.ratio + 1)), attention_order.shape[1] - 1)
        w_indices = attention_order[0, step_index]
        h_indices = attention_order[1, step_index]
        threthold = attention[w_indices, h_indices]

        mask_img = np.zeros(shape=image.shape) # (1, 512, 512)

        if mask_mode in ["blur", "base"]:
            src_image = image

            if mask_mode == "blur":
                base_mask_image = cv2.blur(image.transpose(1,2,0), (self.patch_size, self.patch_size))
            elif mask_mode == "base":
                base_mask_image = Image.open("./datasets/magnetogram/bias_image.png").resize((H, W))
                base_mask_image = np.asarray(base_mask_image, dtype=np.float32) / 255.0

            if len(base_mask_image.shape) == 3:
                print("transform base_mask_image")
                base_mask_image = base_mask_image.transpose(2, 0, 1)
            if len(base_mask_image.shape) == 2:
                base_mask_image = base_mask_image.reshape(C, H, W)

            mask_src = np.where(threthold <= attention, 1.0, 0.0)
            mask_base = np.where(threthold <= attention, 0.0, 1.0)
            
            mask_src = cv2.resize(mask_src, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            mask_base = cv2.resize(mask_base, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                mask_img[c] = src_image[c] * mask_src +  base_mask_image[c] * mask_base
                mask_img[c] = (mask_img[c] - mean[c]) / std[c]
            
        else:
            mask = np.where(threthold <= attention, 1.0, 0.0)
            mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                if mask_mode == "black":
                    mask_img[c] = (image[c] * mask - mean[c]) / std[c]
                elif mask_mode == "mean":
                    mask_img[c] = mask * ((image[c] - mean[c]) / std[c])

        return mask_img


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

def return_attention(
    model: nn.Module,
    idx: int
) -> Tuple[np.ndarray, bool]:

    assert isinstance(model, AttentionBranchModel)
    attentions = model.attention_branch.attention  # (32, 1, 512, 512)
    attention = attentions[idx]
    attention = attention.cpu().clone().detach().numpy()

    return attention