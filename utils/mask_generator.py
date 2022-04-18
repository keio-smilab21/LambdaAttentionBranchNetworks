import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import skimage.measure

from data import get_parameter_depend_in_data_set
from models.attention_branch import AttentionBranchModel
from models.lambda_resnet import Bottleneck
from models.rise import RISE
from utils.utils import module_generator, reverse_normalize

class Mask_Generator():
    def __init__(
        self,
        model: nn.Module,
        params,
        images, # 複数であることに注意
        labels, # 複数であることに注意
        method,
        patch_size,
        step,
        dataset,
        device,
        mask_mode,
        attention_dir,
    ) -> None:

        self.model = model
        self.params = params
        self.images = images
        self.labels = labels
        self.method = method
        self.step = step
        self.device = device
        self.patch_size = patch_size
        self.dataset = dataset
        self.mask_mode = mask_mode
        self.attention_dir = attention_dir

    def create_mask_inputs(self):
        mask_inputs = np.zeros(shape=self.images.shape) # (64, 1, 512, 512)
        attentions = [] # len : 64
        patch_attentions = [] # len : 64
        orders = [] # len 64
        
        for i in range(self.images.shape[0]):
            label = self.labels[i]
            input = self.images[i] # (1, 512, 512)
            input = torch.unsqueeze(input, 0) # (1, 1, 512, 512)
                
            # attentionの値の計算
            # attention : (512, 512)
            attention = calculate_attention(self.model, input, i)
            if not (self.images[i].shape[1:] == attention.shape):
                attention = cv2.resize(
                    attention[0], dsize=(self.images[i].shape[1], self.images[i].shape[2])
                )
            attentions.append(attention) # (512, 512)

            # attentionをパッチにしてパッチごとの順位を計算
            # patch_attentions.append(self.divide_attention_map_into_patch(attentions[i])) # (512, 512)
            patch_attentions = attentions
            orders.append(self.calculate_attention_order(attentions[i], patch_attentions[i])) # (2, 262144)

            # mask_inputを作成して返す
            save_mask_image = True if i % 10 == 0 else False
            mask_input = self.create_mask_image(self.images[i].cpu().numpy(), attentions[i], patch_attentions[i], orders[i], mask_mode=self.mask_mode, save_mask_image=save_mask_image)
            mask_inputs[i] = mask_input
        
        if i == 30:
            # # mask 画像の保存
            # print("mask_image.shape : ", mask_img.shape) #(1, 512, 512)
            img = mask_input.reshape(self.images.shape[2], self.images.shape[3])
            # img = image.reshape(512, 512)
            fig, ax = plt.subplots()
            if img.shape[-1] == 1:
                im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            else:
                im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
            fig.colorbar(im)
            plt.savefig(f"mask_image/deletion/self.input.png")
            plt.clf()
            plt.close()

        # (64, 1, 512, 512)
        return mask_inputs


    def divide_attention_map_into_patch(self, attention):
        assert attention is not None

        self.patch_attention = skimage.measure.block_reduce(
            attention, (self.patch_size, self.patch_size), np.max
        ) # (32, 32) = (512/patch_size, 512/patch_size)


    def calculate_attention_order(self, attention, patch_attention):
        """
        attentionの順番を計算する
        昇順のままでいいので、-消していることに注意 -> やめた
        降順のままにしてdeletionとinsertionを逆にすればいいかな
        """
        attention_flat = np.ravel(patch_attention)
        # 昇順でいいのでそのまま
        order = np.argsort(-attention_flat)

        W, H = attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        return np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )


    def create_mask_image(self, image, attention, patch_attention, attention_order, mask_mode: str="base", save_mask_image: bool = False):
        ratio = 0.1
        
        # TODO ここってなにやってるん
        if not (image.shape[1:] == attention.shape):
            attention = cv2.resize(
                attention[0], dsize=(image.shape[1], image.shape[2])
            )
        C, W, H = image.shape # (1, 512, 512)
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step) # num_insertion : 27 when img_size512, step 10000

        params = get_parameter_depend_in_data_set(self.dataset)
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(image.copy(), mean, std)

        # ここからした実験 : num_insertionの半分にしてる
        step_index = min(self.step * (int(num_insertion*ratio + 1)), attention_order.shape[1] - 1) # 220000
        w_indices = attention_order[0, step_index]
        h_indices = attention_order[1, step_index]
        threthold = patch_attention[w_indices, h_indices]

        mask_img = np.zeros(shape=image.shape) # (1, 512, 512)

        if mask_mode in ["blur", "base"]:
            src_image = image # 元画像(3, 224, 224) -> RGB

            if mask_mode == "blur":
                base_mask_image = cv2.blur(image.transpose(1,2,0), (self.patch_size, self.patch_size)) # (224, 224, 3)
            elif mask_mode == "base":
                base_mask_image = Image.open("./datasets/magnetogram/bias_image.png").resize((H, W)) # ここまではOK
                base_mask_image = np.asarray(base_mask_image, dtype=np.float32) / 255.0 # ここまではOK (512, 512)

            if len(base_mask_image.shape) == 3:
                print("transform base_mask_image")
                base_mask_image = base_mask_image.transpose(2, 0, 1)
            if len(base_mask_image.shape) == 2:
                base_mask_image = base_mask_image.reshape(C, H, W)

            mask_src = np.where(threthold <= patch_attention, 1.0, 0.0)
            mask_base = np.where(threthold <= patch_attention, 0.0, 1.0)
            
            mask_src = cv2.resize(mask_src, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            mask_base = cv2.resize(mask_base, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                mask_img[c] = src_image[c] * mask_src +  base_mask_image[c] * mask_base
                mask_img[c] = (mask_img[c] - mean[c]) / std[c]
            
        else:
            mask = np.where(threthold <= patch_attention, 1.0, 0.0)
            mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                if mask_mode == "black":
                    mask_img[c] = (image[c] * mask - mean[c]) / std[c]
                elif mask_mode == "mean":
                    mask_img[c] = mask * ((image[c] - mean[c]) / std[c])


        # # mask 画像の保存
        # if save_mask_image:
        #     # print("mask_image.shape : ", mask_img.shape) #(1, 512, 512)
        #     img = mask_img.reshape(H, W)
        #     # img = image.reshape(512, 512)
        #     fig, ax = plt.subplots()
        #     if img.shape[-1] == 1:
        #         im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        #     else:
        #         im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        #     fig.colorbar(im)
        #     plt.savefig(f"mask_image/deletion/self.input.png")
        #     plt.clf()
        #     plt.close()

        return mask_img # (1, 512, 512)


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


def calculate_attention(
    model: nn.Module,
    image: torch.Tensor, # (1, 1, 512, 512)
    idx: int
) -> Tuple[np.ndarray, bool]:

    assert isinstance(model, AttentionBranchModel)
    # y = model(image) # image : (1, 1, 512, 512)
    attentions = model.attention_branch.attention  # (32, 1, 512, 512)
    attention = attentions[idx]
    attention: np.ndarray = attention.cpu().clone().detach().numpy()
    # TODO ここにもともとはdetachないのに今回加えなければならないのはどゆこと

    # (1, 128, 128)
    return attention