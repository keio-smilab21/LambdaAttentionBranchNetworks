import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import get_parameter_depend_in_data_set
from utils.utils import reverse_normalize

MASK_RATIO_CHOICES = [0.1 ,0.2, 0.25, 0.3]
WEIGHT = [0.2, 0.5, 0.2, 0.1]
PATCH_CHOICE = [1, 4, 8, 16, 32]
WEIGHT_PATCH = [0.7, 0.2, 0.05, 0.03, 0.02]

class Mask_Generator():
    def __init__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        attention: torch.Tensor,
        patch_size: int,
        step: int,
        dataset: Dataset,
        mask_mode: str,
        ratio_src_image: float = 0.1,
        save_mask_input: bool = False,
        data_name: str = "mognetogram",
    ) -> None:

        self.model = model
        self.images = images                    # B, C, H, W
        self.attention = attention.cpu().clone().detach().numpy()
        self.patch_size = patch_size
        self.step = step
        self.dataset = dataset
        self.mask_mode = mask_mode
        self.ratio = ratio_src_image
        self.save_mask_input = save_mask_input
        self.attentions = []                    # length: 64
        self.orders = []                        # length: 64
        self.data_name = data_name

    def create_mask_inputs(self):
        mask_inputs = np.zeros(shape=self.images.shape) # B, C, H, W

        for i in range(self.images.shape[0]):
            input = self.images[i]            # C, H, W
            input = torch.unsqueeze(input, 0) # 1, C, H, W

            attention = self.attention[i]     # 1, H', W'

            # attention map.shape -> origin_image.shape
            if not (self.images[i].shape[1:] == attention.shape):
                H, W = self.images[i].shape[1], self.images[i].shape[2]
                
                att_dtype = attention.dtype
                if att_dtype == np.float32:
                    attention = cv2.resize(attention[0], dsize=(H, W))
                elif att_dtype == np.float16:
                    attention = cv2.resize(attention[0].astype(np.float32), dsize=(H, W))

            self.attentions.append(attention)  # H, W

            # attentionの順位を計算
            self.orders.append(self.calculate_attention_order(idx=i))

            # マスク画像を作成 : 1, H, W
            mask_input = self.create_mask_image(idx=i)
            mask_inputs[i] = mask_input
        
        if self.save_mask_input:
            # mask 画像の保存
            print("mask_raito : ", self.ratio)
            img = mask_inputs[-1].reshape(self.images.shape[2], self.images.shape[3]) # (512, 512)
            fig, ax = plt.subplots()
            if img.shape[-1] == 1:
                im = ax.imshow(img, cmap="gray")
            else:
                im = ax.imshow(img, cmap="gray")
            fig.colorbar(im)
            plt.savefig(f"mask_image/deletion/mask_ratio_{self.ratio}_{self.data_name}.png")
            plt.clf()
            plt.close()

        return mask_inputs # B, C, H, W


    def calculate_attention_order(self, idx):
        """
        attentionの順番を計算する
        """
        attention = self.attentions[idx]
        attention_flat = np.ravel(attention)
        order = np.argsort(-attention_flat)

        W, H = attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        return np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )


    def create_mask_image(self, idx):
        image = self.images[idx].cpu().numpy()  # C, H, W
        attention = self.attentions[idx]        # H, W
        attention_order = self.orders[idx]      # 2, H*W

        C, W, H = image.shape
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(image.copy(), mean, std)

        step_index = min(self.step * (int(num_insertion*self.ratio + 1)), attention_order.shape[1] - 1)
        w_indices = attention_order[0, step_index]
        h_indices = attention_order[1, step_index]
        threthold = attention[w_indices, h_indices]

        mask_img = np.zeros(shape=image.shape)

        if self.mask_mode in ["blur", "base"]:
            src_image = image

            if self.mask_mode == "blur":
                base_mask_image = cv2.blur(image.transpose(1,2,0), (self.patch_size, self.patch_size))
            elif self.mask_mode == "base":
                base_mask_image = Image.open(f"./datasets/{self.data_name}/{self.data_name}_bias_image.png").resize((H, W))
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
                if self.mask_mode == "black":
                    mask_img[c] = (image[c] * mask - mean[c]) / std[c]
                elif self.mask_mode == "mean":
                    mask_img[c] = mask * ((image[c] - mean[c]) / std[c])

        return mask_img


def map_2d_indices(indices_1d: int, width: int):
    """
    1次元のflattenされたindexを元に戻す
    index配列自体を二次元にするわけではなく、index番号のみ変換
    """
    return [indices_1d // width, indices_1d % width]