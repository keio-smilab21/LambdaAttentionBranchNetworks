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
        self.mask_inputs = np.zeros(shape=self.images.shape) # (8, 1, 512, 512)
        self.attentions = []
        self.patch_attentions = []
        self.orders = []
        
        for i in range(self.images.shape[0]):
            label = self.labels[i]
            input = self.images[i] # (1, 512, 512)
            input = torch.unsqueeze(input, 0)

            base_fname = f"{i+1}_{self.params['classes'][label]}"
            attention_fname = None
            if self.attention_dir is not None:
                attention_fname = os.path.join(self.attention_dir, f"{base_fname}.npy")
                
            # attentionの値の計算
            attention = calculate_attention(self.model, input, label, self.method, self.params, attention_fname)
            if not (self.images[i].shape[1:] == attention.shape):
                attention = cv2.resize(
                    attention[0], dsize=(self.images[i].shape[1], self.images[i].shape[2])
                )
            self.attentions.append(attention)

            # attentionをパッチにしてパッチごとの順位を計算
            self.patch_attentions.append(self.divide_attention_map_into_patch(self.attentions[i]))
            self.orders.append(self.calculate_attention_order(self.attentions[i], self.patch_attentions[i]))

            # mask_inputを作成して返す
            mask_input = self.create_mask_image(self.images[i].cpu().numpy(), self.attentions[i], self.patch_attentions[i], self.orders[i], "deletion", mask_mode=self.mask_mode)
            self.mask_inputs[i] = mask_input
        
        # (8, 1, 512, 512)
        return self.mask_inputs

    
    def divide_attention_map_into_patch(self, attention):
        """
        attentionをpatch単位に分割する
        """
        assert attention is not None
        # print("self.patch_size", self.patch_size) -> 1
        # print("aaaatenion : ", attention.shape) -> (512, 512)

        return skimage.measure.block_reduce(
            attention, (self.patch_size, self.patch_size), np.max
        )


    def calculate_attention_order(self, attention, patch_attention):
        """
        attentionの順番を計算する
        昇順のままでいいので、-消していることに注意
        """
        attention_flat = np.ravel(patch_attention)
        # 昇順でいいのでそのまま
        order = np.argsort(attention_flat)

        W, H = attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        return np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )


    def create_mask_image(self, image, attention, patch_attention, attention_order, mode: str, mask_mode: str="base"):
        # TODO ここってなにやってるん
        if not (image.shape[1:] == attention.shape):
            attention = cv2.resize(
                attention[0], dsize=(image.shape[1], image.shape[2])
            )
        
        C, W, H = image.shape # (1, 512, 512)
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        self.mask_img = np.zeros((num_insertion, C, W, H))
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(image.copy(), mean, std)

        # ここからした実験 : num_insertionの半分にしてる
        # print("self.orders : ", self.orders[i].shape) (2, 262144)
        step_index = min(self.step * (num_insertion//2 + 1), attention_order.shape[1] - 1)
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
            
            if mode == "insertion":
                mask_src = np.where(threthold <= patch_attention, 1.0, 0.0)
                mask_base = np.where(threthold <= patch_attention, 0.0, 1.0)
            elif mode == "deletion":
                mask_src = np.where(threthold <= patch_attention, 0.0, 1.0)
                mask_base = np.where(threthold <= patch_attention, 1.0, 0.0)
            
            mask_src = cv2.resize(mask_src, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            mask_base = cv2.resize(mask_base, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                mask_img[c] = src_image[c] * mask_src +  base_mask_image[c] * mask_base
                mask_img[c] = (mask_img[c] - mean[c]) / std[c]
            
        else:
            if mode == "insertion":
                mask = np.where(threthold <= patch_attention, 1.0, 0.0)
            elif mode == "deletion":
                mask = np.where(threthold <= patch_attention, 0.0, 1.0)

            mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(C):
                if mask_mode == "black":
                    mask_img[c] = (image[c] * mask - mean[c]) / std[c]
                elif mask_mode == "mean":
                    mask_img[c] = mask * ((image[c] - mean[c]) / std[c])


        # for i in range(num_insertion//2):
        #     # print("self.orders : ", self.orders[i].shape) (2, 262144)
        #     step_index = min(self.step * (i + 1), attention_order.shape[1] - 1)
        #     w_indices = attention_order[0, step_index]
        #     h_indices = attention_order[1, step_index]
        #     threthold = patch_attention[w_indices, h_indices]

        #     if mask_mode in ["blur", "base"]:
        #         src_image = image # 元画像(3, 224, 224) -> RGB

        #         if mask_mode == "blur":
        #             base_mask_image = cv2.blur(image.transpose(1,2,0), (self.patch_size, self.patch_size)) # (224, 224, 3)
        #         elif mask_mode == "base":
        #             base_mask_image = Image.open("./datasets/magnetogram/bias_image.png").resize((H, W)) # ここまではOK
        #             base_mask_image = np.asarray(base_mask_image, dtype=np.float32) / 255.0 # ここまではOK (512, 512)

        #         if len(base_mask_image.shape) == 3:
        #             print("transform base_mask_image")
        #             base_mask_image = base_mask_image.transpose(2, 0, 1)
        #         if len(base_mask_image.shape) == 2:
        #             base_mask_image = base_mask_image.reshape(C, H, W)
                
        #         if mode == "insertion":
        #             mask_src = np.where(threthold <= patch_attention, 1.0, 0.0)
        #             mask_base = np.where(threthold <= patch_attention, 0.0, 1.0)
        #         elif mode == "deletion":
        #             mask_src = np.where(threthold <= patch_attention, 0.0, 1.0)
        #             mask_base = np.where(threthold <= patch_attention, 1.0, 0.0)
                
        #         mask_src = cv2.resize(mask_src, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        #         mask_base = cv2.resize(mask_base, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

        #         for c in range(C):
        #             self.input[i, c] = src_image[c] * mask_src +  base_mask_image[c] * mask_base
        #             self.input[i, c] = (self.input[i, c] - mean[c]) / std[c]
                
        #     else:
        #         if mode == "insertion":
        #             mask = np.where(threthold <= patch_attention, 1.0, 0.0)
        #         elif mode == "deletion":
        #             mask = np.where(threthold <= patch_attention, 0.0, 1.0)

        #         mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

        #         for c in range(C):
        #             if mask_mode == "black":
        #                 self.input[i, c] = (image[c] * mask - mean[c]) / std[c]
        #             elif mask_mode == "mean":
        #                 self.input[i, c] = mask * ((image[c] - mean[c]) / std[c])

        # mask 画像の保存
        # print("mask_image.shape : ", mask_img.shape) #(1, 512, 512)
        img = mask_img.reshape(512, 512)
        img = image.reshape(512, 512)
        fig, ax = plt.subplots()
        if img.shape[-1] == 1:
            im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        else:
            im = ax.imshow(img, vmin=0, vmax=1, cmap="gray")
        fig.colorbar(im)
        plt.savefig(f"mask_image/{mode}/self.input.png")
        plt.clf()
        plt.close()

        # print("sssss", self.input[-1, ...].shape) # -> (1, 512, 512)
        return mask_img[-1]


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
    image: torch.Tensor,
    label: torch.Tensor,
    method: str,
    rise_params: Optional[Dict[str, Any]],
    fname: Optional[str],
) -> Tuple[np.ndarray, bool]:
    if method == "ABN":
        assert isinstance(model, AttentionBranchModel)
        # print("AAAAAAAAAAAAAA : ", image.shape) # -> (1, 1, 512, 512)
        y = model(image) # image : (1, 1, 512, 512)
        attentions = model.attention_branch.attention  # (1, W, W)
        attention = attentions[0]
        attention: np.ndarray = attention.cpu().detach().numpy()
        # ここにもともとはdetachないのに今回加えなければならないのはどゆこと
    elif method == "RISE":
        assert rise_params is not None
        rise_model = RISE(
            model,
            n_masks=rise_params["n_masks"],
            p1=rise_params["p1"],
            input_size=rise_params["input_size"],
            initial_mask_size=rise_params["initial_mask_size"],
            n_batch=rise_params["n_batch"],
            mask_path=rise_params["mask_path"],
        )
        attentions = rise_model(image)  # (N_class, W, H)
        attention = attentions[label]
        attention: np.ndarray = attention.cpu().numpy()
    elif method == "npy":
        assert fname is not None
        attention: np.ndarray = np.load(fname)
    elif method == "lambda":
        _ = model(image)
        num_bt = 0
        for module in module_generator(model):
            if isinstance(module, Bottleneck):
                num_bt += 1
                if 2 == num_bt:
                    attention = module.yc
                    attention = (attention - attention.min()) / (
                        attention.max() - attention.min()
                    )
                    break

        attention = attention.cpu().numpy()
    else:
        raise ValueError
    
    # (1, 128, 128)
    return attention