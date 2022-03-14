from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.utils import reverse_normalize, tensor_to_numpy


def save_attention_map(attention: Union[np.ndarray, torch.Tensor], fname: str) -> None:
    attention = tensor_to_numpy(attention)

    fig, ax = plt.subplots()

    min_att = attention.min()
    max_att = attention.max()

    im = ax.imshow(
        attention, interpolation="nearest", cmap="jet", vmin=min_att, vmax=max_att
    )
    fig.colorbar(im)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def save_image_with_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    fname: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    if len(attention.shape) == 3:
        attention = attention[0]

    # image : (C, W, H)
    attention = cv2.resize(attention, dsize=(image.shape[1], image.shape[2]))
    image = reverse_normalize(image.copy(), mean, std)
    image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots()
    if image.shape[2] == 1:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(image, vmin=0, vmax=1)

    im = ax.imshow(attention, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def save_image(
    image: np.ndarray,
    fname: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:

    # image : (C, W, H)
    image = reverse_normalize(image.copy(), mean, std)
    image = image.clip(0, 1)
    image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots()
    if image.shape[0] == 1:
        im = ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        im = ax.imshow(image, vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def save_data_as_plot(
    data: np.ndarray,
    fname: str,
    x: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    xlim: Optional[Union[int, float]] = None,
) -> None:
    """
    dataをプロットしてグラフに保存

    Args:
        data(ndarray): 保存するデータ
        fname(str)   : 保存ファイル名
        x(ndarray)   : 横軸
        label(str)   : 凡例のラベル
    """
    fig, ax = plt.subplots()

    if x is None:
        x = range(len(data))

    ax.plot(x, data, label=label)

    xmax = len(data) if xlim is None else xlim
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.05, 1.05)

    plt.legend()
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05)
    plt.clf()
    plt.close()
