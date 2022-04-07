import numpy as np
import torch
import matplotlib.pyplot as plt

from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from utils.visualize import save_image


def create_bias_image(image_shape: int = (1, 512, 512)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        "magnetogram",
        8,
        512,
        train_ratio = 0.8
    )
    data_param = get_parameter_depend_in_data_set(
        "magnetogram", pos_weight=torch.Tensor([1.0]).to(device)
    )
    mean = data_param["mean"]
    std = data_param["std"]

    train_dataloader = dataloader_dict["Train"] # 5691
    C, W, H = image_shape
    bias_image = torch.zeros((len(train_dataloader), C, W, H)) # (5691, 1, 512, 512)
    
    iter_num = 0
    for data, _ in train_dataloader:
        iter_num += 1
        for i in range(8):
            bias_image[i, ...] += data[i, ...] # (1, 512, 512)
        bias_image[i, ...] /= 8.0
    print("iter_num", iter_num)
    print("Final bias_image.shape", bias_image.shape)

    image = torch.zeros((1, 512, 512))
    print("image: ", image.shape)
    for i in range(iter_num):
        image += bias_image[i, ...]
    image /= iter_num
    print("image : ", image.shape)

    save_image(image, "temp/bisa_image.png", mean, std)

    plt.imshow(image, cmap="gray")
    plt.savefig("temp/bisa_image_raw.png")
