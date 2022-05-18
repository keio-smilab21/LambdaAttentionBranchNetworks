import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from data import create_dataset
from utils.visualize import save_image
from utils.utils import reverse_normalize

def create_bias_image(image_shape: int = (1, 512, 512)):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        "magnetogram",
        1,
        512,
        train_ratio = 0.8
    )
    data_param = get_parameter_depend_in_data_set(
        "magnetogram", pos_weight=torch.Tensor([1.0])
    )
    mean = data_param["mean"]
    std = data_param["std"]

    train_dataloader = dataloader_dict["Train"] # 5691
    print(len(train_dataloader))
    C, W, H = image_shape
    batch_bias_image = np.zeros((len(train_dataloader), C, W, H)) # (5691, 1, 512, 512)
    
    iter_num = 0
    for data, _ in train_dataloader:
        # print("data : ", data.shape)
        data = data.numpy()
        iter_num += 1
        for i in range(8):
            # print("batch_image", batch_bias_image[i,...].shape)
            # print("data : ", data[i, ...].shape)

            batch_bias_image[i, ...] = batch_bias_image[i, ...] + data[i, ...] # (1, 512, 512) : float32
    # print("iter_num", iter_num)
    # print("Final bias_image.shape", batch_bias_image.shape)

    bias_image = np.zeros((1, 512, 512))
    # print("image: ", bias_image.shape)
    for i in range(iter_num):
        bias_image = bias_image + batch_bias_image[i, ...]
    bias_image /= len(train_dataloader) # float32
    # print("image : ", bias_image.shape)

    save_image(bias_image, "temp/bisa_image.png", mean, std)

    plt.imshow(bias_image.reshape(W, H), cmap="gray")
    plt.savefig("temp/bisa_image_raw.png")

def create_bias_image2(image_shape: int = (1, 512, 512), dataname: str = "magnetogram"):
    dataset = create_dataset(
        dataname,
        "train",
        512
    )
    data_param = get_parameter_depend_in_data_set(
        dataname, pos_weight=torch.Tensor([1.0])
    )
    mean = data_param["mean"]
    std = data_param["std"]

    data_num = len(dataset)
    C, W, H = image_shape

    images = dataset.images.astype(np.float32)
    print(images.shape)

    bias_image = np.mean(images, axis=0).astype(np.uint8)
    print("bias_image", bias_image.shape)
    
    # save_image(bias_image[np.newaxis], "temp/bias_image.png", (0,), (1,))
    cv2.imwrite(f"{dataname}_bias_image.png", bias_image)