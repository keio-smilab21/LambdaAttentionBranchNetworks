import argparse
from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data
from data import ALL_DATASETS, create_dataset
from torchvision import transforms
from tqdm import tqdm
from utils.utils import fix_seed
import cv2
from PIL import Image
import glob


def make_id_bias_image(
    dataloader: data.DataLoader, image_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    画像データセットの平均・分散を計算

    Args:
        dataloader(DataLoader): データローダー
        image_size(int)  : 画像サイズ

    Returns:
        各チャンネルの平均・分散
        Tuple[torch.Tensor, torch.Tensor]
    """

    total = 0

    bias_image = torch.zeros((3, 224, 224))

    # 画素値と二乗を加算
    for inputs, _ in tqdm(dataloader):
        inputs.to(device)
        bias_image += inputs[0]
        total += inputs.size(0)

    print("bias_image1", bias_image.shape)
    print("total", total)

    bias_image /= total

    return bias_image

def make_bias():
    # PIL.Image で画像を読み込む
    train_dir = "./datasets/IDRID/B. Disease Grading/1. Original Images/a. Training Set/"
    # print("dir_path ; ", train_dir)

    image_list = []
    for im in glob.glob(train_dir+"IDRiD*"):
        image_list.append(im)
    print(len(image_list)) # 413
    
    image = np.ones((413, 2848, 4288, 3))
    print("image[0] : ", image[0].shape)

    # numpy配列に変換する
    for idx, name in tqdm(enumerate(image_list)):
        image[idx] = np.asarray(Image.open(name))
        print("image dtype :", image[idx].dtype)
        break
    print("image ; ", image.shape)

    # 平均をとる
    im_mean = np.mean(image, axis=0)
    # Image(jpg)に変換する
    # 保存する
    pass

def resize_bias():
    bias_image = Image.open("datasets/IDRID/IDRiD_bias_image.jpg")
    print("bf : ", bias_image.size)
    bias_image = bias_image.resize((4288, 2848), Image.LANCZOS)
    print("af : ", bias_image.size)
    bias_image.save("resize_bias_image.jpg")

def main():
    fix_seed(args.seed, True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = create_dataset(args.dataset, "train", args.image_size, transform)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    bias_image = make_id_bias_image(dataloader, args.image_size)
    # bias_image = bias_image.transpose((1,2,0))
    print("bias_image2: ", bias_image.shape)

    bias_image = bias_image.cpu().detach().numpy().copy()
    bias_image = np.transpose(bias_image, (1,2,0))
    bias_image = (bias_image * 255).astype(np.uint8)
    print("bias_image3: ", bias_image.shape)
    print("bias_image_dtype ;", bias_image.dtype)

    # cv2.imwrite(f"IDRiD_bias_image.png", bias_image)
    Image.fromarray(bias_image).save("IDRiD_bias_image.jpg")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    # main()
    # make_bias()
    resize_bias()
