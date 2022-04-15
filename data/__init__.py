import random
from turtle import pos
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from metrics.accuracy import Accuracy
from metrics.flare import FlareMetric
from torch import nn
from torch.utils.data import Dataset

from data.IDRID import IDRiDDataset
from data.magnetogram import Magnetogram
from data.sampler import BalancedBatchSampler
from losses.losses import DoubleBCE, BCEWithKL, MaskKL

ALL_DATASETS = ["IDRiD", "magnetogram"]


class SubsetWithTransform(Dataset):
    def __init__(
        self, dataset: Dataset, indices: List[int], transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.indices)


def create_dataloader_dict(
    dataset_name: str,
    batch_size: int,
    image_size: int = 224,
    only_test: bool = False,
    train_ratio: float = 0.9,
    is_transform: bool = True
) -> Dict[str, data.DataLoader]:
    """
    データローダーの作成

    Args:
        dataset_name(str) : データセット名
        batch_size  (int) : バッチサイズ
        image_size  (int) : 画像サイズ
        only_test   (bool): テストデータセットのみ作成
        train_ratio(float): valがないとき, train / valの分割割合

    Returns:
        dataloader_dict : データローダーのdict
    """

    test_dataset = create_dataset(dataset_name, "test", image_size)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    if only_test:
        return {"Test": test_dataloader}

    dataset_params = get_parameter_depend_in_data_set(dataset_name)

    # valの作成 or 分割
    if dataset_params["has_val"]:
        train_dataset = create_dataset(
            dataset_name,
            "train",
            image_size,
        )
        val_dataset = create_dataset(
            dataset_name,
            "val",
            image_size,
        )
    else:
        train_dataset, val_dataset = create_train_val_dataset(
            dataset_name, train_ratio, image_size, is_transform
        )

    if dataset_params["sampler"]:
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_sampler=BalancedBatchSampler(train_dataset, 2, batch_size // 2),
        )
    else:
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "Train": train_dataloader,
        "Val": val_dataloader,
        "Test": test_dataloader,
    }

    return dataloader_dict


def get_parameter_depend_in_data_set(
    dataset_name: str,
    loss_type: str="singleBCE",
    pos_weight: torch.Tensor = torch.Tensor([1]),
    dataset_root: str = "./datasets",
) -> Dict[str, Any]:
    """
    データセットのパラメータを取得

    Args:
        dataset_name(str): データセット名

    Returns:
        dict[str, Any]: 平均・分散・クラス名などのパラメータ

    Note:
        クラス変数としてパラメータをもたせるとクラスを作成する必要があるため関数を作成した
    """
    params = dict()
    params["name"] = dataset_name
    params["root"] = dataset_root
    params["has_params"] = False
    params["num_channel"] = 3
    params["sampler"] = True
    # ImageNet
    params["mean"] = (0.485, 0.456, 0.406)
    params["std"] = (0.229, 0.224, 0.225)

    if dataset_name == "IDRiD":
        params["dataset"] = IDRiDDataset
        params["mean"] = (0.4329, 0.2094, 0.0687)
        params["std"] = (0.3083, 0.1643, 0.0829)
        params["classes"] = ("normal", "abnormal")
        params["has_val"] = False

        params["metric"] = Accuracy()
    elif dataset_name == "magnetogram":
        params["dataset"] = Magnetogram
        params["num_channel"] = 1
        params["mean"] = (0.3625,)
        params["std"] = (0.2234,)
        params["classes"] = ("OC", "MX")
        params["has_val"] = True
        params["has_params"] = True
        params["sampler"] = True
        params["years"] = {
            "train": ["2010", "2011", "2012", "2013", "2014", "2015"],
            "val": ["2016"],
            "test": ["2017"],
        }

        params["metric"] = FlareMetric()
    
    if loss_type == "SingleBCE":
        params["criterion"] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "DoubleBCE":
        params["criterion"] = DoubleBCE(pos_weight=pos_weight)
    elif loss_type == "BCEWithKL":
        params["criterion"] = BCEWithKL(pos_weight=pos_weight)
    elif loss_type == "MaskKL":
        params["criterion"] = MaskKL(pos_weight=pos_weight)

    return params


def create_transform(
    image_set: str,
    image_size: int,
    params: Dict[str, Any],
    p_random_erasing: float = 0.5,
    is_transform: bool = False
):
    if image_set == "train" and is_transform:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=5),
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.7, 1.3), ratio=(3 / 4, 4 / 3)
                ),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(params["mean"], params["std"]),
                # transforms.RandomErasing(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(params["mean"], params["std"]),
        ]
    )


def create_train_val_dataset(
    dataset_name: str,
    train_ratio: float,
    image_size: int = 224,
    is_transform: bool = True
):
    params = get_parameter_depend_in_data_set(dataset_name)
    train_transform = create_transform("train", image_size, params, is_transform)
    test_transform = create_transform("test", image_size, params)

    trainval_dataset = params["dataset"](
        root="./datasets",
        image_set="train",
        transform=None,
    )

    indices = range(len(trainval_dataset) - 1)
    train_size = int(train_ratio * len(trainval_dataset)) + 1
    val_size = len(trainval_dataset) - train_size

    train_indices = random.sample(indices, train_size) + [len(trainval_dataset) - 1]
    val_indices = random.sample(indices, val_size)

    train_dataset = SubsetWithTransform(
        trainval_dataset, train_indices, train_transform
    )
    val_dataset = SubsetWithTransform(trainval_dataset, val_indices, test_transform)

    return train_dataset, val_dataset


def create_dataset(
    dataset_name: str,
    image_set: str = "train",
    image_size: int = 224,
    transform: Optional[Callable] = None,
) -> Dataset:
    """
    データセットの作成
    正規化パラメータなどはデータセットごとに作成

    Args:
        dataset_name(str)  : データセット名
        image_set(str)     : train / val / testから選択
        image_size(int)    : 画像サイズ
        transform(Callable): transform

    Returns:
        Dataset : pytorchデータセット
    """
    assert dataset_name in ALL_DATASETS
    params = get_parameter_depend_in_data_set(dataset_name)

    if transform is None:
        transform = create_transform(image_set, image_size, params)

    if params["has_params"]:
        dataset = params["dataset"](
            root="./datasets",
            image_set=image_set,
            params=params,
            transform=transform,
        )
    else:
        dataset = params["dataset"](
            root="./datasets", image_set=image_set, transform=transform
        )

    return dataset
