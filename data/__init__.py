from typing import Any, Callable, Dict, Optional

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from metrics.accuracy import Accuracy
from torch import nn

from data.IDRID import IDRiDDataset

ALL_DATASETS = ["IDRiD"]


def create_dataloader_dict(
    dataset_name: str,
    batch_size: int,
    image_size: int = 224,
    only_test: bool = False,
    train_ratio: float = 0.9,
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

    train_dataset = create_dataset(
        dataset_name,
        "train",
        image_size,
    )

    dataset_params = get_parameter_depend_in_data_set(dataset_name)

    # valの作成 or 分割
    if dataset_params["has_val"]:
        val_dataset = create_dataset(
            dataset_name,
            "val",
            image_size,
        )
    else:
        # lenが()定義されてるとは限らない
        # No `def __len__(self)` default? (data.Datasetより引用)
        # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        train_size = int(train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_dataset.transform = test_dataset.transform

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


def create_dataset(
    dataset_name: str,
    image_set: str = "train",
    image_size: int = 224,
    transform: Optional[Callable] = None,
) -> data.Dataset:
    """
    データセットの作成
    正規化パラメータなどはデータセットごとに作成

    Args:
        dataset_name(str)  : データセット名
        image_set(str)     : train / val / testから選択
        image_size(int)    : 画像サイズ
        transform(Callable): transform

    Returns:
        data.Dataset : pytorchデータセット
    """
    assert dataset_name in ALL_DATASETS
    params = get_parameter_depend_in_data_set(dataset_name)

    if transform is None:
        if image_set == "train":
            transform = transforms.Compose(
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
                    transforms.RandomErasing(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(params["mean"], params["std"]),
                ]
            )

    dataset = params["dataset"](
        root="./datasets", image_set=image_set, transform=transform, params=params
    )

    return dataset


def get_parameter_depend_in_data_set(
    dataset_name: str,
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
    # ImageNet
    params["mean"] = (0.485, 0.456, 0.406)
    params["std"] = (0.229, 0.224, 0.225)
    params["root"] = dataset_root

    if dataset_name == "IDRiD":
        params["name"] = "IDRiD"
        params["dataset"] = IDRiDDataset
        params["mean"] = (0.4329, 0.2094, 0.0687)
        params["std"] = (0.3083, 0.1643, 0.0829)
        params["classes"] = ("normal", "abnormal")
        params["has_val"] = False

        params["metric"] = Accuracy()
        params["criterion"] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return params
