from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.utils.data as data
import torchvision.transforms as transforms
from metrics.accuracy import Accuracy
from metrics.base import Metric
from torch import nn

from data.IDRID import IDRiDDataset

ALL_DATASETS = ["IDRiD"]


def setting_dataset(
    dataset_name: str,
    batch_size: int,
    image_size: int = 224,
    is_test: bool = False,
    train_ratio: float = 0.9,
    dataset_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Union[data.DataLoader, Dict[str, data.DataLoader]],
    Metric,
    nn.modules.loss._Loss,
]:
    """
    データセット名からデータローダー・評価指標・ロスの作成

    Args:
        dataset_name(str) : データセット名
        batch_size  (int) : バッチサイズ
        image_size  (int) : 画像サイズ
        is_test     (bool): テストデータセットのみ作成
        train_ratio(float): valがないとき，train / valの分割割合

    Returns:
        dataloader  : データローダーのdictまたはテストデータローダー
        (Union[data.Dataloader, dict[str, data.Dataloader]])
        metrics     : 評価指標
        criterion   : ロス関数
    """
    train_dataset = create_dataset(
        dataset_name,
        "train",
        image_size,
    )
    test_dataset = create_dataset(dataset_name, "test", image_size)

    if dataset_name == "IDRiD":
        metrics = Accuracy()
        assert dataset_params is not None
        # criterion = nn.CrossEntropyLoss(weight=dataset_params["loss_weights"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=dataset_params["loss_weights"])
        # lenが定義されてるとは限らないがIDRiDでは定義済みのため無視
        # No `def __len__(self)` default? (data.Datasetより引用)
        # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        train_size = int(train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_dataset.transform = test_dataset.transform
    else:
        raise ValueError()

    test_dataloader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    if is_test:
        return test_dataloader, metrics, criterion

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "Train": train_dataloader,
        "Val": val_dataloader,
        "Test": test_dataloader,
    }

    return dataloader_dict, metrics, criterion


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

    params = get_dataset_params(dataset_name)

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

    if dataset_name == "IDRiD":
        dataset = IDRiDDataset(
            root="./datasets", image_set=image_set, transform=transform
        )
    else:
        raise ValueError()

    return dataset


def get_dataset_params(dataset_name: str) -> Dict[str, Any]:
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
    if dataset_name == "IDRiD":
        params["mean"] = (0.4329, 0.2094, 0.0687)
        params["std"] = (0.3083, 0.1643, 0.0829)
        params["classes"] = ("normal", "abnormal")

    return params
