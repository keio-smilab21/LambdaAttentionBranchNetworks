import csv
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Magnetogram(Dataset):
    """
    Attributes:
        root(str): データセット一覧ディレクトリ root/magnetogramが存在する前提
        image_set(str) : データセットの種類 train / val /test
        transform(Callable) : 画像の変換
    """

    def __init__(
        self,
        root: str,
        image_set: str,
        params: Dict[str, Any],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        years = params["years"][image_set]

        self.data_dir = os.path.join(self.root, "magnetogram")
        image_list = []
        self.targets = []

        for year in years:
            image_file = os.path.join(self.data_dir, f"{year}.npy")
            annotation_file = os.path.join(self.data_dir, f"{year}.csv")

            with open(annotation_file) as f:
                reader = csv.reader(f)
                # skip header
                reader.__next__()
                annotations = [row for row in reader]

            # Transpose [Image_fname, Label]
            annotations = [list(x) for x in zip(*annotations)]

            image_list.append(np.load(image_file))
            # flag 0, 1 -> 0/ flag 2, 3 -> 1
            self.targets += list(map(lambda x: int(2 <= int(x)), annotations[1]))

        if image_set == "train":
            image_list.append(np.zeros((1, 512, 512), dtype=np.uint8))
            self.targets += [0]

        self.images = np.concatenate(image_list)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.fromarray(self.images[index])

        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
