import csv
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Aia131(Dataset):
    """
    Attributes:
        root(str): データセット一覧ディレクトリ root/magnetogramが存在する前提
        image_set(str) : データセットの種類 train / val /test
        transform(Callable) : 画像の変換
    """

    def __init__(
        self,
        root: str,
        image_set: str, # train/val/test
        params: Dict[str, Any],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        years = params["years"][image_set]

        self.data_dir = os.path.join(self.root, "aia131")
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
            annotations = [list(x) for x in zip(*annotations)] # (image_name, label, year) * 7990

            images = np.load(image_file) # (B, H, W)
            image_list.append(images)
            # flag 0, 1 -> 0/ flag 2, 3 -> 1
            self.targets += list(map(lambda x: int(2 <= int(x)), annotations[1])) # 45530

        self.images = np.concatenate(image_list) # (7990, 512, 512)

        # add bias_image only when train
        if image_set == "train":
            self.targets += [0]
            bias_image = Image.open("./datasets/aia131/aia131_bias_image.png")
            bias_image = np.asarray(bias_image, dtype=np.uint8)
            bias_image = bias_image[np.newaxis]
            self.images = np.concatenate([self.images, bias_image])


    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.fromarray(self.images[index])

        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.images)