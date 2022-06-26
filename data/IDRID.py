import csv
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class IDRiDDataset(Dataset):
    """
    Attributes:
        root(str): データセット一覧ディレクトリ root/IDRIDが存在する前提
        image_set(str) : データセットの種類 train / val /test
        transform(Callable) : 画像の変換
    """

    def __init__(
        self, root: str, image_set: str = "train", transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform

        # root_dir下にIDRiDデータセットを解凍したと仮定，デフォルトのデータセットパスをハードコーディング
        self.idrid_dir = os.path.join(self.root, "IDRID", "B. Disease Grading")

        image_dir = "a. Training Set"
        annotation_file = "a. IDRiD_Disease Grading_Training Labels.csv"
        if image_set == "test":
            image_dir = "b. Testing Set"
            annotation_file = "b. IDRiD_Disease Grading_Testing Labels.csv"

        self.image_dir = os.path.join(self.idrid_dir, "1. Original Images", image_dir)

        self.annotation_file = os.path.join(
            self.idrid_dir, "2. Groundtruths", annotation_file
        )

        with open(self.annotation_file) as f:
            reader = csv.reader(f)
            # skip header
            reader.__next__()
            annotations = [row for row in reader]

        # Transpose [Image_fname, Label1, Label2]
        annotations = [list(x) for x in zip(*annotations)]

        # Label2は使用しない
        self.images = annotations[0]
        self.images = list(
            map(lambda x: os.path.join(self.image_dir, x + ".jpg"), self.images)
        )
        # Grade 0 -> 0/ Grade 1, 2, 3 -> 1
        self.targets = list(map(lambda x: int(1 <= int(x)), annotations[1]))

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB") # 4288, 2848

        # 内側のみの場合 (710, 295, 3269, 2520)
        # im_name :  IDRiD_XXX.jpg or IDRiD_bias_image.jpg
        im_name = self.images[index].split("/")[-1]
        if im_name != "IDRiD_bias_image.jpg":
            image = image.crop((290, 0, 3701, 2849)) # 3411 2849
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
