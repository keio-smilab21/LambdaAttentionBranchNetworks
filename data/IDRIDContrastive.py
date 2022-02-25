import csv
import os
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_branch import AttentionBranchModel
from numpy import math
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class IDRiDContrastiveDataset(Dataset):
    """
    Attributes:
        root(str): データセット一覧ディレクトリ root/IDRIDが存在する前提
        image_set(str) : データセットの種類 train / val /test
        transform(Callable) : 画像の変換
    """

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        theta_attention: float = 0,
        model: Optional[nn.Module] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.theta_attention = theta_attention
        self.model = model

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

        self.num_original_images = len(self.images)

        self.contrastive = torch.zeros((0, 3, 224, 224))
        self.contrastive_targets = []

    def contrastive_augment(
        self, model: AttentionBranchModel, batch_size: int, device: torch.device
    ):
        batch_size = batch_size // 10
        num_iter = math.ceil(self.num_original_images / batch_size)

        concat_list = []
        for iter in tqdm(range(num_iter), desc="Augmentating"):
            start = iter * batch_size
            end = min((1 + iter) * batch_size, self.num_original_images)

            targets = [self[i][1] for i in range(start, end)]
            images = [self[i][0] for i in range(start, end)]
            inputs = torch.stack(images).to(device)
            _ = model(inputs)
            inputs = inputs.to("cpu")

            attentions = model.attention_branch.attention.cpu()
            attentions = F.interpolate(attentions, inputs.size()[2:])
            attention_pos = torch.where(
                self.theta_attention < attentions, attentions.double(), 0.0
            )
            attention_neg = torch.where(
                self.theta_attention < attentions, 0.0, attentions.double()
            )

            concat_list += [
                (inputs * attention_pos).to("cpu"),
                (inputs * attention_neg).to("cpu"),
            ]

            negative_targets = [-(x - 1) for x in targets]
            contrastive_targets = targets + negative_targets
            self.contrastive_targets += contrastive_targets

        self.contrastive = torch.cat(concat_list)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        if self.num_original_images <= index:
            index -= self.num_original_images
            return self.contrastive[index], self.contrastive_targets[index]

        image = Image.open(self.images[index]).convert("RGB")

        # 内側のみの場合 (710, 295, 326inputs.copy()9, 2520)
        image = image.crop((290, 0, 3701, 2849))
        target = self.targets[index]

        if self.transform is not None:
            image: torch.Tensor = self.transform(image)

        if target == 0 or torch.rand(1).item() < 0.8:
            return image, target

        if self.model is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _ = self.model(image.unsqueeze(0).to(device))
            attentions = self.model.attention_branch.attention.cpu()
            attentions = F.interpolate(attentions, image.size()[1:])

            attention_pos = torch.where(
                self.theta_attention < attentions, attentions.double(), 0.0
            )
            attention_neg = torch.where(
                self.theta_attention < attentions, 0.0, attentions.double()
            )

            random_pn = torch.rand(1).item() < 0.5

            image = image * [attention_pos, attention_neg][random_pn]
            image = image.float()[0]
            if random_pn:
                target = -(target - 1)  # 0 -> 1, 1 -> 0

        return image, target

    def __len__(self) -> int:
        return len(self.images)
