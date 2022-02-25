import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.datasets import VOCDetection


def convert_voc_label(voc_json: Dict):
    """
    VOCのdetectionラベルから画像内のクラス一覧を取得

    Args:
        voc_json(Dict): vocdetectionのラベル

    Returns:
        画像内のオブジェクトクラス名一覧
    """
    annotation = voc_json["annotation"]
    objects = annotation["object"]
    if not isinstance(objects, list):
        objects = [objects]

    classes = [object["name"] for object in objects]

    return set(classes)


class VOCClassification(VOCDetection):
    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.classes = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )

        self.multi_label_binarizer = MultiLabelBinarizer()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        target = convert_voc_label(target)
        target = self.multi_label_binarizer.fit_transform(
            [self.classes, tuple(target)]
        )[1]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
