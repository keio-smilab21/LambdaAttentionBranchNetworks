import math
import os
from typing import Dict, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_parameter_depend_in_data_set
from utils.utils import reverse_normalize
from utils.visualize import save_data_as_plot, save_image

from metrics.base import Metric
import skimage.measure


class ADCC(Metric):
    def __init__(self, model: nn.Module) -> None:
        self.total = 0
        self.total_coherency = 0
        self.total_avg_drop = 0
        self.total_complexity = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        return super().evaluate(preds, labels)

    def score(self) -> Dict[str, float]:
        return super().score()

    def clear(self) -> None:
        return super().clear()
