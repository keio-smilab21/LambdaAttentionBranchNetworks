from typing import Dict

import torch

from metrics.base import Metric


class FlareMetric(Metric):
    def __init__(self) -> None:
        self.total = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.total += labels.size(0)
        preds = torch.max(preds, dim=-1)[1]
        tp, fp, tn, fn = confusion(preds, labels)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def score(self) -> Dict[str, float]:
        return {
            "Acc": self.acc(),
            "TSS": self.tss(),
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
        }

    def acc(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def tss(self) -> float:
        return self.tp / (self.tp + self.fn) - self.fp / (self.fp + self.tn)

    def clear(self) -> None:
        self.total = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0


def confusion(prediction, truth):
    """Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float("inf")).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
