from typing import Dict, List, Tuple

import torch

from metrics.base import Metric


def num_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[int]:
    """
    top-k Accuracyの計算

    Args:
        output(Tensor)  : モデルの出力
        target(Tensor)  : ラベル
        topk(Tuple[int]): 上位何番目までに入っていれば正解とするか

    Returns:
        List[int] : top-k Accuracy（len(topk)個）
    """
    maxk = max(topk)

    # [1, 2, 3, 4, 5].topk(3) -> (values=[5, 4, 3] indices=[4, 3, 2])
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    # topk: [4, 3, 2] target(expand) : [3, 3, 3] -> [F. T. F]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # [[False, False, True], [F, F, F], [T, F, F]]
    # -> [0, 0, 1, 0, 0, 0, 1, 0, 0] -> 2
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)
    return result


class Accuracy(Metric):
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.total += labels.size(0)
        self.correct += num_accuracy(preds, labels)[0]

        preds = torch.max(preds, dim=-1)[1]
        tp, fp, tn, fn = confusion(preds, labels)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def score(self) -> Dict[str, float]:
        return {
            "Acc": self.acc(),
            "TP": int(self.tp),
            "FP": int(self.fp),
            "FN": int(self.fn),
            "TN": int(self.tn),
        }

    def acc(self) -> float:
        return self.correct / self.total

    def clear(self) -> None:
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0


class MultiLabelAccuracy(Metric):
    def __init__(self) -> None:
        self.total_acc_counts = 0
        self.binary_acc_counts = 0
        self.total = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        preds = torch.sigmoid(preds)
        preds = 1 * (preds > 0.5)
        num_classes = labels.size(1)
        num_collects = preds.eq(labels).sum(1)
        num_collects = num_collects.to(torch.float64)

        all_collects = num_collects // num_classes
        ratio_collects = num_collects / num_classes

        self.total_acc_counts += all_collects.sum(0)
        self.binary_acc_counts += ratio_collects.sum(0)
        self.total += labels.size(0)

    def score(self) -> Dict[str, float]:
        scores = {"Total Acc": self.total_acc(), "Binary Acc": self.binary_acc()}
        return scores

    def binary_acc(self) -> float:
        return self.binary_acc_counts / self.total

    def total_acc(self) -> float:
        return self.total_acc_counts / self.total

    def clear(self) -> None:
        self.total_acc_counts = 0
        self.binary_acc_counts = 0
        self.total = 0


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
