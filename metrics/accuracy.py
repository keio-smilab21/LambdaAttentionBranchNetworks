from typing import Dict, List, Tuple

import torch

from metrics.base import Metric


def num_correct_topk(
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

    # 上位maxk個の予測クラスを取得
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    # pred = [1, 2, 3] を label(expand) [3, 3, 3]と比較 (ラベルをmaxk個並べた配列)
    # 上記の例では correct = [False, False, True] でmaxk番目までにTrueがあるかどうかがわかる
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
        self.correct += num_correct_topk(preds, labels)[0]

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
