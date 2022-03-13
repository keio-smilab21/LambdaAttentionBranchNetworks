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


def confusion(output, target) -> Tuple[int, int, int, int]:
    """
    Confusion Matrixの計算

    Args:
        output(Tensor)  : モデルの出力
        target(Tensor)  : ラベル

    Returns:
        true_positive(int) : TPの個数
        false_positive(int): FPの個数
        true_negative(int) : TNの個数
        false_negative(int): FNの個数
    """

    # TP: 1/1 = 1, FP: 1/0 -> inf, TN: 0/0 -> nan, FN: 0/1 -> 0
    confusion_vector = output / target

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float("inf")).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return (
        int(true_positives),
        int(false_positives),
        int(true_negatives),
        int(false_negatives),
    )
