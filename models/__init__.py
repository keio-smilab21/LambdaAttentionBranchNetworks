import os
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from models.attention_branch import add_attention_branch
from models.lambda_resnet import lambda_resnet50

ALL_MODELS = ["lambda_resnet", "resnet", "resnet50"]


def change_num_classes(
    model: nn.Module, num_classes: int, add_flatten: bool = False
) -> nn.Module:
    """
    最終FC層の出力次元をnum_classesに変更

    Args:
        model(nn.Module) : 変更対象のモデル
        num_classes(int) : 変更後のクラス数(=最終層のユニット数)
        add_flatten(bool): 最終層のLinearの前にflattenするか
                           (CNNの場合などに行う / Note参照)

    Returns:
        nn.Module: 最終層の次元をnum_classesにしたmodel

    Note:
        最終層がnn.Linear以外か，out_features == num_classes
        ならばmodelをそのままreturn

        modelがnn.flattenではなくtorch.flattenを使用している場合
        model.children()をするとflattenが消えてしまうためadd_flattenが必要
        nn.flattenを使用している場合はadd_flattenは不要
    """
    modules = list(model.children())
    final_layer = modules[-1]

    if not isinstance(final_layer, nn.Linear):
        return model

    out_dim = final_layer.out_features
    if out_dim == num_classes:
        return model

    in_dim = final_layer.in_features
    final_layer = nn.Linear(in_dim, num_classes)

    if add_flatten:
        return nn.Sequential(*modules[:-1], nn.Flatten(1), final_layer)
    return nn.Sequential(*modules[:-1], final_layer)


def create_model(
    base_model: str,
    num_classes: int = 1000,
    base_pretrained: Optional[str] = None,
    base_pretrained2: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    attention_branch: bool = False,
    division_layer: Optional[str] = None,
    multi_task: bool = False,
    num_tasks: Optional[List[int]] = None,
    theta_attention: float = 0,
) -> Optional[nn.Module]:
    """
    モデル名・パラメータからモデルの作成

    Args:
        base_model(str)       : ベースとするモデル名（resnetなど）
        num_classes(int)      : クラス数
        base_pretrained(str)  : ベースモデルのpretrain path
        base_pretrained2(str) : 最終層変更後のpretrain path
        pretrained_path(str)  : 最終的なモデルのpretrain path
                                （アテンションブランチ化などをした後のモデルのもの）
        attention_branch(bool): アテンションブランチ化するか
        division_layer(str)   : いくつめのlayerで分割してattentionbranchを導入するか
        multi_task(bool)      : マルチタスク化するか
        num_tasks(List[int])  : 各タスクのクラス数，タスク数はlen(num_tasks)
                                全部別にしたい場合[1, 1, ...] (Note参照)
        theta_attention(float): Attention Branch入力時の閾値
                                この値よりattentionが低いピクセルを0にして入力

    Returns:
        nn.Module: 作成したモデル

    Note:
        マルチタスクはマルチラベル（ex: 画像中に複数物体が含まれる）なnクラス分類問題に使用
        ABNで各クラスごとにattentionを得るためにはマルチタスク化 = 複数のPerception Branchが必要

        全クラスごとにPerception Branchを用意すると重くなるため，あるクラスのみ分けたい際にnum_tasksを使用
        例えば20クラスのうち10番目のみ分けたい場合 num_tasks=[9, 1, 10]とすると
        Perception Branchがlen(num_tasks) = 3個でき，クラス数1のPerceptionは10番目に特化している
        10番目にのみ特化したattentionを得ることが可能

        そのため，multi_taskはattention_branch = Trueの際に使用されることを想定
    """
    if base_model == "lambda_resnet":
        model = lambda_resnet50()

        layer_index = {"layer1": -6, "layer2": -5, "layer3": -4}
        # コードを直接変更してnn.flattenにしているためFalse
        add_flatten = False
    elif base_model == "resnet":
        # base_pretrainedに何か書いてあればpretrained有, pathが存在するならば後に上書き
        model = resnet18(pretrained=(base_pretrained is not None))

        layer_index = {"layer1": -6, "layer2": -5, "layer3": -4}
        add_flatten = True
    elif base_model == "resnet50":
        model = resnet50(pretrained=(base_pretrained is not None))
        layer_index = {"layer1": -6, "layer2": -5, "layer3": -4}
        add_flatten = True
    else:
        return None

    # base_pretrainedがpathならば読み込み
    if base_pretrained is not None and os.path.isfile(base_pretrained):
        model.load_state_dict(torch.load(base_pretrained))
        print(f"base pretrained {base_pretrained} loaded.")

    if multi_task:
        assert attention_branch, "multi_taskはattention_branch = Trueを想定"
        # multi_taskのときはLinear: R^d -> 1をタスク数分使うため最終層の出力次元は1
        model = change_num_classes(model, 1, add_flatten)
    else:
        model = change_num_classes(model, num_classes, add_flatten)

    # base_pretrained2がpathならば読み込み
    if base_pretrained2 is not None and os.path.isfile(base_pretrained2):
        model.load_state_dict(torch.load(base_pretrained2))
        print(f"base pretrained2 {base_pretrained2} loaded.")

    if attention_branch:
        assert division_layer is not None
        model = add_attention_branch(
            model,
            layer_index[division_layer],
            num_classes,
            multi_task,
            num_tasks,
            theta_attention,
        )

    # pretrainedがpathならば読み込み
    if pretrained_path is not None and os.path.isfile(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"pretrained {pretrained_path} loaded.")

    return model
