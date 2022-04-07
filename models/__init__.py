import os
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from models.attention_branch import add_attention_branch
from models.lambda_resnet import lambda_resnet50, lambda_resnet26
from models.cnn import CNNModel, model_CNN_3

ALL_MODELS = ["lambda_resnet", "resnet", "resnet50", "CNN", "CNN3", "lambda_resnet26"]


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
    num_channel: int = 3,
    base_pretrained: Optional[str] = None,
    base_pretrained2: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    attention_branch: bool = False,
    division_layer: Optional[str] = None,
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
        theta_attention(float): Attention Branch入力時の閾値
                                この値よりattentionが低いピクセルを0にして入力

    Returns:
        nn.Module: 作成したモデル
    """
    # base modelの作成
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
    elif base_model == "CNN":
        model = CNNModel(num_channel=num_channel)
        layer_index = {"layer1": 4, "layer2": 4}
        add_flatten = False
    elif base_model == "CNN3":
        model = model_CNN_3(num_channels=num_channel, num_classes=2)
        layer_index = {"layer1": 4, "layer2": 4}
        add_flatten = False
    elif base_model == "lambda_resnet_26":
        model = lambda_resnet26()
        layer_index = {"layer1": -6, "layer2": -5, "layer3": -4}
        # コードを直接変更してnn.flattenにしているためFalse
        add_flatten = False
    else:
        return None

    # base_pretrainedがpathならば読み込み
    if base_pretrained is not None and os.path.isfile(base_pretrained):
        model.load_state_dict(torch.load(base_pretrained))
        print(f"base pretrained {base_pretrained} loaded.")

    # 最終層の出力をnum_classに変換
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
            theta_attention,
        )

    # pretrainedがpathならば読み込み
    if pretrained_path is not None and os.path.isfile(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"pretrained {pretrained_path} loaded.")

    return model
