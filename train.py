import argparse
import datetime
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
from torchinfo import summary
from torchvision.models.resnet import Bottleneck as TorchBottleneck
from tqdm import tqdm

from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from evaluate import test
from metrics.base import Metric
from models import ALL_MODELS, create_model
from models.attention_branch import AttentionBranchModel
from models.attention_branch import Bottleneck as ABNBottleneck
from models.lambda_resnet import Bottleneck as LambdaBottleneck
from optim import ALL_OPTIM, create_optimizer
from optim.sam import SAM
from utils.loss import calculate_loss
from utils.utils import fix_seed, module_generator, parse_with_config, save_json
from utils.visualize import save_attention_map


class EarlyStopping:
    """
    Atrributes:
        patience(int): 何回まで値が減少しなくても続けるか、デフォルト7
        delta(float) : 前回のlossに加えてどれだけ良くなったら改善したとみなすか、デフォルト0
        save_dir(str): チェックポイントを保存するディレクトリ、デフォルトは"." (実行しているディレクトリ)
    """

    def __init__(
        self, patience: int = 7, delta: float = 0, save_dir: str = "."
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.save_dir = save_dir

        self.counter: int = 0
        self.early_stop: bool = False
        self.best_val_loss: float = np.Inf

    def __call__(self, val_loss: float, net: nn.Module) -> str:
        if val_loss + self.delta < self.best_val_loss:
            log = f"({self.best_val_loss:.5f} --> {val_loss:.5f})"
            self._save_checkpoint(net)
            self.best_val_loss = val_loss
            self.counter = 0
            return log

        self.counter += 1
        log = f"(> {self.best_val_loss:.5f} {self.counter}/{self.patience})"
        if self.counter >= self.patience:
            self.early_stop = True
        return log

    def _save_checkpoint(self, net: nn.Module) -> None:
        save_path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save(net.state_dict(), save_path)


def set_parameter_trainable(module: nn.Module, is_trainable: bool = True) -> None:
    """
    moduleの全パラメータをis_trainable(bool)に設定

    Args:
        module(nn.Module): 対象のモジュール
        is_trainable(bool)  : パラメータを学習させるか
    """
    for param in module.parameters():
        param.requires_grad = is_trainable


def freeze_model(
    model: nn.Module,
    num_trainable_bottleneck: int = 0,
    fe_trainable: bool = False,
    ab_trainable: bool = False,
    perception_trainable: bool = False,
    final_trainable: bool = True,
) -> None:
    """
    モデルを凍結
    全体的に凍結した後、最終層だけtrainableにする
    その後、後ろからnum_trainable_bottleneck個だけtrainableに

    Args:
        num_trainable_bottleneck(int): 学習させるBottleneck数
        fe_trainable(bool): Feature Extractorを学習させるか
        ab_trainable(bool): Attention Branchを学習させるか
        perception_trainable(bool): Perception Branchを学習させるか

    Note:
        (fe|ab|perception)_trainableはAttentionBranchModel
        のときにのみ使用
        上記よりnum_trainable_bottleneckが優先される。
    """
    if isinstance(model, AttentionBranchModel):
        set_parameter_trainable(model.feature_extractor, fe_trainable)
        set_parameter_trainable(model.attention_branch, ab_trainable)
        set_parameter_trainable(model.perception_branch, perception_trainable)
        modules = module_generator(model.perception_branch, reverse=True)
    else:
        set_parameter_trainable(model, is_trainable=False)
        modules = module_generator(model, reverse=True)

    final_layer = modules.__next__()
    set_parameter_trainable(final_layer, final_trainable)

    num_bottleneck = 0
    for module in modules:
        if num_trainable_bottleneck <= num_bottleneck:
            break

        set_parameter_trainable(module)
        if isinstance(module, (TorchBottleneck, ABNBottleneck, LambdaBottleneck)):
            num_bottleneck += 1


def setting_learning_rate(
    model: nn.Module, lr: float, lr_linear: float, lr_ab: Optional[float] = None
) -> Iterable:
    """
    学習率をレイヤーごとに設定

    Args:
        model (nn.Module): 学習率を設定するモデル
        lr(float)       : 最終層/Attention Branch以外の学習率
        lr_linear(float): 最終層の学習率
        lr_ab(float)    : Attention Branchの学習率

    Returns:
        学習率を設定したIterable
        optim.Optimizerの引数に与える
    """
    if isinstance(model, AttentionBranchModel):
        if lr_ab is None:
            lr_ab = lr_linear
        params = [
            {"params": model.attention_branch.parameters(), "lr": lr_ab},
            {"params": model.perception_branch[:-1].parameters(), "lr": lr},
            {"params": model.perception_branch[-1].parameters(), "lr": lr_linear},
        ]
    else:
        params = [
            {"params": model[:-1].parameters(), "lr": lr},
            {"params": model[-1].parameters(), "lr": lr_linear},
        ]

    return params


def wandb_log(loss: float, metrics: Metric, phase: str) -> None:
    """
    wandbにログを出力
    わかりやすいように各指標の前にphaseを付け足す
    (e.g. Acc -> Train_Acc)

    Args:
        loss(float)    : 損失関数の値
        metircs(Metric): 評価指標
        phase(str)     : train / val / test
    """
    log_items = {f"{phase}_loss": loss}

    for metric, value in metrics.score().items():
        log_items[f"{phase}_{metric}"] = value

    wandb.log(log_items)


def train(
    dataloader: data.DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    metric: Metric,
    lambdas: Optional[Dict[str, float]] = None,
) -> Tuple[float, Metric]:
    total = 0
    total_loss: float = 0
    torch.autograd.set_detect_anomaly(True)

    model.train()
    for data in tqdm(dataloader, desc="Train: "):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = calculate_loss(criterion, outputs, labels, model, lambdas)
        loss.backward()
        total_loss += loss.item()
        metric.evaluate(outputs, labels)

        # OptimizerがSAMのとき2回backwardする
        if isinstance(optimizer, SAM):
            optimizer.first_step(zero_grad=True)
            loss_sam = calculate_loss(criterion, model(inputs), labels, model, lambdas)
            loss_sam.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        total += labels.size(0)

    train_loss = total_loss / total
    return train_loss, metric


def main(args: argparse.Namespace):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        args.dataset,
        args.batch_size,
        args.image_size,
        train_ratio=args.train_ratio,
    )
    data_params = get_parameter_depend_in_data_set(
        args.dataset, pos_weight=torch.Tensor(args.loss_weights).to(device)
    )

    # モデルの作成
    model = create_model(
        args.model,
        num_classes=len(data_params["classes"]),
        base_pretrained=args.base_pretrained,
        base_pretrained2=args.base_pretrained2,
        pretrained_path=args.pretrained,
        attention_branch=args.add_attention_branch,
        division_layer=args.div,
        theta_attention=args.theta_att,
    )
    assert model is not None, "Model name is invalid"
    freeze_model(
        model,
        args.trainable_bottleneck,
        not "fe" in args.freeze,
        not "ab" in args.freeze,
        not "pb" in args.freeze,
        not "linear" in args.freeze,
    )

    # Optimizerの作成
    params = setting_learning_rate(model, args.lr, args.lr_linear, args.lr_ab)
    optimizer = create_optimizer(
        args.optimizer, params, args.lr, args.weight_decay, args.momentum
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    criterion = data_params["criterion"]
    metric = data_params["metric"]

    # run_name の 作成 (for save_dir / wandb)
    if args.model is not None:
        config_file = os.path.basename(args.config)
        run_name = os.path.splitext(config_file)[0]
    else:
        run_name = args.model
    run_name += ["", f"_div{args.div}"][args.add_attention_branch]
    run_name = f"{run_name}_{now_str}"
    if args.run_name is not None:
        run_name = args.run_name

    save_dir = os.path.join(args.save_dir, run_name)
    assert not os.path.isdir(save_dir)
    os.makedirs(save_dir)
    best_path = os.path.join(save_dir, f"best.pt")

    configs = vars(args)
    configs.pop("config")  # モデルと一緒に保存するargsを適用した新configに旧configの情報が入らないように

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, save_dir=save_dir
    )

    wandb.init(project=args.dataset, name=run_name)
    wandb.config.update(configs)
    configs["pretrained"] = best_path
    save_json(configs, os.path.join(save_dir, "config.json"))

    # モデルの詳細表示（torchsummary）
    summary(model, (args.batch_size, 3, args.image_size, args.image_size))

    lambdas = {"att": args.lambda_att}

    model.to(device)

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}]")
        for phase, dataloader in dataloader_dict.items():
            if phase == "Train":
                dataloader.dataset.dataset.model = model
                loss, metric = train(
                    dataloader,
                    model,
                    criterion,
                    optimizer,
                    metric,
                    lambdas=lambdas,
                )
            else:
                loss, metric = test(
                    dataloader,
                    model,
                    criterion,
                    metric,
                    device,
                    phase,
                    lambdas=lambdas,
                )

            metric_log = metric.log()
            log = f"{phase}\t| {metric_log} Loss: {loss:.5f} "

            wandb_log(loss, metric, phase)

            if phase == "Val":
                early_stopping_log = early_stopping(loss, model)
                log += early_stopping_log
                scheduler.step(loss)

            print(log)
            metric.clear()
            if args.add_attention_branch:
                save_attention_map(
                    model.attention_branch.attention[0][0], "attention.png"
                )

        if early_stopping.early_stop:
            print("Early Stopping")
            model.load_state_dict(torch.load(os.path.join(save_dir, f"checkpoint.pt")))
            break

    torch.save(model.state_dict(), os.path.join(save_dir, f"best.pt"))
    print("Training Finished")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, help="path to config file (json)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false")

    # Model
    parser.add_argument("-m", "--model", choices=ALL_MODELS, help="model name")
    parser.add_argument(
        "-add_ab",
        "--add_attention_branch",
        action="store_true",
        help="add Attention Branch",
    )
    parser.add_argument(
        "--div",
        type=str,
        choices=["layer1", "layer2", "layer3"],
        default="layer2",
        help="place to attention branch",
    )
    parser.add_argument("--base_pretrained", type=str, help="path to base pretrained")
    parser.add_argument(
        "--base_pretrained2",
        type=str,
        help="path to base pretrained2 ( after change_num_classes() )",
    )
    parser.add_argument("--pretrained", type=str, help="path to pretrained")
    parser.add_argument(
        "--theta_att", type=float, default=0, help="threthold of attention branch"
    )

    # Freeze
    parser.add_argument(
        "--freeze",
        type=str,
        nargs="*",
        choices=["fe", "ab", "pb", "linear"],
        default=[],
        help="freezing layer",
    )
    parser.add_argument(
        "--trainable_bottleneck",
        type=int,
        default=0,
        help="number of trainable Bottlneck layer",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="ratio for train val split"
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="weights for label by class",
    )

    # Optimizer
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "-optim", "--optimizer", type=str, default="AdamW", choices=ALL_OPTIM
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_linear",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lr_ab",
        "--lr_attention_branch",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--factor", type=float, default=0.3333, help="new_lr = lr * factor"
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=2,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )

    parser.add_argument(
        "--lambda_att", type=float, default=0.1, help="weights for attention loss"
    )

    parser.add_argument(
        "--early_stopping_patience", type=int, default=6, help="Early Stopping patience"
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="path to save checkpoints"
    )

    parser.add_argument(
        "--run_name", type=str, help="save in save_dir/run_name and wandb name"
    )

    return parse_with_config(parser)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
