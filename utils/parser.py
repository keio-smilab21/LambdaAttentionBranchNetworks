import argparse
import json
import sys


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
        choices=["fe", "ab", "perception", "linear"],
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
        "--lambda_var", type=float, default=1, help="weights for variance loss"
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

    args = parse_with_config(parser)

    return args


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    argparseとjsonファイルのconfigを共存させる

    Args:
        parser(ArgumentParser)

    Returns:
        Namespace: argparserと同様に使用可能

    Note:
        configより引数に指定した値が優先
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args
