import argparse
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchinfo import summary
from tqdm import tqdm
import wandb

from data import ALL_DATASETS, get_dataset_params, setting_dataset
from metrics.base import Metric
from metrics.insdel import InsertionDeletion
from metrics.block_insdel import BlockInsertionDeletion
from models import ALL_MODELS, create_model
from models.attention_branch import AttentionBranchModel
from models.lambda_resnet import Bottleneck
from models.rise import RISE
from utils.utils import fix_seed, module_generator, parse_with_config
from utils.visualize import save_image, save_image_with_attention_map


def calculate_attention(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    method: str,
    rise_params: Optional[Dict[str, Any]],
    fname: Optional[str],
) -> Tuple[np.ndarray, bool]:
    correct = True
    if method == "ABN":
        assert isinstance(model, AttentionBranchModel)
        y = model(image)
        # print(torch.argmax(y[0]), label)
        if torch.argmax(y[0]).item() != label.item():
            correct = False
        attentions = model.attention_branch.attention  # (1, W, W)
        attention = attentions[0]
        attention: np.ndarray = attention.cpu().numpy()
    elif method == "RISE":
        assert rise_params is not None
        rise_model = RISE(
            model,
            n_masks=rise_params["n_masks"],
            p1=rise_params["p1"],
            input_size=rise_params["input_size"],
            initial_mask_size=rise_params["initial_mask_size"],
            n_batch=rise_params["n_batch"],
            mask_path=rise_params["mask_path"],
        )
        attentions = rise_model(image)  # (N_class, W, H)
        attention = attentions[label]
        attention: np.ndarray = attention.cpu().numpy()
    elif method == "npy":
        attention: np.ndarray = np.load(fname)
    elif method == "lambda":
        _ = model(image)
        num_bt = 0
        for module in module_generator(model):
            if isinstance(module, Bottleneck):
                num_bt += 1
                if 2 == num_bt:
                    attention = module.yc
                    attention = (attention - attention.min()) / (
                        attention.max() - attention.min()
                    )
                    break

        attention = attention.cpu().numpy()
    else:
        raise ValueError

    return attention, correct


@torch.no_grad()
def visualize(
    dataloader: data.DataLoader,
    att_model: nn.Module,
    base_model: nn.Module,
    method: str,
    batch_size: int,
    save_dir: str,
    device: torch.device,
    params: Dict[str, Any],
    rise_params: Dict[str, Any],
    evaluate: bool = False,
    block_size: int = 32,
    attention_dir: Optional[str] = None,
) -> Union[None, Metric]:
    if evaluate:
        metrics = BlockInsertionDeletion()
        insdel_save_dir = os.path.join(save_dir, "insdel")
        if not os.path.isdir(insdel_save_dir):
            os.makedirs(insdel_save_dir)

    att_model.eval()
    for i, data in enumerate(tqdm(dataloader, desc=f"Visualizing: ")):

        inputs, labels = data[0].to(device), data[1].to(device)
        image = inputs[0].cpu().numpy()
        label = labels[0]

        base_fname = f"{i+1}_{params['classes'][label]}"

        attention_fname = None
        if attention_dir is not None:
            attention_fname = os.path.join(attention_dir, f"{base_fname}.npy")

        attention, correct = calculate_attention(
            att_model, inputs, label, method, rise_params, attention_fname
        )
        if attention is None:
            continue
        # np.save(f"{save_dir}/{base_fname}.npy", attention)

        if evaluate:
            metrics.evaluate(
                base_model,
                image.copy(),
                attention,
                label,
                batch_size,
                args.insdel_step,
                device,
                block_size=block_size,
            )
            metrics.save_roc_curve(insdel_save_dir)
            base_fname = f"{base_fname}_{metrics.ins_auc - metrics.del_auc:.4f}"

        save_fname = os.path.join(
            save_dir, f"{base_fname}_{['x', 'o'][metrics.correct]}.png"
        )
        save_image_with_attention_map(
            image, attention, save_fname, params["mean"], params["std"]
        )

    if evaluate:
        return metrics


def main() -> None:
    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成
    dataset_params = {"loss_weights": torch.Tensor(args.loss_weights).to(device)}
    dataloader, model_params, _, _ = setting_dataset(
        args.dataset, 1, args.image_size, is_test=True, dataset_params=dataset_params
    )
    assert isinstance(dataloader, data.DataLoader)
    params = get_dataset_params(args.dataset)

    mask_path = os.path.join(args.root_dir, "masks.npy")
    if not os.path.isfile(mask_path):
        mask_path = None
    rise_params = {
        "n_masks": args.num_masks,
        "p1": args.p1,
        "input_size": (args.image_size, args.image_size),
        "initial_mask_size": (args.rise_scale, args.rise_scale),
        "n_batch": args.batch_size,
        "mask_path": mask_path,
    }

    # モデルの作成
    attention_model = create_model(
        args.model,
        num_classes=model_params["num_classes"],
        base_pretrained=args.base_pretrained,
        base_pretrained2=args.base_pretrained2,
        pretrained_path=args.pretrained,
        attention_branch=args.add_attention_branch,
        division_layer=args.div,
        multi_task=model_params["is_multi_task"],
        num_tasks=model_params["num_tasks"],
        theta_attention=args.theta_att,
    )
    assert attention_model is not None, "Model name is invalid"

    if args.orig_model:
        eval_model = "orig"
        base_model = create_model(
            args.model,
            num_classes=model_params["num_classes"],
            pretrained_path="checkpoints/idrid_lambda-base.pt",
            attention_branch=False,
            division_layer=args.div,
            multi_task=model_params["is_multi_task"],
            num_tasks=model_params["num_tasks"],
        )
        assert base_model is not None
    else:
        eval_model = "same"
        base_model = attention_model

    # run_nameをpretrained pathから取得
    # checkpoints/run_name/checkpoint.pt -> run_name
    run_name = args.pretrained.split(os.sep)[-2]
    save_dir = os.path.join(
        "outputs",
        f"{args.method}{args.block_size}_eval{eval_model}_{run_name}",
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    summary(attention_model, (args.batch_size, 3, args.image_size, args.image_size))

    attention_model.to(device)
    base_model.to(device)

    wandb.init(project=args.dataset, name=run_name)
    wandb.config.update(vars(args))

    metrics = visualize(
        dataloader,
        attention_model,
        base_model,
        args.method,
        args.batch_size,
        save_dir,
        device,
        params,
        rise_params,
        args.visualize_only,
        block_size=args.block_size,
        attention_dir=args.attention_dir,
    )

    if metrics is not None:
        print(metrics.log())
        print(metrics.log_gspread())
        wandb.run.summary["insertion"] = metrics.insertion()
        wandb.run.summary["deletion"] = metrics.deletion()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        "--orig_model",
        action="store_true",
        help="calc insdel score by using original model",
    )
    parser.add_argument(
        "--theta_att", type=float, default=0, help="threthold of attention branch"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="weights for label by class",
    )

    parser.add_argument("--root_dir", type=str, default="./outputs/")
    parser.add_argument("--visualize_only", action="store_false")
    parser.add_argument("--insdel_step", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=32)

    parser.add_argument(
        "--method", type=str, choices=["ABN", "RISE", "npy", "lambda"], default="ABN"
    )

    parser.add_argument("--num_masks", type=int, default=10000)
    parser.add_argument("--rise_scale", type=int, default=9)
    parser.add_argument(
        "--p1", type=float, default=0.3, help="percentage of mask [pixel = (0, 0, 0)]"
    )

    parser.add_argument("--attention_dir", type=str, help="path to attention npy file")

    args = parse_with_config(parser)

    main()
