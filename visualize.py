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

from data import ALL_DATASETS, get_parameter_depend_in_data_set, create_dataloader_dict
from evaluate import test
from metrics.base import Metric
from metrics.patch_insdel import PatchInsertionDeletion
from models import ALL_MODELS, create_model
from models.attention_branch import AttentionBranchModel
from models.lambda_resnet import Bottleneck
from models.rise import RISE
from utils.utils import fix_seed, module_generator, parse_with_config
from utils.visualize import save_image_with_attention_map


def calculate_attention(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    method: str,
    rise_params: Optional[Dict[str, Any]],
    fname: Optional[str],
) -> Tuple[np.ndarray, bool]:
    if method == "ABN":
        assert isinstance(model, AttentionBranchModel)
        y = model(image)
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
        assert fname is not None
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

    return attention


@torch.no_grad()
def visualize(
    dataloader: data.DataLoader,
    model: nn.Module,
    method: str,
    batch_size: int,
    patch_size: int,
    step: int,
    save_dir: str,
    all_class: bool,
    params: Dict[str, Any],
    device: torch.device,
    evaluate: bool = False,
    attention_dir: Optional[str] = None,
    mask_mode: str = "black",
) -> Union[None, Metric]:
    if evaluate:
        pid = PatchInsertionDeletion(
            model, batch_size, patch_size, step, params["name"], device, mask_mode=mask_mode, dataloader=dataloader
        )
        insdel_save_dir = os.path.join(save_dir, "insdel")
        if not os.path.isdir(insdel_save_dir):
            os.makedirs(insdel_save_dir)

    model.eval()
    for i, data in enumerate(tqdm(dataloader, desc=f"Visualizing: ")):
        inputs, labels = data[0].to(device), data[1].to(device) # inputs : (1, 1, 512, 512)
        image = inputs[0].cpu().numpy()
        label = labels[0]

        if label != 1 and not all_class:
            continue

        base_fname = f"{i+1}_{params['classes'][label]}"

        attention_fname = None
        if attention_dir is not None:
            attention_fname = os.path.join(attention_dir, f"{base_fname}.npy")

        attention = calculate_attention(
            model, inputs, label, method, params, attention_fname
        ) # (1, 228, 228)
        if attention is None:
            continue
        if method == "RISE":
            np.save(f"{save_dir}/{base_fname}.npy", attention)

        if evaluate:
            pid.evaluate(
                image.copy(),
                attention,
                label,
            )
            pid.save_roc_curve(insdel_save_dir)
            base_fname = f"{base_fname}_{pid.ins_auc - pid.del_auc:.4f}"

        save_fname = os.path.join(save_dir, f"{base_fname}.png")
        save_image_with_attention_map(
            image, attention, save_fname, params["mean"], params["std"]
        )

    if evaluate:
        return pid


def main(args: argparse.Namespace) -> None:
    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        args.dataset, 1, args.image_size, only_test=True
    )
    dataloader = dataloader_dict["Test"]
    assert isinstance(dataloader, data.DataLoader)

    params = get_parameter_depend_in_data_set(args.dataset, pos_weight=torch.Tensor(args.loss_weights).to(device))

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
    params.update(rise_params)

    # モデルの作成
    model = create_model(
        args.model,
        num_classes=len(params["classes"]),
        num_channel=params["num_channel"],
        base_pretrained=args.base_pretrained,
        base_pretrained2=args.base_pretrained2,
        pretrained_path=args.pretrained,
        attention_branch=args.add_attention_branch,
        division_layer=args.div,
        theta_attention=args.theta_att,
    )
    assert model is not None, "Model name is invalid"

    # run_nameをpretrained pathから取得
    # checkpoints/run_name/checkpoint.pt -> run_name
    run_name = args.pretrained.split(os.sep)[-2]
    save_dir = os.path.join(
        "outputs",
        f"{run_name}_{args.method}{args.block_size}",
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    summary(
        model,
        (args.batch_size, params["num_channel"], args.image_size, args.image_size),
    )

    model.to(device)

    wandb.init(project=args.dataset, name=run_name)
    wandb.config.update(vars(args))

    metrics = visualize(
        dataloader,
        model,
        args.method,
        args.batch_size,
        args.block_size,
        args.insdel_step,
        save_dir,
        args.all_class,
        params,
        device,
        args.visualize_only,
        attention_dir=args.attention_dir,
        mask_mode = args.mask_mode
    )

    test_acc = args.test_acc
    print(f"Test_acc : {test_acc}")
    if metrics is not None:
        print(metrics.log())
        for key, value in metrics.score().items():
            wandb.run.summary[key] = value


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--all_class", action="store_true")
    # recommend (size, step) in 512x512 = (1, 10000), (2, 2500), (4, 500), (8, 100), (16, 20), (32, 10), (64, 5), (128, 1)
    # recommend (size, step) in 224x224 = (1, 500), (2, 100), (4, 20), (8, 10), (16, 5), (32, 1)
    parser.add_argument("--insdel_step", type=int, default=10)
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
    parser.add_argument("--test_acc", type=float, help="test_acc when best val_loss")
    parser.add_argument("--mask_mode", type=str, choices=["black", "mean", "blur", "base"], default="black")

    return parse_with_config(parser)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
