import argparse
import json
import sys


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
