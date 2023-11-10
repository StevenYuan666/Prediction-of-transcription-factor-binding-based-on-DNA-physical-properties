from argparse import ArgumentParser, ArgumentTypeError


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def parse_args():  # Parse command line arguments
    parser = ArgumentParser(description="mlm_seq")
    parser.add_argument(
        "--use_data", default=10, type=int, help="The number of datum used"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--lr", default=2e-4, type=float, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--use_pretrained",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes using pretrained model from hugging face",
    )
    parser.add_argument(
        "--log_wandb",
        type=str2bool,
        nargs="?",
        default=False,
        help="True denotes using wandb to log the training process",
    )
    parser.add_argument(
        "--seed",
        default=1205,
        type=int,
        help="Seeds helps the reproducibility of the experiment",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-1,
        type=float,
        help="Weight Decay for the optimizer",
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train", type=str)
    parser.add_argument(
        "--output_dir", type=str, default="saved/best_model/", help="Output directory"
    )
    parser.add_argument(
        "--load_best_model_at_end", type=str2bool, nargs="?", default=True
    )
    args = parser.parse_args()
    return args
