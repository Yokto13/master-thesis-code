import argparse
import logging
import os
import sys
import warnings

import gin
import numpy as np
import torch
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from logger import setup_logging

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer import Trainer


warnings.filterwarnings(
    "ignore", 
    message=".*CUDA is not available or torch_xla is imported.*"
)
warnings.filterwarnings(
    "ignore", 
    message=".*Consider using one of the following signatures instead.*addcmul_.*"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to gin config file")
    parser.add_argument("--name_specifier", type=str, default="", help="Optional name specifier")
    parser.add_argument("--wandb_group", type=str, default="", help="Optional wandb group name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--log_dir", type=str, default="logdir", help="Directory to save logs")
    parser.add_argument("--save_obs", action="store_true", help="Save buffer observations at end of training")
    parser.add_argument("--save_buffer", action="store_true", help="Save entire buffer at end of training")
    parser.add_argument("--load_buffer", type=str, default="", help="Path to saved buffer .npz to load")
    parser.add_argument("--reset_rnn_states", action="store_true", help="Zero RNN states in loaded buffer (for warm-start)")
    parser.add_argument("--supervised", action="store_true", help="Run in supervised learning mode")
    args = parser.parse_args()

    setup_logging(args.log_level)

    name = os.path.splitext(os.path.basename(args.config))[0]
    if args.name_specifier:
        name += f"_{args.name_specifier}"

    gin.parse_config_file(args.config)
    logger.debug("Active Gin Config:\n%s", gin.config_str())

    config_dict = {}
    for (scope, selector), value in gin.config._CONFIG.items():
        formatted_key = f"{scope}/{selector}" if scope else selector
        config_dict[formatted_key] = value

    artifact = wandb.Artifact(name=name, type="source_code")
    artifact.add_dir("src")

    logger.info("Starting training run: %s (group: %s)", name, args.wandb_group or "none")

    wandb.init(
        project="my-dreamer", name=name, group=args.wandb_group if args.wandb_group else None, config=config_dict
    )

    wandb.run.log_artifact(artifact)

    trainer = Trainer(
        seed=args.seed,
        run_name=name,
        save_obs=args.save_obs,
        save_buffer=args.save_buffer,
        load_buffer=args.load_buffer,
        reset_rnn_states=args.reset_rnn_states,
        log_dir=args.log_dir,
        supervised=args.supervised,
    )
    trainer.train()


if __name__ == "__main__":
    main()
