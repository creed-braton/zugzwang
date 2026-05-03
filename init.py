import argparse
from pathlib import Path

from engine.model import Config, Model


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    # Architecture
    parser.add_argument(
        "--res-blocks", type=int, default=8, help="number of residual blocks"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=128,
        help="number of convolution channels",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=8,
        help="number of board history steps to encode",
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay (L2 regularization)",
    )

    # Self-play
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="MCTS temperature"
    )
    parser.add_argument(
        "--greedy-threshold",
        type=int,
        default=30,
        help="move number after which greedy selection is used",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet noise alpha",
    )
    parser.add_argument(
        "--dirichlet-epsilon",
        type=float,
        default=0.25,
        help="Dirichlet noise mixing fraction",
    )

    # Training
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="number of training iterations",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="training epochs per iteration",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="training batch size"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="number of recent iterations to keep in replay buffer",
    )

    # MCTS
    parser.add_argument(
        "--c", type=float, default=1.25, help="UCB exploration constant"
    )
    parser.add_argument(
        "--fpu-reduction",
        type=float,
        default=0.2,
        help="first-play urgency reduction",
    )

    # Inference
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="inference server batch size",
    )
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=0.005,
        help="inference server batch wait timeout (seconds)",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="zugzwang-init",
        description="Create a fresh model checkpoint from CLI args.",
    )
    _add_config_args(parser)
    args = parser.parse_args()

    config = Config.from_args(args)
    model = Model.init(config)
    path = Path("models") / model.run_id / f"{model.model_id}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(path)


if __name__ == "__main__":
    main()
