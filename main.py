import argparse
import logging
import sys
import uuid

from zugzwang.train import train_self_play


def main():
    parser = argparse.ArgumentParser(description="Zugzwang chess training")

    # Model architecture
    parser.add_argument(
        "--num-blocks", type=int, default=6, help="number of residual blocks"
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
        "--num-games", type=int, default=512, help="games per iteration"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=400,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="MCTS temperature"
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=30,
        help="move number after which greedy selection is used",
    )

    # Training
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="number of training iterations",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="training epochs per iteration"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="log every N batches"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="use CUDA if available",
    )
    parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda", help="disable CUDA"
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="UUID of a previous run to resume",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("zugzwang")

    run_id = uuid.UUID(args.resume) if args.resume else None
    train_self_play(args, logger, id=run_id)


if __name__ == "__main__":
    main()
