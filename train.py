import argparse
import logging
import sys
import uuid
from pathlib import Path

from engine.model import Model


def main():
    parser = argparse.ArgumentParser(
        prog="zugzwang-train",
        description="Train a model from its checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="use CUDA if available",
    )
    parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda", help="disable CUDA"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="log every N batches"
    )
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

    from engine.train import train

    model = Model.load(args.checkpoint)
    run_id = uuid.UUID(args.resume) if args.resume else uuid.uuid4()
    train(
        model,
        checkpoint=args.checkpoint,
        run_id=run_id,
        log_interval=args.log_interval,
        cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
