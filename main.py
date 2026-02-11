from argparse import Namespace

import torch

from model import ZugzwangNet, train


def main():
    if torch.cuda.is_available():
        print("Cuda available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZugzwangNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    args = Namespace(
        batch_size=64,
        log_interval=10,
        iterations=50,
        num_games=1000,
        epochs=5,
    )

    train(args, model, device, optimizer)


if __name__ == "__main__":
    main()
