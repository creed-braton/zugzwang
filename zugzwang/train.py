import os
import uuid
from logging import Logger

import torch
import torch.nn.functional as F

from .dataset import self_play
from .net import Net


def train_self_play(args, logger: Logger, id: uuid.UUID | None = None):
    os.makedirs("models", exist_ok=True)

    hyper_parameters = vars(args)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    input_dim = 14 * args.history_steps + 7
    model = Net(input_dim, num_blocks=args.num_blocks, channels=args.channels).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_iteration = 0

    if id is None:
        id = uuid.uuid4()
        logger.info("Starting new training run %s", id)
    else:
        checkpoint_path = os.path.join("models", f"{id}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint["iteration"]

        saved_params = checkpoint.get("hyper_parameters", {})
        diff = {
            k: (saved_params.get(k), v)
            for k, v in hyper_parameters.items()
            if saved_params.get(k) != v
        }
        if diff:
            logger.warning("Hyper-parameter mismatch with checkpoint:")
            for k, (old, new) in diff.items():
                logger.warning("  %s: %s -> %s", k, old, new)
            if input("Continue with new hyper-parameters? [y/N] ").lower() != "y":
                logger.info("Aborting training")
                return

        logger.info(
            "Resuming run %s from iteration %d", id, start_iteration
        )

    save_path = os.path.join("models", f"{id}.pth")

    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        logger.info(
            "Iteration %d/%d: generating %d games",
            iteration, start_iteration + args.iterations, args.num_games,
        )

        dataset = self_play(
            model,
            device,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            temperature=args.temperature,
            temp_threshold=args.temp_threshold,
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )

        model.train()
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, policy_target, value_target) in enumerate(loader):
                data = data.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device)

                optimizer.zero_grad()
                policy_pred, value_pred = model(data)

                log_probs = F.log_softmax(policy_pred, dim=1)
                policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred.squeeze(), value_target)
                total_loss = policy_loss + value_loss

                total_loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    logger.info(
                        "Epoch %d [%d/%d (%.0f%%)] total_loss=%.4f policy_loss=%.4f value_loss=%.4f",
                        epoch,
                        batch_idx * len(data),
                        len(loader.dataset),
                        100.0 * batch_idx / len(loader),
                        total_loss.item(),
                        policy_loss.item(),
                        value_loss.item(),
                    )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "hyper_parameters": hyper_parameters,
            },
            save_path,
        )
        logger.info("Model saved to %s (iteration %d)", save_path, iteration)
