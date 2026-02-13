import os
from datetime import datetime

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import POLICY_DIM, board_to_tensor
from mcts import mcts_search


class ZugzwangNet(nn.Module):
    def __init__(self):
        super(ZugzwangNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.embedding = nn.Linear(64 * 8 * 8, 512)
        self.policy_head = nn.Linear(512, POLICY_DIM)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # shape: (batch_size, 128*8*8)

        embedding = F.relu(self.embedding(x))  # shape: (batch_size, 512)

        policy = self.policy_head(embedding)
        value = torch.tanh(self.value_head(embedding))

        return embedding, policy, value


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def _train_epoch(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, policy_target, value_target) in enumerate(
        train_loader
    ):
        data, policy_target, value_target = (
            data.to(device),
            policy_target.to(device),
            value_target.to(device),
        )

        optimizer.zero_grad()
        _, policy_pred, value_pred = model(data)

        log_probs = F.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def train(args, model, device, optimizer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"zugzwang_{timestamp}.pth")

    for iteration in range(1, args.iterations + 1):
        print(
            f"\n--- Iteration {iteration}/{args.iterations}: generating {args.num_games} games ---"
        )
        dataset = _generate_games(
            model,
            device,
            num_games=args.num_games,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            temp_threshold=args.temp_threshold,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        for epoch in range(1, args.epochs + 1):
            _train_epoch(
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                args.log_interval,
            )

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path} (iteration {iteration})")


def test(model, device, test_loader):
    model.eval()
    test_policy_loss = 0
    test_value_loss = 0
    correct_policy = 0

    with torch.no_grad():
        for data, policy_target, value_target in test_loader:
            data, policy_target, value_target = (
                data.to(device),
                policy_target.to(device),
                value_target.to(device),
            )
            _, policy_pred, value_pred = model(data)

            log_probs = F.log_softmax(policy_pred, dim=1)
            test_policy_loss += (
                -(policy_target * log_probs).sum(dim=1).sum().item()
            )
            test_value_loss += F.mse_loss(
                value_pred.squeeze(), value_target, reduction="sum"
            ).item()

            pred_policy = policy_pred.argmax(dim=1)
            target_policy = policy_target.argmax(dim=1)
            correct_policy += pred_policy.eq(target_policy).sum().item()

    test_policy_loss /= len(test_loader.dataset)
    test_value_loss /= len(test_loader.dataset)
    policy_accuracy = 100.0 * correct_policy / len(test_loader.dataset)

    print(
        f"\nTest set: Policy Loss: {test_policy_loss:.4f}, Value Loss: {test_value_loss:.4f}, "
        f"Policy Accuracy: {correct_policy}/{len(test_loader.dataset)} ({policy_accuracy:.0f}%)\n"
    )


def _generate_games(
    model: ZugzwangNet,
    device,
    num_games: int = 100,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temp_threshold: int = 30,
) -> ChessDataset:
    model.eval()
    all_states = []
    all_policies = []
    all_values = []

    for game_idx in range(num_games):
        board = chess.Board()
        game_states = []
        game_policies = []
        move_count = 0

        while not board.is_game_over():
            state = board_to_tensor(board)

            # Temperature annealing: explore early, go greedy later
            use_exploration = move_count < temp_threshold
            temp = temperature if use_exploration else 0.0

            move, policy_target = mcts_search(
                board,
                model,
                device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature=temp,
                add_noise=use_exploration,
            )

            game_states.append(state)
            game_policies.append(policy_target)
            board.push(move)
            move_count += 1

        result = board.result()
        if result == "1-0":
            game_result = 1.0
        elif result == "0-1":
            game_result = -1.0
        else:
            game_result = 0.0

        for turn_idx in range(len(game_states)):
            # Even turns = white to move, odd turns = black to move
            value = game_result if turn_idx % 2 == 0 else -game_result
            all_values.append(value)

        all_states.extend(game_states)
        all_policies.extend(game_policies)

        if (game_idx + 1) % 100 == 0 or game_idx == 0:
            print(f"  Generated {game_idx + 1}/{num_games} games")

    return ChessDataset(
        torch.stack(all_states),
        torch.stack(all_policies),
        torch.tensor(all_values, dtype=torch.float32),
    )
