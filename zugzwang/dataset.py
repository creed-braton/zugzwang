import asyncio

import chess
import torch

from .encode import POLICY_DIM, board_to_tensor, legal_moves
from .infer import InferenceBatcher
from .search import Node


class Dataset(torch.utils.data.Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


async def _generate_game(
    batcher, num_simulations, temperature=1.0, temp_threshold=30
):
    board = chess.Board()
    states = []
    policies = []
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        root = Node()

        for _ in range(num_simulations):
            await root.simulate(board, batcher.infer, temperature=temperature)

        states.append(board_to_tensor(board, batcher.history_steps))

        # Policy target: normalized visit counts over all legal moves
        policy_target = torch.zeros(POLICY_DIM)
        moves, indices = legal_moves(board)
        for move, idx in zip(moves, indices):
            if move in root.children:
                policy_target[idx] = root.children[move].visit_count
        policy_target /= policy_target.sum()
        policies.append(policy_target)

        # Move selection: sample early, greedy later
        if move_count < temp_threshold:
            visit_counts = torch.tensor(
                [root.children[m].visit_count for m in moves],
                dtype=torch.float32,
            )
            visit_counts /= visit_counts.sum()
            move_idx = torch.multinomial(visit_counts, 1).item()
        else:
            move_idx = max(
                range(len(moves)),
                key=lambda i: root.children[moves[i]].visit_count,
            )

        board.push(moves[move_idx])
        move_count += 1

    # Game outcome from white's perspective
    result = board.result(claim_draw=True)
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    # Value targets from each position's current player perspective
    values = torch.tensor(
        [outcome if i % 2 == 0 else -outcome for i in range(len(states))],
        dtype=torch.float32,
    )

    return states, policies, values


async def _self_play_async(
    model,
    device,
    num_games,
    num_simulations,
    batch_size,
    temperature,
    temp_threshold,
):
    batcher = InferenceBatcher(model, device, batch_size=batch_size)
    batcher_task = asyncio.create_task(batcher.run())

    results = await asyncio.gather(
        *[
            _generate_game(
                batcher, num_simulations, temperature, temp_threshold
            )
            for _ in range(num_games)
        ]
    )

    batcher_task.cancel()
    try:
        await batcher_task
    except asyncio.CancelledError:
        pass

    all_states = []
    all_policies = []
    all_values = []
    for states, policies, values in results:
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.append(values)

    return Dataset(
        torch.stack(all_states),
        torch.stack(all_policies),
        torch.cat(all_values),
    )


def self_play(
    model,
    device,
    num_games=512,
    num_simulations=400,
    batch_size=64,
    temperature=1.0,
    temp_threshold=30,
) -> Dataset:
    return asyncio.run(
        _self_play_async(
            model,
            device,
            num_games,
            num_simulations,
            batch_size,
            temperature,
            temp_threshold,
        )
    )
