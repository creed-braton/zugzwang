import asyncio

import chess
import torch
from tqdm import tqdm

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
    batcher,
    num_simulations,
    temperature=1.0,
    greedy_threshold=30,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
):
    board = chess.Board()
    states = []
    policies = []
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        root = Node()

        await root.simulate(board, batcher.infer)
        root.add_dirichlet_noise(dirichlet_epsilon, dirichlet_alpha)
        for _ in range(num_simulations - 1):
            await root.simulate(board, batcher.infer)

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
        if move_count < greedy_threshold:
            visit_counts = torch.tensor(
                [root.children[m].visit_count for m in moves],
                dtype=torch.float32,
            )
            visit_counts = visit_counts ** (1.0 / temperature)
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
    greedy_threshold,
    history_steps=8,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
):
    batcher = InferenceBatcher(model, device, batch_size=batch_size, history_steps=history_steps)
    batcher_task = asyncio.create_task(batcher.run())

    total_moves = 0
    pbar = tqdm(total=num_games, unit="game", desc="Self-play")

    async def _play_and_track():
        nonlocal total_moves
        states, policies, values = await _generate_game(
            batcher,
            num_simulations,
            temperature,
            greedy_threshold,
            dirichlet_epsilon,
            dirichlet_alpha,
        )
        total_moves += len(states)

        postfix = {"moves": total_moves}
        if batcher.start_time is not None:
            elapsed = asyncio.get_event_loop().time() - batcher.start_time
            if elapsed > 0:
                postfix["inf/s"] = int(batcher.total_inferences / elapsed)
            if batcher.total_batches > 0:
                postfix["batch"] = round(
                    batcher.total_inferences / batcher.total_batches, 1
                )
        pbar.set_postfix(postfix)
        pbar.update(1)

        return states, policies, values

    results = await asyncio.gather(
        *[_play_and_track() for _ in range(num_games)]
    )

    pbar.close()

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

    return torch.stack(all_states), torch.stack(all_policies), torch.cat(all_values)


def self_play(
    model,
    device,
    num_games=512,
    num_simulations=400,
    batch_size=64,
    temperature=1.0,
    greedy_threshold=30,
    history_steps=8,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return asyncio.run(
        _self_play_async(
            model,
            device,
            num_games,
            num_simulations,
            batch_size,
            temperature,
            greedy_threshold,
            history_steps,
            dirichlet_epsilon,
            dirichlet_alpha,
        )
    )
