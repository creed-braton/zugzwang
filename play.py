import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import chess
import torch
import torch.multiprocessing as mp

from engine.dataset import Dataset
from engine.encode import POLICY_DIM, board_to_tensor, legal_moves
from engine.infer import InferenceClient, InferenceServer
from engine.model import Config, Model
from engine.search import Node


def _visit_count_policy(
    root: Node, moves: list[chess.Move], indices: torch.Tensor
) -> torch.Tensor:
    policy = torch.zeros(POLICY_DIM)
    counts = torch.tensor(
        [root.children[m].visit_count for m in moves],
        dtype=torch.float32,
    )
    total = counts.sum()
    if total > 0:
        counts = counts / total
    policy[indices] = counts
    return policy


def _select_move(
    root: Node,
    moves: list[chess.Move],
    move_count: int,
    greedy_threshold: int,
) -> chess.Move:
    counts = torch.tensor(
        [root.children[m].visit_count for m in moves],
        dtype=torch.float32,
    )
    if move_count < greedy_threshold:
        probs = counts / counts.sum()
        idx = int(torch.multinomial(probs, 1).item())
    else:
        idx = int(counts.argmax().item())
    return moves[idx]


def _top_line_depth(root: Node) -> int:
    depth = 0
    node = root
    while node.children:
        node = max(node.children.values(), key=lambda c: c.visit_count)
        depth += 1
    return depth


def _tree_size(root: Node) -> int:
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children.values())
    return count


async def _play_game(
    client: InferenceClient, config: Config
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    list[float],
    list[int],
    list[int],
    float,
    str,
]:
    board = chess.Board()
    states: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    search_values: list[float] = []
    tree_sizes: list[int] = []
    top_line_depths: list[int] = []
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        root = Node()
        await root.simulate(
            board, client.infer, c=config.c, fpu_reduction=config.fpu_reduction
        )
        # add noise after the first simulate has expanded the root
        root.add_dirichlet_noise(
            config.dirichlet_epsilon, config.dirichlet_alpha
        )
        for _ in range(config.num_simulations - 1):
            await root.simulate(
                board,
                client.infer,
                c=config.c,
                fpu_reduction=config.fpu_reduction,
            )

        moves, indices = legal_moves(board)
        states.append(board_to_tensor(board, config.history_steps))
        policies.append(_visit_count_policy(root, moves, indices))
        search_values.append(root.value_sum / root.visit_count)
        tree_sizes.append(_tree_size(root))
        top_line_depths.append(_top_line_depth(root))

        move = _select_move(root, moves, move_count, config.greedy_threshold)
        board.push(move)
        move_count += 1

    result = board.outcome(claim_draw=True)
    if result.winner is chess.WHITE:
        outcome = 1.0
    elif result.winner is chess.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0
    termination = result.termination.name

    n = len(states)
    values = torch.tensor(
        [outcome if i % 2 == 0 else -outcome for i in range(n)],
        dtype=torch.float32,
    )

    return (
        states,
        policies,
        values,
        search_values,
        tree_sizes,
        top_line_depths,
        outcome,
        termination,
    )


async def _worker_main(
    client: InferenceClient,
    num_games: int,
    config: Config,
    model_id: str,
    run_id: str,
    iteration: int,
    worker_id: int,
    out_dir: Path,
    concurrent_games: int,
) -> None:
    log = logging.getLogger(f"worker-{worker_id}")
    log.info(
        "starting: num_games=%d concurrent=%d", num_games, concurrent_games
    )

    await client.connect()

    sem = asyncio.Semaphore(concurrent_games)
    games_done = 0
    positions_total = 0
    outcome_hist: dict[float, int] = {}
    termination_hist: dict[str, int] = {}

    async def run_one() -> None:
        nonlocal games_done, positions_total
        async with sem:
            result = await _play_game(client, config)

        # write the finished game to disk and drop it from memory before
        # other concurrent games push peak usage further
        ds = Dataset(
            model_id=model_id,
            num_simulations=config.num_simulations,
            iteration=iteration,
            run_id=run_id,
            worker_id=worker_id,
        )
        game_id = ds.add_game(*result)
        await asyncio.to_thread(ds.save, out_dir / f"game-{game_id}.pth")

        outcome = result[6]
        termination = result[7]
        games_done += 1
        positions_total += len(ds)
        outcome_hist[outcome] = outcome_hist.get(outcome, 0) + 1
        termination_hist[termination] = termination_hist.get(termination, 0) + 1

    await asyncio.gather(*[run_one() for _ in range(num_games)])

    await client.disconnect()

    log.info(
        "done: games=%d positions=%d outcomes=%s terminations=%s",
        games_done,
        positions_total,
        outcome_hist,
        termination_hist,
    )


def _worker_entrypoint(
    client: InferenceClient,
    num_games: int,
    config: Config,
    model_id: str,
    run_id: str,
    iteration: int,
    worker_id: int,
    out_dir: Path,
    concurrent_games: int,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [w{worker_id}] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    asyncio.run(
        _worker_main(
            client,
            num_games,
            config,
            model_id,
            run_id,
            iteration,
            worker_id,
            out_dir,
            concurrent_games,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="zugzwang-play",
        description="Run self-play from a checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--num-games",
        type=int,
        default=2048,
        help="number of self-play games to generate",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="number of worker processes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets"),
        help="directory to write per-game dataset files",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=32,
        help="max concurrent games per worker process",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("play")

    model = Model.load(args.checkpoint)
    config = model.config
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    workers = min(args.workers, args.num_games)
    base, rem = divmod(args.num_games, workers)
    games_per_worker = [base + (1 if i < rem else 0) for i in range(workers)]

    log.info(
        "loaded checkpoint %s (model_id=%s run_id=%s iteration=%d)",
        args.checkpoint,
        model.model_id,
        model.run_id,
        model.iteration,
    )
    log.info(
        "self-play: device=%s workers=%d games=%d sims=%d concurrent=%d",
        device,
        workers,
        args.num_games,
        config.num_simulations,
        args.concurrent_games,
    )

    server = InferenceServer(
        model=model.net,
        device=device,
        num_clients=workers,
        batch_size=config.inference_batch_size,
        timeout=config.inference_timeout,
        history_steps=config.history_steps,
        model_id=model.model_id,
    )

    with server:
        ctx = mp.get_context("spawn")
        procs = []
        for i, n in enumerate(games_per_worker):
            proc = ctx.Process(
                target=_worker_entrypoint,
                args=(
                    server.clients[i],
                    n,
                    config,
                    model.model_id,
                    model.run_id,
                    model.iteration,
                    i,
                    args.out_dir,
                    args.concurrent_games,
                ),
            )
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    log.info(
        "inference: total=%d batches=%d",
        server.total_inferences,
        server.total_batches,
    )


if __name__ == "__main__":
    main()
