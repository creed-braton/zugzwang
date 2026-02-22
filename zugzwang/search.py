from __future__ import annotations

import math
from typing import Callable

import chess
import torch

from .encode import legal_moves


class Node:
    def __init__(self, parent: "Node" | None = None, prior=0.0):
        self.parent = parent
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.children: dict[chess.Move, Node] = {}

    def _select(self, c):
        best_score = -float("inf")
        best_move = None
        best_node = None

        for move, child in self.children.items():
            if child.visit_count == 0:
                quality = 0.0
            else:
                quality = -child.value_sum / child.visit_count

            ucb = (
                c
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = quality + ucb

            if score > best_score:
                best_score = score
                best_move = move
                best_node = child

        return best_move, best_node

    def _expand(self, board: chess.Board, policy):
        moves, indices = legal_moves(board)
        priors = policy[indices]
        priors /= priors.sum()
        for move, prior in zip(moves, priors):
            self.children[move] = Node(prior=prior.item(), parent=self)

    def add_dirichlet_noise(self, epsilon, alpha):
        children = list(self.children.values())
        noise = torch.distributions.Dirichlet(
            torch.full((len(children),), alpha)
        ).sample()
        for child, n in zip(children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * n.item()

    async def simulate(
        self,
        board: chess.Board,
        inference: Callable,
        c=1.41,
    ):
        node = self
        depth = 0

        while node.children:
            move, node = node._select(c)
            board.push(move)
            depth += 1

        if board.is_game_over():
            value = -1.0 if board.is_checkmate() else 0.0
        else:
            policy, value = await inference(board)
            node._expand(board, policy)

        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

        for _ in range(depth):
            board.pop()
