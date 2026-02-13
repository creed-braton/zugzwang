import math

import chess
import numpy as np
import torch

from encode import POLICY_DIM, _get_legal_moves, _get_move_index, board_to_tensor


class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def ucb_score(parent, child, c_puct=1.5):
    q = child.value()
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q + u


def select_leaf(node):
    while node.is_expanded and node.children:
        _, node = max(node.children.items(), key=lambda kv: ucb_score(node, kv[1]))
    return node


def expand_node(node, model, device):
    board = node.board

    # Terminal states
    if board.is_checkmate():
        node.is_expanded = True
        return -1.0  # Side to move lost
    if board.is_game_over():
        node.is_expanded = True
        return 0.0  # Draw

    # Neural network evaluation
    state = board_to_tensor(board).unsqueeze(0).to(device)
    _, policy_logits, value = model(state)
    value = value.item()

    # Get legal moves and their priors via softmax over legal logits
    moves, indices = _get_legal_moves(board)
    legal_logits = policy_logits.squeeze(0)[indices]
    priors = torch.softmax(legal_logits, dim=0)

    # Create child nodes
    flip = not board.turn
    for i, move in enumerate(moves):
        child_board = board.copy()
        child_board.push(move)
        child = MCTSNode(child_board, parent=node, move=move, prior=priors[i].item())
        move_idx = _get_move_index(move, flip)
        node.children[move_idx] = child

    node.is_expanded = True
    return value  # From current player's perspective


def backpropagate(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += -value
        value = -value
        node = node.parent


def add_dirichlet_noise(root, epsilon=0.25, alpha=0.3):
    if not root.children:
        return
    noise = np.random.dirichlet([alpha] * len(root.children))
    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


@torch.no_grad()
def mcts_search(
    board,
    model,
    device,
    num_simulations=100,
    c_puct=1.5,
    temperature=1.0,
    add_noise=True,
):
    root = MCTSNode(board.copy())

    # Expand root
    value = expand_node(root, model, device)
    backpropagate(root, value)

    if add_noise:
        add_dirichlet_noise(root)

    # Run simulations
    for _ in range(num_simulations):
        leaf = select_leaf(root)
        if not leaf.is_expanded:
            value = expand_node(leaf, model, device)
        else:
            # Terminal node reached again
            if leaf.board.is_checkmate():
                value = -1.0
            else:
                value = 0.0
        backpropagate(leaf, value)

    # Extract visit counts and build policy target
    policy_target = torch.zeros(POLICY_DIM, dtype=torch.float32)
    visit_counts = {}
    for move_idx, child in root.children.items():
        visit_counts[move_idx] = child.visit_count
        policy_target[move_idx] = child.visit_count

    # Select move based on temperature
    if temperature < 1e-3:
        # Greedy: pick move with most visits
        best_idx = max(visit_counts, key=visit_counts.get)
        move = root.children[best_idx].move
    else:
        # Sample proportional to visit_count^(1/temperature)
        indices = list(visit_counts.keys())
        counts = torch.tensor(
            [float(visit_counts[i]) for i in indices], dtype=torch.float32
        )
        counts = counts ** (1.0 / temperature)
        probs = counts / counts.sum()
        chosen = torch.multinomial(probs, 1).item()
        best_idx = indices[chosen]
        move = root.children[best_idx].move

    # Normalize policy target to a distribution
    total = policy_target.sum()
    if total > 0:
        policy_target = policy_target / total

    return move, policy_target
