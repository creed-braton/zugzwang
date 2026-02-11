import chess
import torch


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(13, 8, 8, dtype=torch.float32)
    # flip when its not white's move to encode pawn direction uniformly
    vertical_flip = not board.turn
    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        row = 7 - row if vertical_flip else row
        piece = board.piece_at(square)
        if piece is None:
            continue
        # XOR on the color when we flip
        color = piece.color ^ vertical_flip
        # each plane encodes a piece so the opponents encoding starts at 6
        plane_offset = 6 if not color else 0
        piece = piece.symbol().lower()
        plane = "pnbrqk".index(piece) + plane_offset
        tensor[plane, row, col] = 1.0

    if board.has_queenside_castling_rights(board.turn):
        tensor[12, 0, 0] = 1.0
    if board.has_kingside_castling_rights(board.turn):
        tensor[12, 0, 7] = 1.0
    if board.has_queenside_castling_rights(vertical_flip):
        tensor[12, 7, 0] = 1.0
    if board.has_kingside_castling_rights(vertical_flip):
        tensor[12, 7, 7] = 1.0
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        row = 7 - row if vertical_flip else row
        tensor[12, row, col] = 1.0

    return tensor


_UNDERPROM_PCS_DIM = 3
_FILE_DIM = 8
_PAWN_DIR_DIM = 3
_SQUARE_DIM = 64 * 64
_UNDERPROM_DIM = _UNDERPROM_PCS_DIM * _FILE_DIM * _PAWN_DIR_DIM
POLICY_DIM = _SQUARE_DIM + _UNDERPROM_DIM


def _get_move_index(move: chess.Move, flip: bool) -> int:
    row, col = divmod(move.from_square, 8)
    row = 7 - row if flip else row
    from_square = row * 8 + col
    row, col = divmod(move.to_square, 8)
    row = 7 - row if flip else row
    to_square = row * 8 + col

    if move.promotion and move.promotion != chess.QUEEN:
        piece_index = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(
            move.promotion
        )
        from_file = chess.square_file(from_square)
        direction = chess.square_file(to_square) - from_file + 1
        return (
            _SQUARE_DIM
            + piece_index * _FILE_DIM * _PAWN_DIR_DIM
            + from_file * _PAWN_DIR_DIM
            + direction
        )

    return from_square * 64 + to_square


def _get_legal_moves(
    board: chess.Board,
) -> tuple[list[chess.Move], torch.Tensor]:
    """Return legal moves and a tensor of their policy indices."""
    flip = not board.turn
    moves = list(board.legal_moves)
    indices = torch.tensor(
        [_get_move_index(m, flip) for m in moves], dtype=torch.long
    )
    return moves, indices


def sample_move(
    board: chess.Board,
    logits: torch.Tensor,
    temperature: float = 1.0,
    model: torch.nn.Module | None = None,
    alpha: float = 3.0,
) -> tuple[chess.Move, float, torch.Tensor]:
    moves, indices = _get_legal_moves(board)
    legal_logits = logits[indices]

    if model is not None and len(moves) > 1:
        device = logits.device
        # For each legal move, play it and evaluate the resulting position
        states = []
        for move in moves:
            board.push(move)
            states.append(board_to_tensor(board))
            board.pop()
        batch = torch.stack(states).to(device)
        _, _, values = model(batch)
        # values are from the opponent's perspective; negate so that
        # moves leading to bad positions for the opponent score high
        legal_logits = legal_logits + alpha * (-values.squeeze())

    legal_logits = legal_logits / temperature
    log_probs = legal_logits - legal_logits.logsumexp(dim=0)
    probs = log_probs.exp()

    chosen = torch.multinomial(probs, 1).item()
    move = moves[chosen]

    return move, log_probs[chosen].item(), indices[chosen].item()
