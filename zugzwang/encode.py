import chess
import torch


def board_to_tensor(board: chess.Board, history_steps=8) -> torch.Tensor:
    tensor = torch.zeros(14 * history_steps + 7, 8, 8, dtype=torch.float32)
    # flip when its not white's move to encode pawn direction uniformly
    vertical_flip = not board.turn
    removed_moves = []
    for t in range(history_steps):
        offset = t * 14

        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            row = 7 - row if vertical_flip else row
            piece = board.piece_at(square)
            if piece is None:
                continue
            color = piece.color ^ vertical_flip
            plane_offset = 6 if not color else 0
            p = piece.symbol().lower()
            plane = "pnbrqk".index(p) + plane_offset
            tensor[offset + plane, row, col] = 1.0

        if board.is_repetition(2):
            tensor[offset + 12] = 1.0
        if board.is_repetition(3):
            tensor[offset + 13] = 1.0

        if board.move_stack:
            removed_moves.append(board.pop())
        else:
            break

    for move in reversed(removed_moves):
        board.push(move)

    offset = 14 * history_steps

    if board.has_queenside_castling_rights(board.turn):
        tensor[offset] = 1.0
    if board.has_kingside_castling_rights(board.turn):
        tensor[offset + 1] = 1.0
    if board.has_queenside_castling_rights(not board.turn):
        tensor[offset + 2] = 1.0
    if board.has_kingside_castling_rights(not board.turn):
        tensor[offset + 3] = 1.0

    tensor[offset + 4] = board.halfmove_clock
    tensor[offset + 5] = board.fullmove_number

    return tensor


_UNDERPROM_PCS_DIM = 3  # promoting to knight, bishop, rook needs to be encoded in separate state
_FILE_DIM = 8  # amount of column positions underpromotions can happen from
_PAWN_DIR_DIM = 3  # pawn has 3 directions, left take, right take, forward
_UNDERPROM_DIM = _UNDERPROM_PCS_DIM * _FILE_DIM * _PAWN_DIR_DIM
_SQUARE_DIM = 64 * 64
POLICY_DIM = _SQUARE_DIM + _UNDERPROM_DIM


def _move_to_index(move: chess.Move, flip: bool) -> int:
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


def legal_moves(
    board: chess.Board,
) -> tuple[list[chess.Move], torch.Tensor]:
    """Return legal moves and a tensor of their policy indices."""
    flip = not board.turn
    moves = list(board.legal_moves)
    indices = torch.tensor(
        [_move_to_index(m, flip) for m in moves], dtype=torch.long
    )
    return moves, indices
