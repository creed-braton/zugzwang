import chess
from encode import board_to_tensor

board = chess.Board()
print(board_to_tensor(board))
