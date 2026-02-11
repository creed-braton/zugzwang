import glob
import os

import chess
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from encode import board_to_tensor, sample_move
from model import ZugzwangNet

app = FastAPI()

# Load the most recent model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZugzwangNet().to(device)

checkpoints = sorted(glob.glob(os.path.join("models", "*.pth")))
if checkpoints:
    model.load_state_dict(torch.load(checkpoints[-1], map_location=device))
    print(f"Loaded checkpoint: {checkpoints[-1]}")
else:
    print("WARNING: No checkpoint found in models/, using random weights")
model.eval()


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str  # UCI string e.g. "e2e4"
    fen: str  # board FEN after the move


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    board = chess.Board(req.fen)
    state = board_to_tensor(board)
    with torch.no_grad():
        _, policy_logits, _ = model(state.unsqueeze(0).to(device))
    move, _, _ = sample_move(board, policy_logits.squeeze(0), temperature=0.1)
    board.push(move)
    return MoveResponse(move=move.uci(), fen=board.fen())


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Zugzwang</title>
<link rel="stylesheet"
      href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"/>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #e0e0e0;
    font-family: system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 2rem 1rem;
  }
  h1 { font-size: 1.6rem; margin-bottom: 0.3rem; color: #fff; }
  #subtitle { font-size: 0.9rem; color: #888; margin-bottom: 1.2rem; }
  #board-wrap { width: 480px; max-width: 95vw; }
  #status {
    margin-top: 1rem; font-size: 1rem; min-height: 1.5em;
    color: #ccc; text-align: center;
  }
  .btn-row { margin-top: 1rem; display: flex; gap: 0.6rem; }
  button {
    background: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    padding: 0.5rem 1.2rem; border-radius: 6px; cursor: pointer;
    font-size: 0.9rem; transition: background 0.15s;
  }
  button:hover { background: #0f3460; }
</style>
</head>
<body>
<h1>Zugzwang</h1>
<p id="subtitle">Play against the neural net</p>
<div id="board-wrap">
  <div id="board"></div>
</div>
<div id="status">Your move (white).</div>
<div class="btn-row">
  <button onclick="newGame()">New Game</button>
  <button onclick="flipBoard()">Flip Board</button>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script>
const statusEl = document.getElementById('status');
let game = new Chess();
let thinking = false;

function onDragStart(source, piece) {
  if (thinking) return false;
  if (game.game_over()) return false;
  if (game.turn() === 'b') return false;
  if (piece.search(/^b/) !== -1) return false;
}

function onDrop(source, target) {
  // Attempt the move (allow promotions to queen by default)
  const move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';
  updateStatus();

  if (!game.game_over()) {
    thinking = true;
    statusEl.textContent = 'Zugzwang is thinking...';
    fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen: game.fen() })
    })
    .then(r => r.json())
    .then(data => {
      game.move(data.move, { sloppy: true });
      board.position(game.fen());
      thinking = false;
      updateStatus();
    });
  }
}

function onSnapEnd() { board.position(game.fen()); }

function updateStatus() {
  if (game.in_checkmate()) {
    statusEl.textContent = game.turn() === 'w' ? 'Checkmate — black wins!' : 'Checkmate — white wins!';
  } else if (game.in_draw()) {
    statusEl.textContent = 'Draw.';
  } else if (game.in_check()) {
    statusEl.textContent = (game.turn() === 'w' ? 'White' : 'Black') + ' is in check.';
  } else if (game.turn() === 'w') {
    statusEl.textContent = 'Your move.';
  }
}

function newGame() {
  game = new Chess();
  board.start();
  thinking = false;
  statusEl.textContent = 'Your move (white).';
}

function flipBoard() { board.flip(); }

const board = Chessboard('board', {
  draggable: true,
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
