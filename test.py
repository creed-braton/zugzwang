import os
import random
import struct
import uuid
from datetime import datetime

import chess
import torch

from zugzwang.encode import board_to_tensor
from zugzwang.net import Net

HISTORY_STEPS = 8
RES_BLOCKS = 6
CHANNELS = 128
BATCH_SIZE = 1024
INPUT_DIM = 14 * HISTORY_STEPS + 7

# ---------------------------------------------------------------------------
# 1. Init & save an empty network
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("onnx", exist_ok=True)

model_id = uuid.uuid4()
model = Net(INPUT_DIM, res_blocks=RES_BLOCKS, channels=CHANNELS)
model.eval()

save_path = os.path.join("models", f"{model_id}.pth")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "iteration": 0,
        "hyper_parameters": {
            "history_steps": HISTORY_STEPS,
            "res_blocks": RES_BLOCKS,
            "channels": CHANNELS,
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
    save_path,
)
print(f"Saved model to {save_path}")

# ---------------------------------------------------------------------------
# 2. Export to ONNX
# ---------------------------------------------------------------------------
onnx_path = os.path.join("onnx", f"{model_id}.onnx")
dummy_input = torch.randn(1, INPUT_DIM, 8, 8)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["board"],
    output_names=["policy", "value"],
    dynamic_axes={
        "board": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
    },
    opset_version=25,
)
print(f"Exported ONNX to {onnx_path}")

# ---------------------------------------------------------------------------
# 3. Generate 1024 board states as tensors
# ---------------------------------------------------------------------------
random.seed(42)
tensors = []

for _ in range(BATCH_SIZE):
    board = chess.Board()
    # play a random number of moves (0-80) to get diverse positions
    num_moves = random.randint(0, 80)
    for _ in range(num_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))
    tensors.append(board_to_tensor(board, history_steps=HISTORY_STEPS))

batch = torch.stack(tensors)  # (1024, 119, 8, 8)
assert batch.shape == (BATCH_SIZE, INPUT_DIM, 8, 8)

# ---------------------------------------------------------------------------
# 4. Store as binary: header + raw float32
#
# Format (all little-endian):
#   uint32  ndim
#   uint32  shape[0]  (batch_size)
#   uint32  shape[1]  (channels)
#   uint32  shape[2]  (height)
#   uint32  shape[3]  (width)
#   float32 data[...]  (contiguous, row-major)
# ---------------------------------------------------------------------------
bin_path = os.path.join("onnx", f"{model_id}.bin")
data = batch.contiguous().numpy()

with open(bin_path, "wb") as f:
    f.write(struct.pack("<I", len(data.shape)))
    for dim in data.shape:
        f.write(struct.pack("<I", dim))
    f.write(data.tobytes())

print(f"Saved {BATCH_SIZE} board tensors to {bin_path}")
print(f"  shape: {tuple(data.shape)}")
print(f"  dtype: float32")
print(f"  file size: {os.path.getsize(bin_path)} bytes")
print(f"\nModel ID: {model_id}")
