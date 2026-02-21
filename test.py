import os
import uuid
from datetime import datetime

import onnx
import torch

from zugzwang.net import Net

HISTORY_STEPS = 8
RES_BLOCKS = 6
CHANNELS = 128
INPUT_DIM = 14 * HISTORY_STEPS + 7

# ---------------------------------------------------------------------------
# 1. Init & save an empty network
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

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
onnx_path = os.path.join("models", f"{model_id}.onnx")
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

# Ensure all weights are embedded inside the .onnx file (no external data)
onnx_model = onnx.load(onnx_path)
onnx.save(onnx_model, onnx_path, save_as_external_data=False)

data_path = onnx_path + ".data"
if os.path.exists(data_path):
    os.remove(data_path)

print(f"Exported ONNX to {onnx_path}")
print(f"\nModel ID: {model_id}")
