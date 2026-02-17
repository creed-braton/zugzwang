import argparse
import logging
import os
import sys

import torch

from zugzwang.net import Net

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Export a .pth checkpoint to ONNX"
)
parser.add_argument(
    "model_id",
    type=str,
    nargs="?",
    default=None,
    help="UUID of the model to export (omit to export the most recently updated model)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("zugzwang.export")

# ---------------------------------------------------------------------------
# Resolve checkpoint
# ---------------------------------------------------------------------------
models_dir = "models"

if args.model_id is None:
    best_path = None
    best_updated_at = ""
    for filename in os.listdir(models_dir):
        if not filename.endswith(".pth"):
            continue
        path = os.path.join(models_dir, filename)
        cp = torch.load(path, map_location="cpu")
        updated_at = cp.get("updated_at", "")
        if updated_at > best_updated_at:
            best_updated_at = updated_at
            best_path = path
    if best_path is None:
        logger.error("No .pth checkpoints found in %s/", models_dir)
        sys.exit(1)
    checkpoint_path = best_path
    model_id = os.path.basename(best_path).removesuffix(".pth")
else:
    checkpoint_path = os.path.join(models_dir, f"{args.model_id}.pth")
    model_id = args.model_id

logger.info("Checkpoint: %s", checkpoint_path)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
checkpoint = torch.load(checkpoint_path, map_location="cpu")
params = checkpoint["hyper_parameters"]

logger.info("Model hyper-parameters:")
for key, value in sorted(params.items()):
    logger.info("  %s = %s", key, value)
logger.info("Training iteration: %d", checkpoint["iteration"])

input_dim = 14 * params["history_steps"] + 7
model = Net(
    input_dim, res_blocks=params["res_blocks"], channels=params["channels"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ---------------------------------------------------------------------------
# Export to ONNX
# ---------------------------------------------------------------------------
os.makedirs("onnx", exist_ok=True)
onnx_path = os.path.join("onnx", f"{model_id}.onnx")

dummy_input = torch.randn(1, input_dim, 8, 8)

logger.info("Exporting to %s ...", onnx_path)
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
logger.info("Done — saved %s", onnx_path)
