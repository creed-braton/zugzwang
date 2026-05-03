import argparse
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import torch

from .net import Net, hash_model


@dataclass(frozen=True)
class Config:
    # Architecture
    res_blocks: int
    channels: int
    history_steps: int

    # Optimizer
    lr: float
    weight_decay: float

    # Self-play
    num_simulations: int
    temperature: float
    greedy_threshold: int
    dirichlet_alpha: float
    dirichlet_epsilon: float

    # Training
    num_iterations: int
    num_epochs: int
    batch_size: int
    window_size: int

    # MCTS
    c: float
    fpu_reduction: float

    # Inference
    inference_batch_size: int
    inference_timeout: float

    @property
    def input_dim(self) -> int:
        return 14 * self.history_steps + 7

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


class Model:
    def __init__(
        self,
        net: Net,
        config: Config,
        model_id: str,
        run_id: str,
        created_at: str,
        iteration: int = 0,
    ):
        self.net = net
        self.config = config
        self.model_id = model_id
        self.run_id = run_id
        self.created_at = created_at
        self.iteration = iteration

    @classmethod
    def init(cls, config: Config) -> "Model":
        net = Net(config.input_dim, config.res_blocks, config.channels)
        return cls(
            net=net,
            config=config,
            model_id=hash_model(net),
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        blob = torch.load(path, weights_only=False, map_location="cpu")
        config = Config(**blob["config"])
        net = Net(config.input_dim, config.res_blocks, config.channels)
        net.load_state_dict(blob["state_dict"])
        created_at = blob.get("created_at")
        if created_at is None:
            mtime = datetime.fromtimestamp(
                Path(path).stat().st_mtime, tz=timezone.utc
            )
            created_at = mtime.isoformat()
        return cls(
            net=net,
            config=config,
            model_id=blob["model_id"],
            run_id=blob["run_id"],
            created_at=created_at,
            iteration=blob["iteration"],
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": asdict(self.config),
                "model_id": self.model_id,
                "run_id": self.run_id,
                "iteration": self.iteration,
                "created_at": self.created_at,
            },
            path,
        )
