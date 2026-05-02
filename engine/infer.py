"""Multi-process inference API."""

from __future__ import annotations

import chess
import torch
import torch.nn as nn


class InferenceError(Exception):
    """Base class for inference errors."""


class InferenceServerError(InferenceError):
    """Forward-pass exception on the server, surfaced to callers."""


class InferenceShutdown(InferenceError):
    """Raised on pending callers when the API is shut down."""


class InferenceClient:
    """Per-worker handle, picklable into a spawned worker process."""

    async def connect(self) -> None:
        """Register with the server. Call once before any infer()."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Unregister from the server, cancelling any in-flight requests."""
        raise NotImplementedError

    async def infer(self, board: chess.Board) -> tuple[torch.Tensor, float]:
        """Run inference for one board; returns (policy on CPU, value in [-1, 1])."""
        raise NotImplementedError


class InferenceServer:
    """Multi-process inference server; constructed once in the parent."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        batch_size: int = 64,
        timeout: float = 0.005,
        history_steps: int = 8,
        cpu_affinity: int | list[int] | None = None,
    ):
        raise NotImplementedError

    def start(self) -> None:
        """Spawn the server process and open IPC channels."""
        raise NotImplementedError

    def client(self) -> InferenceClient:
        """Mint a picklable handle for one worker."""
        raise NotImplementedError

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain, terminate the server, unblock pending callers."""
        raise NotImplementedError

    @property
    def total_inferences(self) -> int:
        raise NotImplementedError

    @property
    def total_batches(self) -> int:
        raise NotImplementedError

    @property
    def start_time(self) -> float | None:
        raise NotImplementedError

    def __enter__(self) -> InferenceServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
