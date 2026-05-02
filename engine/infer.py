"""Multi-process inference API."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any

import chess
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from .encode import board_to_tensor
from .net import hash_model


class InferenceError(Exception):
    """Base class for inference errors."""


class InferenceServerError(InferenceError):
    """Forward-pass exception on the server, surfaced to callers."""


class InferenceShutdown(InferenceError):
    """Raised on pending callers when the API is shut down."""


# --- IPC message types -------------------------------------------------------


@dataclass
class _Register:
    client_id: int
    response_q: Any


@dataclass
class _Deregister:
    client_id: int


class _Shutdown:
    """Sentinel — terminates the server (on request_q) or a client reader
    thread (on response_q)."""


@dataclass
class _Request:
    client_id: int
    request_id: int
    tensor: torch.Tensor


@dataclass
class _Response:
    request_id: int
    policy: torch.Tensor | None
    value: float | None
    error: str | None


# --- Server-side helpers (top-level so they're picklable for spawn) ----------


def _set_affinity(cores: int | list[int]) -> None:
    if isinstance(cores, int):
        cores = [cores]
    os.sched_setaffinity(0, cores)


def _collect_batch(
    request_q: Any, batch_size: int, timeout: float
) -> list[Any]:
    """Block for first item, then drain up to batch_size with deadline."""
    items: list[Any] = [request_q.get()]
    deadline = time.monotonic() + timeout
    while len(items) < batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            items.append(request_q.get(timeout=remaining))
        except Empty:
            break
    return items


def _run_forward(
    model: nn.Module, device: torch.device, items: list[_Request]
) -> list[tuple[torch.Tensor, float]]:
    batch = torch.stack([item.tensor for item in items]).to(device)
    with torch.no_grad():
        policy_logits, value = model(batch)
    policy = torch.softmax(policy_logits, dim=1).cpu()
    value = value.squeeze(-1).cpu()
    return [(policy[i], float(value[i])) for i in range(len(items))]


def _dispatch_results(
    items: list[_Request],
    results: list[tuple[torch.Tensor, float]],
    registry: dict[int, Any],
) -> None:
    for item, (policy, value) in zip(items, results):
        q = registry.get(item.client_id)
        if q is None:
            continue
        q.put(_Response(item.request_id, policy, value, None))


def _dispatch_error(
    items: list[_Request], exc: BaseException, registry: dict[int, Any]
) -> None:
    msg = f"{type(exc).__name__}: {exc}"
    for item in items:
        q = registry.get(item.client_id)
        if q is None:
            continue
        q.put(_Response(item.request_id, None, None, msg))


def _broadcast_shutdown(registry: dict[int, Any]) -> None:
    for q in registry.values():
        q.put(_Shutdown())


def _server_main(
    model: nn.Module,
    device: str | torch.device,
    request_q: Any,
    counters: dict[str, Any],
    cpu_affinity: int | list[int] | None,
    batch_size: int,
    timeout: float,
) -> None:
    if cpu_affinity is not None:
        _set_affinity(cpu_affinity)
    device = torch.device(device)
    model = model.to(device).eval()
    counters["start_time"].value = time.time()

    registry: dict[int, Any] = {}
    shutdown_received = False

    while not shutdown_received:
        items = _collect_batch(request_q, batch_size, timeout)

        data_items: list[_Request] = []
        for item in items:
            if isinstance(item, _Shutdown):
                shutdown_received = True
            elif isinstance(item, _Register):
                registry[item.client_id] = item.response_q
            elif isinstance(item, _Deregister):
                registry.pop(item.client_id, None)
            else:
                data_items.append(item)

        if data_items:
            try:
                results = _run_forward(model, device, data_items)
                _dispatch_results(data_items, results, registry)
                with counters["total_inferences"].get_lock():
                    counters["total_inferences"].value += len(data_items)
                with counters["total_batches"].get_lock():
                    counters["total_batches"].value += 1
            except Exception as e:
                _dispatch_error(data_items, e, registry)

    _broadcast_shutdown(registry)


# --- Future helpers (called via call_soon_threadsafe from the reader) --------


def _set_future_result(fut: asyncio.Future, result: Any) -> None:
    if not fut.done():
        fut.set_result(result)


def _set_future_exception(fut: asyncio.Future, exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)


# --- Public API --------------------------------------------------------------


class InferenceClient:
    """Per-worker handle, picklable into a spawned worker process."""

    def __init__(
        self,
        request_q: Any,
        response_q: Any,
        client_id: int,
        history_steps: int,
        model_id: str,
    ):
        self._request_q = request_q
        self._response_q = response_q
        self._client_id = client_id
        self.history_steps = history_steps
        self.model_id = model_id
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[int, asyncio.Future] | None = None
        self._reader: threading.Thread | None = None
        self._lock: threading.Lock | None = None
        self._next_id = 0
        self._connected = False

    async def connect(self) -> None:
        """Register with the server. Call once before any infer()."""
        if self._connected:
            raise InferenceError("client already connected")
        self._loop = asyncio.get_running_loop()
        self._pending = {}
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._request_q.put(_Register(self._client_id, self._response_q))
        self._connected = True

    async def disconnect(self) -> None:
        """Unregister from the server, cancelling any in-flight requests."""
        if not self._connected:
            return
        self._connected = False
        self._request_q.put(_Deregister(self._client_id))
        self._response_q.put(_Shutdown())
        await asyncio.to_thread(self._reader.join)

    async def infer(self, board: chess.Board) -> tuple[torch.Tensor, float]:
        """Run inference for one board; returns (policy on CPU, value in [-1, 1])."""
        if not self._connected:
            raise InferenceError("client not connected; call connect() first")
        tensor = board_to_tensor(board, self.history_steps)
        request_id, future = self._send(tensor)
        try:
            return await future
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def _send(self, tensor: torch.Tensor) -> tuple[int, asyncio.Future]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            future = self._loop.create_future()
            self._pending[request_id] = future
        self._request_q.put(_Request(self._client_id, request_id, tensor))
        return request_id, future

    def _reader_loop(self) -> None:
        while True:
            msg = self._response_q.get()
            if isinstance(msg, _Shutdown):
                self._fail_all_pending(InferenceShutdown())
                return
            self._resolve(msg)

    def _resolve(self, msg: _Response) -> None:
        with self._lock:
            fut = self._pending.get(msg.request_id)
        if fut is None:
            return
        if msg.error is not None:
            self._loop.call_soon_threadsafe(
                _set_future_exception, fut, InferenceServerError(msg.error)
            )
        else:
            self._loop.call_soon_threadsafe(
                _set_future_result, fut, (msg.policy, msg.value)
            )

    def _fail_all_pending(self, exc: BaseException) -> None:
        with self._lock:
            futures = list(self._pending.values())
            self._pending.clear()
        for fut in futures:
            self._loop.call_soon_threadsafe(_set_future_exception, fut, exc)


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
        model_id: str | None = None,
    ):
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._timeout = timeout
        self._history_steps = history_steps
        self._cpu_affinity = cpu_affinity
        self.model_id = model_id if model_id is not None else hash_model(model)

        self._ctx = mp.get_context("spawn")
        self._request_q = self._ctx.Queue()
        self._counters = {
            "total_inferences": self._ctx.Value("q", 0),
            "total_batches": self._ctx.Value("q", 0),
            "start_time": self._ctx.Value("d", 0.0),
        }
        self._process: mp.Process | None = None
        self._next_client_id = 0

    def start(self) -> None:
        """Spawn the server process and open IPC channels."""
        if self._process is not None:
            raise InferenceError("server already started")
        self._process = self._ctx.Process(
            target=_server_main,
            args=(
                self._model,
                self._device,
                self._request_q,
                self._counters,
                self._cpu_affinity,
                self._batch_size,
                self._timeout,
            ),
            daemon=True,
        )
        self._process.start()

    def client(self) -> InferenceClient:
        """Mint a picklable handle for one worker."""
        client_id = self._next_client_id
        self._next_client_id += 1
        response_q = self._ctx.Queue()
        return InferenceClient(
            request_q=self._request_q,
            response_q=response_q,
            client_id=client_id,
            history_steps=self._history_steps,
            model_id=self.model_id,
        )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain, terminate the server, unblock pending callers."""
        if self._process is None:
            return
        if self._process.is_alive():
            self._request_q.put(_Shutdown())
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
        self._process = None

    @property
    def total_inferences(self) -> int:
        return self._counters["total_inferences"].value

    @property
    def total_batches(self) -> int:
        return self._counters["total_batches"].value

    @property
    def start_time(self) -> float | None:
        t = self._counters["start_time"].value
        return t if t > 0 else None

    def __enter__(self) -> InferenceServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
