# Inference API — Requirements

## Goal

Replace the in-process `InferenceBatcher` with an inference API where:

- One **inference server** runs in its own OS process, pinned to a
  dedicated CPU core, and owns the model + device.
- Many **caller processes** run self-play in parallel on other cores.
  Each issues inference requests and awaits the result.
- The server batches requests across all callers (by max batch size or
  by timeout, whichever comes first) and runs a single forward pass per
  batch.

The current `InferenceBatcher` cannot scale past one process: its queue
is asyncio-only, and a single Python process cannot run search and
forward passes in parallel because of the GIL. This work item delivers
the multi-process replacement.

## Initialization

The API is constructed once in a parent process before any worker
starts. The parent owns the model and configuration. The API itself
brings up the server process; callers do not.

Required configuration:

- `model` — the network. The API is responsible for transferring
  weights to the server process at startup.
- `device` — where the forward pass runs (`cpu`, `cuda:0`, ...).
- `num_workers` — number of caller processes that will connect.
- `batch_size` — max requests per forward pass.
- `timeout` — max wall-clock time the server waits to fill a batch
  before flushing a partial batch (seconds, e.g. `0.005`).
- `history_steps` — used by callers to encode boards consistently with
  the model's expected input.
- `cpu_affinity` *(optional)* — core(s) to pin the server to. If unset,
  the OS schedules.

The API must produce a per-worker **client handle**. The handle is the
object passed into a spawned worker and used inside it to issue
requests. The handle exposes `history_steps` (read-only) so callers can
encode boards consistently — matches today's
`InferenceBatcher.history_steps`.

## Request / Response model

### Input — one request

A worker calls:

```python
policy, value = await client.infer(board)
```

- `board: chess.Board` — same call surface as today's
  `InferenceBatcher.infer`, so existing call sites of `infer(board)` in
  `search.py` and `dataset.py` need no change.
- Encoding (`board → tensor`) must happen in the **caller process** so
  that work parallelizes across cores. The server only runs the
  forward pass.
- The client may have many `infer` calls in flight at once; each must
  be tracked independently and resolved in any order.

### Output — one response

- `tuple[torch.Tensor, float]` — `(policy, value)`. `policy` is
  softmaxed and on CPU. `value` is a Python float in `[-1, 1]`.
- Each `await infer(...)` resolves to its own response. No mixing
  across concurrent calls, no head-of-line blocking inside one client.

### Errors

- A forward-pass exception on the server must surface as an exception
  on every caller whose request was in the affected batch. Fail loud:
  no hangs, no silent drops.
- During shutdown, any caller still awaiting `infer` must receive a
  clear exception rather than a pending future.

## Parallelism

- The server runs in its own OS process, not a thread, and should be
  pinned to a dedicated CPU core (`cpu_affinity`) so it is not
  descheduled in favor of workers.
- Each worker runs in its own OS process on a different core. Workers
  are independent; a slow or blocked worker must not stall another.
- Tensor data must move between processes via **shared memory**, not
  pickle-by-value. IPC cost must not dominate forward-pass cost.
- A single batch may mix requests from any subset of connected workers
  (cross-worker batching), not just requests from one worker.

## Lifecycle

- **Start**: bring up the server process, open the IPC channels, become
  ready to accept requests.
- **Shutdown(timeout)**: stop accepting new requests, drain in-flight
  work, terminate the server, unblock every caller future. If the
  server does not exit within `timeout`, force-terminate it.
- The API must expose counters for tqdm-style progress reporting
  (total inferences, total batches, start time). `dataset.py` reads
  these from the current `InferenceBatcher`; the new API must keep
  that observable.

## Acceptance criteria

- N worker processes issuing `await client.infer(board)` concurrently
  each receive the correct `(policy, value)` for their own board.
- Under load, server batches contain requests from more than one
  worker (proves cross-worker batching, not just per-worker batching).
- A forced server-side exception surfaces on every in-flight caller as
  a raised exception, not a hang.
- Shutdown returns within the configured timeout — no orphan
  processes, no pending futures.
- Throughput with N worker processes scales meaningfully past a single
  in-process `InferenceBatcher` run on the same host.

## Out of scope

- Spawning self-play workers; changes to `dataset.py` / `train.py` /
  `search.py` beyond what is needed to consume the new API.
- Cross-machine inference.
- Priority queues, CUDA streams, pinned memory, model hot-swap.
