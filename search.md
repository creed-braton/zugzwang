# Search Implementation Plan (C, Thread-Safe MCTS)

## Overview

Migrate the Python MCTS (`zugzwang/search.py`, `zugzwang/dataset.py`, `zugzwang/infer.py`) to C with
thread-safe tree expansion so that multiple worker threads can run simulations concurrently on a shared
search tree. The ONNX Runtime C API is already linked in the build.

---

## 1. Structs

### 1.1 `Node`

The fundamental tree building block. Every node corresponds to a board state reached by a specific
move from its parent.

```c
typedef struct Node {
    // --- tree pointers ---
    struct Node *parent;
    struct Node *children;      // heap-allocated array of children
    uint16_t    *moves;         // parallel array: moves[i] led to children[i]
    uint16_t     num_children;  // length of children/moves arrays (0 = leaf)

    // --- MCTS statistics ---
    _Atomic int32_t  visit_count; // N(s,a)  – atomically incremented
    _Atomic double   value_sum;   // W(s,a)  – atomically updated (virtual loss pattern)
    float            prior;       // P(s,a)  – written once at expansion, read-only after

    // --- synchronisation ---
    pthread_mutex_t  expand_mutex; // held during expansion to prevent duplicate work
    _Atomic uint8_t  expanded;     // 0 = leaf, 1 = expanded (acts as a fast lock-free check)
} Node;
```

**Why atomics for visit_count / value_sum?**
During backpropagation every thread walks its own path up the tree and increments these counters.
Using `_Atomic` (C11 `<stdatomic.h>`) avoids the need for a mutex on the hot path. For `value_sum`
we use `atomic_fetch_add` on a double (or cast to int64 fixed-point if the platform lacks lock-free
double atomics — see section 5).

**Why a separate expand_mutex?**
Two threads can arrive at the same leaf simultaneously. The mutex serialises expansion so only
one thread actually calls the network and writes the children array; the other thread waits and
then proceeds with selection on the freshly-expanded node.

### 1.2 `NodePool` (arena allocator)

Allocating nodes individually with `malloc` inside tight MCTS loops is slow and fragments memory.
A pool pre-allocates a large block and hands out nodes bump-pointer style.

```c
typedef struct {
    Node   *nodes;          // pre-allocated flat array
    size_t  capacity;       // total slots
    _Atomic size_t next;    // bump pointer (atomic for thread-safe allocation)
} NodePool;
```

- `node_pool_init(pool, capacity)` — `calloc` the array, init capacity.
- `node_pool_alloc(pool, count)` — atomically advance `next` by `count`, return pointer to first.
- `node_pool_reset(pool)` — set `next = 0` between searches (reuse memory).
- `node_pool_destroy(pool)` — free the array.

Pre-size based on `num_simulations * average_branching_factor` (e.g. 35) with some headroom.

### 1.3 `InferenceRequest` / `InferenceBatch`

To batch neural-network evaluations from multiple threads we need a request queue.

```c
typedef struct {
    float      *input;       // encoded board tensor (input_dim * 64 floats)
    float      *policy_out;  // points into caller-owned buffer (NUM_MOVES floats)
    float      *value_out;   // points into caller-owned buffer (1 float)
    pthread_mutex_t   mutex; // per-request signal
    pthread_cond_t    cond;  // per-request signal
    _Atomic uint8_t   ready; // set to 1 when result is written
} InferenceRequest;

typedef struct {
    InferenceRequest **queue;    // ring buffer of pending requests
    size_t             capacity;
    _Atomic size_t     head;     // producer write index
    _Atomic size_t     tail;     // consumer read index

    pthread_mutex_t    mutex;    // guards batch collection
    pthread_cond_t     cond;     // wakes consumer when work arrives

    size_t             batch_size; // max batch before dispatch
    int                timeout_us; // max wait before dispatching partial batch
} InferenceBatch;
```

### 1.4 `OnnxModel`

Wraps the ONNX Runtime session and its allocator so inference code has a clean handle.

```c
typedef struct {
    OrtEnv            *env;
    OrtSession        *session;
    OrtSessionOptions *options;
    OrtMemoryInfo     *mem_info;

    size_t input_dim;       // number of input planes (14 * history_steps + 7)
    size_t policy_dim;      // NUM_MOVES (4168)
} OnnxModel;
```

- `onnx_model_load(model, path, input_dim)` — create env/session, set CPU provider.
- `onnx_model_run_batch(model, inputs, batch_size, policy_out, value_out)` — single batched call.
- `onnx_model_destroy(model)` — release all ORT resources.

### 1.5 `SearchConfig`

All tunable knobs in one place.

```c
typedef struct {
    int     num_simulations;   // simulations per root move (e.g. 400 / 800)
    float   c_puct;            // UCB exploration constant (default 1.41)
    float   temperature;       // prior temperature during expansion
    int     num_threads;       // worker threads for simulation
    size_t  batch_size;        // max NN batch size
    int     batch_timeout_us;  // microseconds to wait for a full batch
    int     history_steps;     // history depth for encoding (default 8)
    int     greedy_threshold;  // move number after which to pick greedily
} SearchConfig;
```

### 1.6 `Search`

Top-level handle that owns the tree and coordinates workers.

```c
typedef struct {
    Node         *root;
    NodePool      pool;
    Board         root_board;     // copy of the position at the root
    SearchConfig  config;

    InferenceBatch batch;
    OnnxModel      model;

    pthread_t     *threads;       // worker thread handles
    pthread_t      batch_thread;  // inference consumer thread
    _Atomic int    running;       // flag to stop workers
} Search;
```

---

## 2. Board Encoding

Port `encode.py:board_to_tensor()` and `encode.py:_move_to_index()` to C.

```c
// Fills `out` with (input_dim * 64) floats representing the board for the NN.
void encode_board(const Board *b, float *out, int history_steps);

// Maps a move to a policy index [0, NUM_MOVES).
// `flip` is true when it is black's turn (mirrors ranks).
int move_to_policy_index(uint16_t move, int flip);
```

These are pure functions with no shared state, so they are inherently thread-safe.

---

## 3. Thread Architecture

```
┌──────────────────────────────────────────────────┐
│                    main thread                    │
│  search_run(search, board) → best move           │
└────────┬────────────────────────────┬────────────┘
         │ spawns                     │ spawns
    ┌────▼──────┐  ...  ┌────────────▼───┐
    │ worker 0  │       │  batch thread   │
    │ worker 1  │       │  (inference     │
    │ worker …  │       │   consumer)     │
    │ worker N  │       │                 │
    └────┬──────┘       └────────┬───────┘
         │  submit                │  collect & run
         │  InferenceRequest      │  ORT batch
         └───────────────────────►│
              InferenceBatch
```

### Worker thread loop (one per thread)

```
while simulations_remaining > 0:
    1. SELECT  – walk tree from root using UCB (read atomics)
    2. EXPAND  – lock leaf expand_mutex, check `expanded` flag
                 if already expanded → retry selection from that node
                 encode board, submit InferenceRequest, block on cond
    3. BACKPROP – walk back to root:
                  atomic_fetch_add(&node->visit_count, 1)
                  atomic_fetch_add(&node->value_sum, value)  (see §5)
                  value = -value
```

### Batch consumer thread

```
while running:
    1. wait on InferenceBatch cond (with timeout)
    2. collect up to batch_size requests
    3. stack inputs into contiguous buffer
    4. onnx_model_run_batch(...)
    5. scatter policy/value back into each request buffer
    6. signal each request's cond → unblocks the worker
```

---

## 4. Virtual Loss

To prevent all threads from exploring the same path, apply a **virtual loss** during selection:

- When a thread descends through a node during SELECT, it does `atomic_fetch_add(&node->visit_count, 1)`
  and `atomic_fetch_add(&node->value_sum, -1.0)` (pessimistic).
- After evaluation in BACKPROP, it corrects: adds `(value + 1.0)` to value_sum (undoes the -1.0 and adds the real value).

This naturally spreads threads across different branches without any extra locking.

---

## 5. Atomic Double Workaround

C11 `_Atomic double` may not be lock-free on all platforms. Two options:

**Option A — Fixed-point int64:**
Store `value_sum` as `_Atomic int64_t` scaled by e.g. 1e6. This is guaranteed lock-free on
x86-64 and aarch64. Conversion: `(double)value_sum / 1e6`.

**Option B — `__atomic_fetch_add` (GCC/Clang):**
GCC and Clang support `__atomic_fetch_add` for doubles via `__ATOMIC_RELAXED`. Non-portable but
works on the target (Linux x86-64).

Recommend **Option A** for portability — values are bounded [-1, +1] per visit so precision is fine.

---

## 6. Public API

```c
// Lifecycle
void search_init(Search *s, const char *model_path, SearchConfig config);
void search_destroy(Search *s);

// Run a full MCTS from the given position, returns best move.
uint16_t search_best_move(Search *s, const Board *board);

// Access root policy (normalised visit counts) after search — needed for self-play targets.
void search_root_policy(const Search *s, float policy_out[NUM_MOVES]);
```

---

## 7. File Layout

| File | Contents |
|------|----------|
| `search.h` | All struct definitions, public API declarations |
| `search.c` | Node operations, MCTS select/expand/backprop, worker threads |
| `encode.h` | `encode_board()`, `move_to_policy_index()` declarations |
| `encode.c` | Board-to-tensor encoding, move-to-index mapping |
| `onnx.h`   | `OnnxModel` struct, load/run/destroy declarations |
| `onnx.c`   | ONNX Runtime C API wrapper |

---

## 8. Synchronisation Summary

| Resource | Mechanism | Why |
|----------|-----------|-----|
| `visit_count` | `_Atomic int32_t` | many threads read/write on every simulation, must be fast |
| `value_sum` | `_Atomic int64_t` (fixed-point) | same hot path as visit_count |
| `prior` | plain `float` | written once before node is visible, read-only after |
| Node expansion | `pthread_mutex_t` per node | exactly one thread may expand a leaf |
| `expanded` flag | `_Atomic uint8_t` | fast lockless check to skip the mutex when already expanded |
| `NodePool.next` | `_Atomic size_t` | bump allocator, lock-free allocation for workers |
| Inference queue | `pthread_mutex_t` + `pthread_cond_t` | producer/consumer batching |
| Per-request signal | `pthread_mutex_t` + `pthread_cond_t` | worker blocks until its result is ready |
| `Search.running` | `_Atomic int` | stop signal for all threads |

---

## 9. Dependencies

- `<stdatomic.h>` (C11)
- `<pthread.h>` (POSIX threads)
- ONNX Runtime C API (`onnxruntime_c_api.h`, already in `worker/lib/onnx/include`)
- Existing `chess.h` / `chess.c` (Board, legal_moves, make_move, undo_move)
