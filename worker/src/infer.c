#define _POSIX_C_SOURCE 200809L

#include "infer.h"

#include <onnxruntime_c_api.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_HISTORY_STEPS 16
#define MAX_TENSOR_FLOATS ((14 * MAX_HISTORY_STEPS + 7) * 64)

#define ORT_CHECK(api, expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        fprintf(stderr, "ORT: %s\n", (api)->GetErrorMessage(_s)); \
        (api)->ReleaseStatus(_s); \
        goto cleanup; \
    } \
} while(0)

/* ── Private Types ────────────────────────────────────────────────── */

typedef struct {
    float           tensor[MAX_TENSOR_FLOATS];
    InferResult     result;
    int             ready;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} InferRequest;

struct InferenceBatcher {
    const OrtApi   *api;
    OrtEnv         *env;
    OrtSession     *session;
    OrtMemoryInfo  *memory_info;

    int             batch_size;
    int             history_steps;
    int             tensor_planes;
    double          timeout_sec;

    InferRequest  **queue;
    int             queue_count;
    pthread_mutex_t queue_mutex;
    pthread_cond_t  queue_cond;

    pthread_t       worker;
    int             shutdown;
};

/* ── Softmax ──────────────────────────────────────────────────────── */

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv_sum;
}

/* ── Board Encoding ───────────────────────────────────────────────── */

static void board_to_tensor(Board *board, int history_steps,
                            int tensor_planes, float *tensor) {
    memset(tensor, 0, (size_t)tensor_planes * 64 * sizeof(float));

    int vertical_flip = (board->turn == 1);
    int undone = 0;

    for (int t = 0; t < history_steps; t++) {
        int offset = t * 14;

        /* Piece planes */
        for (int sq = 0; sq < 64; sq++) {
            int piece = board->squares[sq];
            if (piece == EMPTY) continue;

            int row = sq / 8;
            int col = sq % 8;
            if (vertical_flip) row = 7 - row;

            int is_white = (piece < 6);
            int color = is_white ^ vertical_flip;
            int plane_offset = color ? 0 : 6;
            int piece_type = piece % 6;

            tensor[(offset + plane_offset + piece_type) * 64
                   + row * 8 + col] = 1.0f;
        }

        /* Repetition planes */
        int reps = repetitions(board);
        if (reps >= 1)
            for (int i = 0; i < 64; i++)
                tensor[(offset + 12) * 64 + i] = 1.0f;
        if (reps >= 2)
            for (int i = 0; i < 64; i++)
                tensor[(offset + 13) * 64 + i] = 1.0f;

        /* Walk history back */
        if (board->ply > 0) {
            undo_move(board);
            undone++;
        } else {
            break;
        }
    }

    /* Restore board */
    for (int i = 0; i < undone; i++)
        make_move(board, board->history[board->ply].move);

    /* Auxiliary planes */
    int aux = 14 * history_steps;

    int self_qs, self_ks, opp_qs, opp_ks;
    if (board->turn == 0) {
        self_qs = (board->castling >> 1) & 1;  /* WQ */
        self_ks = (board->castling >> 0) & 1;  /* WK */
        opp_qs  = (board->castling >> 3) & 1;  /* BQ */
        opp_ks  = (board->castling >> 2) & 1;  /* BK */
    } else {
        self_qs = (board->castling >> 3) & 1;  /* BQ */
        self_ks = (board->castling >> 2) & 1;  /* BK */
        opp_qs  = (board->castling >> 1) & 1;  /* WQ */
        opp_ks  = (board->castling >> 0) & 1;  /* WK */
    }

    if (self_qs)
        for (int i = 0; i < 64; i++)
            tensor[(aux + 0) * 64 + i] = 1.0f;
    if (self_ks)
        for (int i = 0; i < 64; i++)
            tensor[(aux + 1) * 64 + i] = 1.0f;
    if (opp_qs)
        for (int i = 0; i < 64; i++)
            tensor[(aux + 2) * 64 + i] = 1.0f;
    if (opp_ks)
        for (int i = 0; i < 64; i++)
            tensor[(aux + 3) * 64 + i] = 1.0f;

    float hm = (float)board->halfmove;
    float fm = (float)board->fullmove;
    for (int i = 0; i < 64; i++)
        tensor[(aux + 4) * 64 + i] = hm;
    for (int i = 0; i < 64; i++)
        tensor[(aux + 5) * 64 + i] = fm;

    /* Plane aux+6 stays 0 (unused) */
}

/* ── Worker Thread ────────────────────────────────────────────────── */

static void *worker_loop(void *arg) {
    InferenceBatcher *b = (InferenceBatcher *)arg;
    InferRequest **batch = malloc((size_t)b->batch_size * sizeof(InferRequest *));
    int tensor_floats = b->tensor_planes * 64;

    while (1) {
        int batch_count = 0;

        /* Wait for at least one request */
        pthread_mutex_lock(&b->queue_mutex);
        while (b->queue_count == 0 && !b->shutdown)
            pthread_cond_wait(&b->queue_cond, &b->queue_mutex);

        if (b->shutdown && b->queue_count == 0) {
            pthread_mutex_unlock(&b->queue_mutex);
            break;
        }

        /* Drain all available requests */
        batch_count = b->queue_count;
        if (batch_count > b->batch_size)
            batch_count = b->batch_size;
        memcpy(batch, b->queue,
               (size_t)batch_count * sizeof(InferRequest *));
        int remaining = b->queue_count - batch_count;
        if (remaining > 0)
            memmove(b->queue, b->queue + batch_count,
                    (size_t)remaining * sizeof(InferRequest *));
        b->queue_count = remaining;

        /* Wait for more if batch not full */
        if (batch_count < b->batch_size) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            long sec  = (long)b->timeout_sec;
            long nsec = (long)((b->timeout_sec - (double)sec) * 1e9);
            deadline.tv_sec  += sec;
            deadline.tv_nsec += nsec;
            if (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec++;
                deadline.tv_nsec -= 1000000000L;
            }

            while (batch_count < b->batch_size && !b->shutdown) {
                int rc = pthread_cond_timedwait(&b->queue_cond,
                                                &b->queue_mutex,
                                                &deadline);
                /* Drain new arrivals */
                int take = b->queue_count;
                if (take > b->batch_size - batch_count)
                    take = b->batch_size - batch_count;
                if (take > 0) {
                    memcpy(batch + batch_count, b->queue,
                           (size_t)take * sizeof(InferRequest *));
                    batch_count += take;
                    int rest = b->queue_count - take;
                    if (rest > 0)
                        memmove(b->queue, b->queue + take,
                                (size_t)rest * sizeof(InferRequest *));
                    b->queue_count = rest;
                }
                if (rc == ETIMEDOUT) break;
            }
        }

        pthread_mutex_unlock(&b->queue_mutex);

        if (batch_count == 0) continue;

        /* Build contiguous input buffer */
        size_t tensor_bytes = (size_t)tensor_floats * sizeof(float);
        float *input_data = malloc((size_t)batch_count * tensor_bytes);
        for (int i = 0; i < batch_count; i++)
            memcpy(input_data + i * tensor_floats,
                   batch[i]->tensor, tensor_bytes);

        /* ONNX inference */
        int ort_ok = 0;
        OrtValue *input_tensor = NULL;
        OrtValue *outputs[2] = {NULL, NULL};

        int64_t shape[] = {batch_count, b->tensor_planes, 8, 8};
        OrtStatus *status = b->api->CreateTensorWithDataAsOrtValue(
            b->memory_info, input_data,
            (size_t)batch_count * tensor_bytes,
            shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor);

        if (status) {
            fprintf(stderr, "ORT: %s\n", b->api->GetErrorMessage(status));
            b->api->ReleaseStatus(status);
        } else {
            const char *input_names[]  = {"board"};
            const char *output_names[] = {"policy", "value"};
            status = b->api->Run(
                b->session, NULL,
                input_names,
                (const OrtValue *const *)&input_tensor, 1,
                output_names, 2, outputs);
            if (status) {
                fprintf(stderr, "ORT: %s\n",
                        b->api->GetErrorMessage(status));
                b->api->ReleaseStatus(status);
            } else {
                ort_ok = 1;
            }
        }

        /* Extract results */
        if (ort_ok) {
            float *policy_data = NULL;
            float *value_data  = NULL;
            b->api->GetTensorMutableData(outputs[0],
                                         (void **)&policy_data);
            b->api->GetTensorMutableData(outputs[1],
                                         (void **)&value_data);

            for (int i = 0; i < batch_count; i++) {
                softmax(policy_data + i * NUM_MOVES, NUM_MOVES);
                memcpy(batch[i]->result.policy,
                       policy_data + i * NUM_MOVES,
                       NUM_MOVES * sizeof(float));
                batch[i]->result.value = value_data[i];
            }
        }

        /* Signal all requests as done */
        for (int i = 0; i < batch_count; i++) {
            pthread_mutex_lock(&batch[i]->mutex);
            batch[i]->ready = 1;
            pthread_cond_signal(&batch[i]->cond);
            pthread_mutex_unlock(&batch[i]->mutex);
        }

        /* Cleanup */
        if (outputs[0]) b->api->ReleaseValue(outputs[0]);
        if (outputs[1]) b->api->ReleaseValue(outputs[1]);
        if (input_tensor) b->api->ReleaseValue(input_tensor);
        free(input_data);
    }

    free(batch);
    return NULL;
}

/* ── Public API ───────────────────────────────────────────────────── */

InferenceBatcher *init_batcher(const char *model_path, int batch_size,
                               double timeout_sec, int history_steps) {
    if (history_steps > MAX_HISTORY_STEPS) {
        fprintf(stderr, "infer: history_steps %d exceeds max %d\n",
                history_steps, MAX_HISTORY_STEPS);
        return NULL;
    }

    InferenceBatcher *b = calloc(1, sizeof(InferenceBatcher));
    if (!b) return NULL;

    b->batch_size    = batch_size;
    b->timeout_sec   = timeout_sec;
    b->history_steps = history_steps;
    b->tensor_planes = 14 * history_steps + 7;

    /* ONNX Runtime setup */
    b->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!b->api) {
        fprintf(stderr, "infer: failed to get ORT API\n");
        free(b);
        return NULL;
    }

    OrtSessionOptions *opts = NULL;

    ORT_CHECK(b->api,
              b->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                "zugzwang", &b->env));
    ORT_CHECK(b->api,
              b->api->CreateSessionOptions(&opts));
    ORT_CHECK(b->api,
              b->api->CreateSession(b->env, model_path,
                                    opts, &b->session));
    b->api->ReleaseSessionOptions(opts);
    opts = NULL;

    ORT_CHECK(b->api,
              b->api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                          OrtMemTypeDefault,
                                          &b->memory_info));

    /* Queue (capacity 2x batch_size for safety) */
    b->queue = calloc((size_t)batch_size * 2, sizeof(InferRequest *));
    if (!b->queue) goto cleanup;

    pthread_mutex_init(&b->queue_mutex, NULL);
    pthread_cond_init(&b->queue_cond, NULL);

    /* Start worker thread */
    if (pthread_create(&b->worker, NULL, worker_loop, b) != 0)
        goto cleanup;

    return b;

cleanup:
    if (opts) b->api->ReleaseSessionOptions(opts);
    if (b->memory_info) b->api->ReleaseMemoryInfo(b->memory_info);
    if (b->session) b->api->ReleaseSession(b->session);
    if (b->env) b->api->ReleaseEnv(b->env);
    free(b->queue);
    free(b);
    return NULL;
}

void free_batcher(InferenceBatcher *b) {
    if (!b) return;

    pthread_mutex_lock(&b->queue_mutex);
    b->shutdown = 1;
    pthread_cond_signal(&b->queue_cond);
    pthread_mutex_unlock(&b->queue_mutex);

    pthread_join(b->worker, NULL);

    pthread_mutex_destroy(&b->queue_mutex);
    pthread_cond_destroy(&b->queue_cond);

    b->api->ReleaseMemoryInfo(b->memory_info);
    b->api->ReleaseSession(b->session);
    b->api->ReleaseEnv(b->env);

    free(b->queue);
    free(b);
}

int infer(InferenceBatcher *b, Board *board, InferResult *result) {
    InferRequest req;
    memset(&req.result, 0, sizeof(req.result));
    req.ready = 0;
    pthread_mutex_init(&req.mutex, NULL);
    pthread_cond_init(&req.cond, NULL);

    board_to_tensor(board, b->history_steps, b->tensor_planes, req.tensor);

    /* Queue request */
    pthread_mutex_lock(&b->queue_mutex);
    b->queue[b->queue_count++] = &req;
    pthread_cond_signal(&b->queue_cond);
    pthread_mutex_unlock(&b->queue_mutex);

    /* Wait for result */
    pthread_mutex_lock(&req.mutex);
    while (!req.ready)
        pthread_cond_wait(&req.cond, &req.mutex);
    pthread_mutex_unlock(&req.mutex);

    *result = req.result;

    pthread_mutex_destroy(&req.mutex);
    pthread_cond_destroy(&req.cond);

    return 0;
}
