#ifndef INFER_H
#define INFER_H

#include "chess.h"

typedef struct {
    float policy[NUM_MOVES];  /* 4168 softmaxed probabilities */
    float value;              /* tanh-activated scalar [-1, 1] */
} InferResult;

typedef struct InferenceBatcher InferenceBatcher;

InferenceBatcher *init_batcher(const char *model_path, int batch_size,
                               double timeout_sec, int history_steps,
                               int num_search_threads);
void free_batcher(InferenceBatcher *b);
int infer(InferenceBatcher *b, Board *board, InferResult *result);

void board_to_tensor(Board *board, int history_steps,
                     int tensor_planes, float *tensor);
int batcher_tensor_planes(const InferenceBatcher *b);
int batcher_history_steps(const InferenceBatcher *b);

#endif
