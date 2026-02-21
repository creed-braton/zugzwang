#ifndef SEARCH_H
#define SEARCH_H

#include "chess.h"
#include "infer.h"
#include <stdatomic.h>

/* ── Node states ─────────────────────────────────────────────────── */

enum {
    NODE_UNEXPANDED = 0,
    NODE_EXPANDING  = 1,
    NODE_EXPANDED   = 2,
    NODE_TERMINAL   = 3
};

/* ── Search tree node ────────────────────────────────────────────── */

typedef struct Node {
    struct Node  *parent;
    struct Node  *children;       /* contiguous array, allocated once */
    uint16_t     *child_moves;    /* parallel array of encoded moves */
    int16_t       num_children;
    atomic_int    visit_count;    /* atomic for virtual loss */
    atomic_int    value_sum_fp;   /* fixed-point: value * 100000 */
    float         prior;          /* write-once during parent expansion */
    atomic_int    state;          /* NODE_UNEXPANDED -> EXPANDING -> EXPANDED | TERMINAL */
    float         terminal_value; /* write-once if terminal */
} Node;

/* ── Configuration ───────────────────────────────────────────────── */

typedef struct {
    int   num_simulations;    /* total MCTS sims per position (e.g. 400) */
    int   num_threads;        /* search threads per tree (e.g. 4) */
    int   concurrent_games;   /* games running in parallel (e.g. 64) */
    float c_puct;             /* exploration constant (default 1.41) */
    float temperature;        /* policy prior temperature */
    int   greedy_threshold;   /* move # after which to select greedily */
    float dirichlet_alpha;    /* Dirichlet noise alpha (0 = disabled) */
    float dirichlet_weight;   /* noise mixing fraction (e.g. 0.25) */
} SearchConfig;

/* ── Results ─────────────────────────────────────────────────────── */

typedef struct {
    uint16_t best_move;
    float    policy[NUM_MOVES];   /* normalized visit counts */
    float    root_value;
    int      total_visits;
} SearchResult;

/* ── Dataset recording ───────────────────────────────────────────── */

typedef struct {
    float *states;      /* [count * tensor_floats] encoded board tensors */
    float *policies;    /* [count * NUM_MOVES] visit count distributions */
    float *values;      /* [count] game outcomes */
    int    count;
    int    tensor_floats;
} GameRecord;

/* ── Public API ──────────────────────────────────────────────────── */

SearchConfig default_search_config(void);

SearchResult search(Board *board, InferenceBatcher *batcher,
                    const SearchConfig *config);

GameRecord *self_play_game(InferenceBatcher *batcher,
                           const SearchConfig *config);

int self_play(InferenceBatcher *batcher, const SearchConfig *config,
              int num_games, const char *output_path);

int  write_game_record(const GameRecord *record, const char *path);
void free_game_record(GameRecord *record);

#endif
