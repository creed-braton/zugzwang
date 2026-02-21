#define _POSIX_C_SOURCE 200809L

#include "search.h"
#include "rng.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FP_SCALE 100000

/* ── Gamma / Dirichlet sampling ──────────────────────────────────── */

/* Marsaglia-Tsang method for Gamma(alpha, 1) when alpha >= 1 */
static float gamma_sample_ge1(Rng *rng, float alpha) {
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);

    for (;;) {
        float x, v;
        do {
            /* Box-Muller for standard normal */
            float u1 = rng_next_float(rng);
            float u2 = rng_next_float(rng);
            if (u1 < 1e-30f) u1 = 1e-30f;
            x = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
            v = 1.0f + c * x;
        } while (v <= 0.0f);

        v = v * v * v;
        float u = rng_next_float(rng);
        if (u < 1.0f - 0.0331f * (x * x) * (x * x))
            return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
            return d * v;
    }
}

/* Gamma(alpha, 1) for any alpha > 0 */
static float gamma_sample(Rng *rng, float alpha) {
    if (alpha >= 1.0f)
        return gamma_sample_ge1(rng, alpha);

    /* For alpha < 1: use Gamma(alpha+1) * U^(1/alpha) */
    float g = gamma_sample_ge1(rng, alpha + 1.0f);
    float u = rng_next_float(rng);
    if (u < 1e-30f) u = 1e-30f;
    return g * powf(u, 1.0f / alpha);
}

static void dirichlet_sample(Rng *rng, float alpha, float *out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = gamma_sample(rng, alpha);
        sum += out[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++)
            out[i] *= inv;
    }
}

/* ── Node helpers ────────────────────────────────────────────────── */

static Node *alloc_node(void) {
    Node *n = calloc(1, sizeof(Node));
    if (!n) return NULL;
    atomic_store(&n->state, NODE_UNEXPANDED);
    atomic_store(&n->visit_count, 0);
    atomic_store(&n->value_sum_fp, 0);
    return n;
}

static void free_tree(Node *node) {
    if (!node) return;
    int nc = node->num_children;
    for (int i = 0; i < nc; i++)
        free_tree(&node->children[i]);
    free(node->children);
    free(node->child_moves);
    /* Only free root (children are in contiguous array, freed by parent) */
}

static void free_root(Node *root) {
    if (!root) return;
    int nc = root->num_children;
    /* Recursively free grandchildren first */
    for (int i = 0; i < nc; i++) {
        Node *child = &root->children[i];
        int gc = child->num_children;
        for (int j = 0; j < gc; j++)
            free_tree(&child->children[j]);
        free(child->children);
        free(child->child_moves);
    }
    free(root->children);
    free(root->child_moves);
    free(root);
}

/* ── UCB selection ───────────────────────────────────────────────── */

static int select_child(Node *node, float c_puct) {
    int best_idx = 0;
    float best_score = -1e30f;
    int parent_visits = atomic_load(&node->visit_count);
    float sqrt_parent = sqrtf((float)parent_visits);

    int nc = node->num_children;
    for (int i = 0; i < nc; i++) {
        Node *child = &node->children[i];
        int vc = atomic_load(&child->visit_count);

        float q;
        if (vc == 0)
            q = 0.0f;
        else
            q = -(float)atomic_load(&child->value_sum_fp) / ((float)vc * FP_SCALE);

        float ucb = c_puct * child->prior * sqrt_parent / (1.0f + (float)vc);
        float score = q + ucb;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

/* ── Expand a leaf node ──────────────────────────────────────────── */

static float expand_or_evaluate(Node *node, Board *board,
                                InferenceBatcher *batcher,
                                float temperature) {
    /* Try to claim expansion rights via CAS */
    int expected = NODE_UNEXPANDED;
    if (atomic_compare_exchange_strong(&node->state, &expected, NODE_EXPANDING)) {
        /* We won: check for terminal position first */
        uint16_t moves[NUM_MOVES];
        int num_moves = legal_moves(board, moves);
        int result = game_result(board, num_moves);

        if (result != 2) {
            /* Terminal position */
            node->terminal_value = (float)result;
            node->num_children = 0;
            atomic_store(&node->state, NODE_TERMINAL);
            return node->terminal_value;
        }

        /* Neural network evaluation */
        InferResult infer_result;
        infer(batcher, board, &infer_result);

        /* Create children */
        node->children = calloc((size_t)num_moves, sizeof(Node));
        node->child_moves = malloc((size_t)num_moves * sizeof(uint16_t));
        node->num_children = (int16_t)num_moves;

        /* Compute priors from policy */
        float prior_sum = 0.0f;
        for (int i = 0; i < num_moves; i++)
            prior_sum += infer_result.policy[moves[i]];

        float inv_sum = (prior_sum > 0.0f) ? 1.0f / prior_sum : 0.0f;

        for (int i = 0; i < num_moves; i++) {
            node->child_moves[i] = moves[i];
            Node *child = &node->children[i];
            child->parent = node;
            float p = infer_result.policy[moves[i]] * inv_sum;
            if (temperature != 1.0f && p > 0.0f)
                p = powf(p, 1.0f / temperature);
            child->prior = p;
            atomic_store(&child->state, NODE_UNEXPANDED);
            atomic_store(&child->visit_count, 0);
            atomic_store(&child->value_sum_fp, 0);
        }

        /* Re-normalize priors after temperature */
        if (temperature != 1.0f) {
            float s = 0.0f;
            for (int i = 0; i < num_moves; i++)
                s += node->children[i].prior;
            if (s > 0.0f) {
                float inv = 1.0f / s;
                for (int i = 0; i < num_moves; i++)
                    node->children[i].prior *= inv;
            }
        }

        atomic_store(&node->state, NODE_EXPANDED);
        return infer_result.value;
    }

    /* Another thread is expanding this node - return 0 (virtual loss handles it) */
    return 0.0f;
}

/* ── Backup ──────────────────────────────────────────────────────── */

static void backup(Node *node, float value) {
    /* visit_count was already incremented during selection (virtual loss).
       We only need to add the actual value. */
    while (node != NULL) {
        int fp_val = (int)(value * FP_SCALE);
        atomic_fetch_add(&node->value_sum_fp, fp_val);
        value = -value;
        node = node->parent;
    }
}

/* ── Single simulation (one MCTS iteration) ──────────────────────── */

typedef struct {
    Node            *root;
    Board            board;      /* thread-local copy */
    InferenceBatcher *batcher;
    const SearchConfig *config;
    atomic_int      *sim_counter;
    Rng              rng;
} ThreadArg;

static void simulate_once(ThreadArg *arg) {
    Node *node = arg->root;
    Board *board = &arg->board;
    int depth = 0;

    /* Selection: walk down the tree following UCB */
    while (atomic_load(&node->state) == NODE_EXPANDED && node->num_children > 0) {
        int idx = select_child(node, arg->config->c_puct);
        Node *child = &node->children[idx];

        /* Virtual loss: increment visit count before evaluation */
        atomic_fetch_add(&child->visit_count, 1);

        make_move(board, node->child_moves[idx]);
        node = child;
        depth++;
    }

    /* Evaluate leaf */
    float value;
    int node_state = atomic_load(&node->state);
    if (node_state == NODE_TERMINAL) {
        value = node->terminal_value;
    } else {
        value = expand_or_evaluate(node, board, arg->batcher,
                                   arg->config->temperature);
    }

    /* Backup value */
    backup(node, value);

    /* Undo moves to restore board state */
    for (int i = 0; i < depth; i++)
        undo_move(board);
}

/* ── Worker thread entry ─────────────────────────────────────────── */

static void *search_worker(void *raw) {
    ThreadArg *arg = (ThreadArg *)raw;
    int total = arg->config->num_simulations;

    while (1) {
        int sim = atomic_fetch_add(arg->sim_counter, 1);
        if (sim >= total) break;
        simulate_once(arg);
    }

    return NULL;
}

/* ── Public: search() ────────────────────────────────────────────── */

SearchResult search(Board *board, InferenceBatcher *batcher,
                    const SearchConfig *config) {
    SearchResult result;
    memset(&result, 0, sizeof(result));

    Node *root = alloc_node();
    if (!root) return result;

    /* Expand root single-threaded */
    Board root_board;
    memcpy(&root_board, board, sizeof(Board));

    /* Initial visit for root */
    atomic_store(&root->visit_count, 1);
    float root_val = expand_or_evaluate(root, &root_board, batcher,
                                        config->temperature);

    if (atomic_load(&root->state) == NODE_TERMINAL || root->num_children == 0) {
        result.root_value = root_val;
        result.total_visits = 1;
        free_root(root);
        return result;
    }

    /* Apply Dirichlet noise to root priors */
    if (config->dirichlet_alpha > 0.0f) {
        Rng noise_rng;
        rng_seed(&noise_rng, (uint64_t)(uintptr_t)root ^ 0xdeadbeefcafeULL);
        int nc = root->num_children;
        float *noise = malloc((size_t)nc * sizeof(float));
        if (noise) {
            dirichlet_sample(&noise_rng, config->dirichlet_alpha, noise, nc);
            float w = config->dirichlet_weight;
            for (int i = 0; i < nc; i++)
                root->children[i].prior =
                    (1.0f - w) * root->children[i].prior + w * noise[i];
            free(noise);
        }
    }

    /* Backup root expansion value */
    {
        int fp_val = (int)(root_val * FP_SCALE);
        atomic_fetch_add(&root->value_sum_fp, fp_val);
    }

    /* Launch search threads */
    int num_threads = config->num_threads;
    if (num_threads < 1) num_threads = 1;

    /* We already did 1 simulation (root expansion), dispatch the rest */
    atomic_int sim_counter;
    atomic_store(&sim_counter, 1); /* start from 1 since root expansion counts */

    if (num_threads <= 1) {
        /* Inline path: run search on the calling thread (no pthread overhead).
           Critical for self-play with thousands of concurrent games. */
        ThreadArg arg;
        arg.root = root;
        memcpy(&arg.board, board, sizeof(Board));
        arg.batcher = batcher;
        arg.config = config;
        arg.sim_counter = &sim_counter;
        rng_seed(&arg.rng, 6364136223846793005ULL);
        search_worker(&arg);
    } else {
        /* Multi-threaded path for evaluation / analysis */
        pthread_t *threads = malloc((size_t)num_threads * sizeof(pthread_t));
        ThreadArg *args = malloc((size_t)num_threads * sizeof(ThreadArg));

        for (int i = 0; i < num_threads; i++) {
            args[i].root = root;
            memcpy(&args[i].board, board, sizeof(Board));
            args[i].batcher = batcher;
            args[i].config = config;
            args[i].sim_counter = &sim_counter;
            rng_seed(&args[i].rng, (uint64_t)(i + 1) * 6364136223846793005ULL);
        }

        for (int i = 0; i < num_threads; i++)
            pthread_create(&threads[i], NULL, search_worker, &args[i]);

        for (int i = 0; i < num_threads; i++)
            pthread_join(threads[i], NULL);

        free(threads);
        free(args);
    }

    /* Extract results from root */
    int nc = root->num_children;
    float total_visits_f = 0.0f;

    /* Build policy from visit counts */
    memset(result.policy, 0, sizeof(result.policy));
    int best_idx = 0;
    int best_visits = 0;

    for (int i = 0; i < nc; i++) {
        int vc = atomic_load(&root->children[i].visit_count);
        result.policy[root->child_moves[i]] = (float)vc;
        total_visits_f += (float)vc;
        if (vc > best_visits) {
            best_visits = vc;
            best_idx = i;
        }
    }

    /* Normalize policy */
    if (total_visits_f > 0.0f) {
        float inv = 1.0f / total_visits_f;
        for (int i = 0; i < nc; i++)
            result.policy[root->child_moves[i]] *= inv;
    }

    result.best_move = root->child_moves[best_idx];
    result.total_visits = atomic_load(&root->visit_count);
    result.root_value = (float)atomic_load(&root->value_sum_fp) /
                        ((float)result.total_visits * FP_SCALE);

    free_root(root);

    return result;
}

/* ── Move sampling ───────────────────────────────────────────────── */

static uint16_t sample_move_from_policy(
    const float *policy, const uint16_t *moves, int num_moves, Rng *rng) {

    float total = 0.0f;
    for (int i = 0; i < num_moves; i++)
        total += policy[moves[i]];

    if (total <= 0.0f)
        return moves[0];

    float r = rng_next_float(rng) * total;
    float cumul = 0.0f;

    for (int i = 0; i < num_moves; i++) {
        cumul += policy[moves[i]];
        if (r < cumul)
            return moves[i];
    }

    return moves[num_moves - 1];
}

/* ── Self-play: single game ──────────────────────────────────────── */

GameRecord *self_play_game(InferenceBatcher *batcher,
                           const SearchConfig *config) {
    int tensor_planes = batcher_tensor_planes(batcher);
    int history_steps = batcher_history_steps(batcher);
    int tensor_floats = tensor_planes * 64;

    /* Pre-allocate for a reasonable game length */
    int capacity = 512;
    float *states   = malloc((size_t)capacity * (size_t)tensor_floats * sizeof(float));
    float *policies = malloc((size_t)capacity * NUM_MOVES * sizeof(float));
    int count = 0;

    Board board;
    init_board(&board);

    static atomic_int game_counter = 0;
    int game_id = atomic_fetch_add(&game_counter, 1);

    Rng rng;
    rng_seed(&rng, (uint64_t)game_id * 6364136223846793005ULL + 42);

    int move_count = 0;

    while (1) {
        uint16_t moves[NUM_MOVES];
        int num_legal = legal_moves(&board, moves);
        int result = game_result(&board, num_legal);

        if (result != 2)
            break;

        /* Run MCTS search */
        SearchResult sr = search(&board, batcher, config);

        /* Record state */
        if (count >= capacity) {
            capacity *= 2;
            states   = realloc(states, (size_t)capacity * (size_t)tensor_floats * sizeof(float));
            policies = realloc(policies, (size_t)capacity * NUM_MOVES * sizeof(float));
        }

        Board board_copy;
        memcpy(&board_copy, &board, sizeof(Board));
        board_to_tensor(&board_copy, history_steps, tensor_planes,
                        states + (size_t)count * (size_t)tensor_floats);

        /* Record policy target */
        memcpy(policies + (size_t)count * NUM_MOVES, sr.policy,
               NUM_MOVES * sizeof(float));

        count++;

        /* Select move */
        uint16_t chosen;
        if (move_count < config->greedy_threshold)
            chosen = sample_move_from_policy(sr.policy, moves, num_legal, &rng);
        else
            chosen = sr.best_move;

        make_move(&board, chosen);
        move_count++;
    }

    /* Determine game outcome */
    uint16_t moves[NUM_MOVES];
    int num_legal = legal_moves(&board, moves);
    int result = game_result(&board, num_legal);

    float outcome;
    if (result == -1) {
        /* Side to move is checkmated: that side loses */
        /* If board.turn == 0 (white checkmated), outcome from white's perspective = -1 */
        /* If board.turn == 1 (black checkmated), outcome from white's perspective = +1 */
        outcome = (board.turn == 0) ? -1.0f : 1.0f;
    } else {
        outcome = 0.0f; /* draw */
    }

    /* Fill value targets from each position's current player perspective */
    float *values = malloc((size_t)count * sizeof(float));
    for (int i = 0; i < count; i++) {
        /* Position i was played at move i: white's turn if i%2==0 */
        if (i % 2 == 0)
            values[i] = outcome;       /* white's perspective */
        else
            values[i] = -outcome;      /* black's perspective */
    }

    GameRecord *record = malloc(sizeof(GameRecord));
    record->states = states;
    record->policies = policies;
    record->values = values;
    record->count = count;
    record->tensor_floats = tensor_floats;

    return record;
}

/* ── Dataset I/O ─────────────────────────────────────────────────── */

int write_game_record(const GameRecord *record, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "search: cannot open %s for writing\n", path);
        return -1;
    }

    int32_t count = record->count;
    int32_t tensor_floats = record->tensor_floats;

    fwrite(&count, sizeof(int32_t), 1, f);
    fwrite(&tensor_floats, sizeof(int32_t), 1, f);
    fwrite(record->states, sizeof(float),
           (size_t)count * (size_t)tensor_floats, f);
    fwrite(record->policies, sizeof(float),
           (size_t)count * NUM_MOVES, f);
    fwrite(record->values, sizeof(float), (size_t)count, f);

    fclose(f);
    return 0;
}

void free_game_record(GameRecord *record) {
    if (!record) return;
    free(record->states);
    free(record->policies);
    free(record->values);
    free(record);
}

/* ── Self-play: concurrent N games ───────────────────────────────── */

typedef struct {
    InferenceBatcher   *batcher;
    const SearchConfig *config;
    atomic_int         *games_dispatched;
    int                 num_games;

    /* Shared result collector (mutex-protected) */
    pthread_mutex_t    *collect_mutex;
    float             **all_states;
    float             **all_policies;
    float             **all_values;
    int                *total_count;
    int                *total_capacity;
    int                 tensor_floats;
    atomic_int         *games_completed;
} GameWorkerArg;

static void *game_worker(void *raw) {
    GameWorkerArg *arg = (GameWorkerArg *)raw;

    while (1) {
        int g = atomic_fetch_add(arg->games_dispatched, 1);
        if (g >= arg->num_games) break;

        GameRecord *record = self_play_game(arg->batcher, arg->config);
        if (!record) continue;

        /* Collect result under lock */
        pthread_mutex_lock(arg->collect_mutex);

        while (*arg->total_count + record->count > *arg->total_capacity) {
            *arg->total_capacity *= 2;
            *arg->all_states = realloc(*arg->all_states,
                (size_t)*arg->total_capacity * (size_t)arg->tensor_floats * sizeof(float));
            *arg->all_policies = realloc(*arg->all_policies,
                (size_t)*arg->total_capacity * NUM_MOVES * sizeof(float));
            *arg->all_values = realloc(*arg->all_values,
                (size_t)*arg->total_capacity * sizeof(float));
        }

        int off = *arg->total_count;
        memcpy(*arg->all_states + (size_t)off * (size_t)arg->tensor_floats,
               record->states,
               (size_t)record->count * (size_t)arg->tensor_floats * sizeof(float));
        memcpy(*arg->all_policies + (size_t)off * NUM_MOVES,
               record->policies,
               (size_t)record->count * NUM_MOVES * sizeof(float));
        memcpy(*arg->all_values + off,
               record->values,
               (size_t)record->count * sizeof(float));

        *arg->total_count += record->count;
        int done = atomic_fetch_add(arg->games_completed, 1) + 1;

        fprintf(stderr, "self-play: game %d/%d done (%d moves, %d total positions)\n",
                done, arg->num_games, record->count, *arg->total_count);

        pthread_mutex_unlock(arg->collect_mutex);

        free_game_record(record);
    }

    return NULL;
}

int self_play(InferenceBatcher *batcher, const SearchConfig *config,
              int num_games, const char *output_path) {
    int tensor_planes = batcher_tensor_planes(batcher);
    int tensor_floats = tensor_planes * 64;

    int total_count = 0;
    int total_capacity = 4096;
    float *all_states   = malloc((size_t)total_capacity * (size_t)tensor_floats * sizeof(float));
    float *all_policies = malloc((size_t)total_capacity * NUM_MOVES * sizeof(float));
    float *all_values   = malloc((size_t)total_capacity * sizeof(float));

    pthread_mutex_t collect_mutex;
    pthread_mutex_init(&collect_mutex, NULL);

    atomic_int games_dispatched;
    atomic_store(&games_dispatched, 0);
    atomic_int games_completed;
    atomic_store(&games_completed, 0);

    int cg = config->concurrent_games;
    if (cg < 1) cg = 1;
    if (cg > num_games) cg = num_games;

    GameWorkerArg *args = malloc((size_t)cg * sizeof(GameWorkerArg));
    pthread_t *threads = malloc((size_t)cg * sizeof(pthread_t));

    for (int i = 0; i < cg; i++) {
        args[i].batcher          = batcher;
        args[i].config           = config;
        args[i].games_dispatched = &games_dispatched;
        args[i].num_games        = num_games;
        args[i].collect_mutex    = &collect_mutex;
        args[i].all_states       = &all_states;
        args[i].all_policies     = &all_policies;
        args[i].all_values       = &all_values;
        args[i].total_count      = &total_count;
        args[i].total_capacity   = &total_capacity;
        args[i].tensor_floats    = tensor_floats;
        args[i].games_completed  = &games_completed;
    }

    fprintf(stderr, "self-play: %d games, %d concurrent, %d threads/tree, %d sims/move\n",
            num_games, cg, config->num_threads, config->num_simulations);

    /* Use 256KB stacks for game workers (default 8MB × 8192 = 64GB virtual).
       Peak measured stack ~101KB; 256KB gives 2.5x headroom. */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);

    for (int i = 0; i < cg; i++)
        pthread_create(&threads[i], &attr, game_worker, &args[i]);

    pthread_attr_destroy(&attr);

    for (int i = 0; i < cg; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    free(args);
    pthread_mutex_destroy(&collect_mutex);

    /* Write combined dataset */
    GameRecord combined = {
        .states = all_states,
        .policies = all_policies,
        .values = all_values,
        .count = total_count,
        .tensor_floats = tensor_floats
    };

    int ret = write_game_record(&combined, output_path);

    free(all_states);
    free(all_policies);
    free(all_values);

    fprintf(stderr, "self-play: wrote %d positions to %s\n",
            total_count, output_path);

    return ret;
}

/* ── Default config ──────────────────────────────────────────────── */

SearchConfig default_search_config(void) {
    return (SearchConfig){
        .num_simulations  = 400,
        .num_threads      = 1,
        .concurrent_games = 8192,
        .c_puct           = 1.41f,
        .temperature      = 1.0f,
        .greedy_threshold = 30,
        .dirichlet_alpha  = 0.03f,
        .dirichlet_weight = 0.25f
    };
}
