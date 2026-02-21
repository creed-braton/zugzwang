#include "chess.h"
#include "infer.h"
#include "search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --model <path> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>       ONNX model path (required)\n"
        "  --output <path>      Output dataset path (default: dataset.bin)\n"
        "  --games <n>          Number of self-play games (default: 1)\n"
        "  --concurrent <n>     Games running in parallel (default: 8192)\n"
        "  --sims <n>           MCTS simulations per move (default: 400)\n"
        "  --threads <n>        Search threads per tree (default: 1)\n"
        "  --batch-size <n>     Inference batch size (default: 2048)\n"
        "  --timeout <sec>      Batch timeout in seconds (default: 0.1)\n"
        "  --history <n>        History steps for encoding (default: 8)\n"
        "  --c-puct <f>         Exploration constant (default: 1.41)\n"
        "  --temperature <f>    Policy temperature (default: 1.0)\n"
        "  --greedy-after <n>   Greedy move selection after N moves (default: 30)\n"
        "  --dirichlet-alpha <f>  Dirichlet noise alpha, 0=off (default: 0.03)\n"
        "  --dirichlet-weight <f> Noise mixing weight (default: 0.25)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *output_path = "dataset.bin";
    int num_games = 1;
    int batch_size = 2048;
    double timeout_sec = 0.1;
    int history_steps = 8;

    SearchConfig config = default_search_config();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc)
            num_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--concurrent") == 0 && i + 1 < argc)
            config.concurrent_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sims") == 0 && i + 1 < argc)
            config.num_simulations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            config.num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            timeout_sec = atof(argv[++i]);
        else if (strcmp(argv[i], "--history") == 0 && i + 1 < argc)
            history_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--c-puct") == 0 && i + 1 < argc)
            config.c_puct = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            config.temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--greedy-after") == 0 && i + 1 < argc)
            config.greedy_threshold = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dirichlet-alpha") == 0 && i + 1 < argc)
            config.dirichlet_alpha = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--dirichlet-weight") == 0 && i + 1 < argc)
            config.dirichlet_weight = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model is required\n");
        usage(argv[0]);
        return 1;
    }

    init_zobrist_table();
    init_lookup_tables();

    /* Total threads that can submit inference requests simultaneously:
       concurrent_games * num_threads (each game's search workers) */
    int total_search_threads = config.concurrent_games * config.num_threads;

    InferenceBatcher *batcher = init_batcher(model_path, batch_size,
                                             timeout_sec, history_steps,
                                             total_search_threads);
    if (!batcher) {
        fprintf(stderr, "Failed to initialize inference batcher\n");
        return 1;
    }

    int ret = self_play(batcher, &config, num_games, output_path);

    free_batcher(batcher);
    return ret;
}
