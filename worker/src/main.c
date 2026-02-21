#include "chess.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    init_zobrist_table();
    init_lookup_tables();

    Board b;
    init_board(&b);

    uint16_t moves[NUM_MOVES];
    char buf[8];
    char line[64];
    int first_draw = 1;
    int prev_error = 0;

    for (;;) {
        int n = legal_moves(&b, moves);
        int result = game_result(&b, n);

        if (!first_draw) {
            int up = 13 + prev_error;
            printf("\033[%dA", up);
        }
        first_draw = 0;
        prev_error = 0;

        print_board(&b);

        if (result != 2) {
            if (result == -1)
                printf("%s wins by checkmate!\n",
                       b.turn == 0 ? "Black" : "White");
            else
                printf("Draw!\n");
            break;
        }

        printf("\n%s to move\n", b.turn == 0 ? "White" : "Black");

        printf("Legal:");
        for (int i = 0; i < n; i++) {
            move_to_str(moves[i], &b, buf);
            printf(" %s", buf);
        }
        printf("\n");

        printf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin))
            break;

        /* strip newline */
        line[strcspn(line, "\n")] = '\0';

        if (strcmp(line, "quit") == 0)
            break;

        uint16_t move;
        if (!str_to_move(&b, line, &move)) {
            printf("Illegal move: %s\n", line);
            prev_error = 1;
            continue;
        }

        make_move(&b, move);
    }

    return 0;
}
