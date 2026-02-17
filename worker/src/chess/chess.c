#define NUM_PIECES  12
#define NUM_SQUARES 64
#define NUM_CASTLING 4
#define NUM_EP_FILES 8
#define MAX_HISTORY 1024
#define MAX_MOVES   256

#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

enum {
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
    EMPTY = -1
};

typedef struct {
    uint8_t from;
    uint8_t to;
    int8_t promotion;
} Move;

typedef struct {
    Move move;
    int8_t captured;
    uint8_t castling;
    int8_t ep_file;
    uint8_t halfmove;
    uint8_t repetitions;
    uint64_t hash;
} Undo;

typedef struct {
    int8_t squares[NUM_SQUARES];
    uint8_t side;
    uint8_t castling;
    int8_t ep_file;
    uint8_t halfmove;
    uint8_t repetitions;
    uint16_t fullmove;
    uint64_t hash;
    uint16_t ply;
    Undo history[MAX_HISTORY];
} Board;

typedef struct {
    uint64_t piece_sq[NUM_PIECES][NUM_SQUARES];
    uint64_t side_to_move;
    uint64_t castling[NUM_CASTLING];
    uint64_t en_passant[NUM_EP_FILES];
} ZobristTable;

/* --- PRNG (xoshiro256**) for high-quality random 64-bit numbers --- */

static uint64_t rng_s[4];

static uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}

static void rng_seed(uint64_t seed) {
    /* SplitMix64 to seed the state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

void zobrist_init(ZobristTable *zt, uint64_t seed) {
    rng_seed(seed);

    for (int p = 0; p < NUM_PIECES; p++)
        for (int sq = 0; sq < NUM_SQUARES; sq++)
            zt->piece_sq[p][sq] = rng_next();

    zt->side_to_move = rng_next();

    for (int i = 0; i < NUM_CASTLING; i++)
        zt->castling[i] = rng_next();

    for (int i = 0; i < NUM_EP_FILES; i++)
        zt->en_passant[i] = rng_next();
}

uint64_t hash(const ZobristTable *zt, const Board *b) {
    uint64_t h = 0;

    for (int sq = 0; sq < NUM_SQUARES; sq++) {
        if (b->squares[sq] != EMPTY)
            h ^= zt->piece_sq[b->squares[sq]][sq];
    }

    if (b->side)
        h ^= zt->side_to_move;

    for (int i = 0; i < NUM_CASTLING; i++) {
        if (b->castling & (1 << i))
            h ^= zt->castling[i];
    }

    if (b->ep_file >= 0)
        h ^= zt->en_passant[b->ep_file];

    return h;
}

uint64_t update_hash(const ZobristTable *zt, uint64_t h,
                             int piece, int from, int to,
                             int captured) {
    h ^= zt->piece_sq[piece][from];
    h ^= zt->piece_sq[piece][to];

    if (captured != EMPTY)
        h ^= zt->piece_sq[captured][to];

    h ^= zt->side_to_move;

    return h;
}

void init_board(Board *b, const ZobristTable *zt) {
    static const int8_t start[NUM_SQUARES] = {
        WR, WN, WB, WQ, WK, WB, WN, WR,
        WP, WP, WP, WP, WP, WP, WP, WP,
        EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
        EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
        EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
        EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY,
        BP, BP, BP, BP, BP, BP, BP, BP,
        BR, BN, BB, BQ, BK, BB, BN, BR,
    };

    for (int sq = 0; sq < NUM_SQUARES; sq++)
        b->squares[sq] = start[sq];

    b->side     = 0;
    b->castling = 0xF;
    b->ep_file  = -1;
    b->halfmove    = 0;
    b->repetitions = 1;
    b->fullmove    = 1;
    b->ply         = 0;
    b->hash        = hash(zt, b);
}

/* --- Move: push & pop --- */

static const uint8_t castle_mask[NUM_SQUARES] = {
    (uint8_t)~CASTLE_WQ, 0xF, 0xF, 0xF, (uint8_t)~(CASTLE_WK | CASTLE_WQ), 0xF, 0xF, (uint8_t)~CASTLE_WK,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF,
    (uint8_t)~CASTLE_BQ, 0xF, 0xF, 0xF, (uint8_t)~(CASTLE_BK | CASTLE_BQ), 0xF, 0xF, (uint8_t)~CASTLE_BK,
};

void push(Board *b, const ZobristTable *zt, Move move) {
    Undo *u = &b->history[b->ply++];
    u->move        = move;
    u->castling    = b->castling;
    u->ep_file     = b->ep_file;
    u->halfmove    = b->halfmove;
    u->repetitions = b->repetitions;
    u->hash        = b->hash;

    int piece = b->squares[move.from];
    int captured = b->squares[move.to];
    int type = piece % 6;
    uint64_t h = b->hash;

    /* clear old en passant from hash */
    if (b->ep_file >= 0)
        h ^= zt->en_passant[b->ep_file];

    /* en passant capture */
    int ep_sq = -1;
    if (type == WP && b->ep_file >= 0) {
        int ep_target = b->side ? (16 + b->ep_file) : (40 + b->ep_file);
        if (move.to == ep_target) {
            ep_sq = b->side ? move.to + 8 : move.to - 8;
            captured = b->squares[ep_sq];
            h ^= zt->piece_sq[captured][ep_sq];
            b->squares[ep_sq] = EMPTY;
        }
    }
    u->captured = captured;

    /* remove piece from origin */
    h ^= zt->piece_sq[piece][move.from];
    b->squares[move.from] = EMPTY;

    /* remove captured piece from destination (not for en passant) */
    if (captured != EMPTY && ep_sq < 0)
        h ^= zt->piece_sq[captured][move.to];

    /* promotion */
    int placed = (move.promotion != EMPTY) ? move.promotion : piece;

    /* place piece at destination */
    h ^= zt->piece_sq[placed][move.to];
    b->squares[move.to] = placed;

    /* castling: move the rook */
    if (type == WK) {
        int diff = (int)move.to - (int)move.from;
        if (diff == 2 || diff == -2) {
            int rook = b->side * 6 + WR;
            int rook_from, rook_to;
            if (diff > 0) {
                rook_from = move.to + 1;
                rook_to   = move.from + 1;
            } else {
                rook_from = move.to - 2;
                rook_to   = move.from - 1;
            }
            h ^= zt->piece_sq[rook][rook_from];
            h ^= zt->piece_sq[rook][rook_to];
            b->squares[rook_from] = EMPTY;
            b->squares[rook_to]   = rook;
        }
    }

    /* update castling rights */
    uint8_t old_castling = b->castling;
    b->castling &= castle_mask[move.from] & castle_mask[move.to];
    uint8_t castle_diff = old_castling ^ b->castling;
    for (int i = 0; i < NUM_CASTLING; i++) {
        if (castle_diff & (1 << i))
            h ^= zt->castling[i];
    }

    /* en passant file */
    int move_diff = (int)move.to - (int)move.from;
    if (type == WP && (move_diff == 16 || move_diff == -16)) {
        b->ep_file = move.to % 8;
        h ^= zt->en_passant[b->ep_file];
    } else {
        b->ep_file = -1;
    }

    /* halfmove clock */
    if (type == WP || captured != EMPTY)
        b->halfmove = 0;
    else
        b->halfmove++;

    /* fullmove number */
    if (b->side)
        b->fullmove++;

    /* switch side */
    b->side ^= 1;
    h ^= zt->side_to_move;

    b->hash = h;

    /* repetition count: scan same-side positions since last irreversible move */
    int rep = 1;
    int stop = (int)b->ply - (int)b->halfmove;
    if (stop < 0) stop = 0;
    for (int i = (int)b->ply - 2; i >= stop; i -= 2) {
        if (b->history[i].hash == h)
            rep++;
    }
    b->repetitions = rep;
}

void pop(Board *b) {
    Undo *u = &b->history[--b->ply];
    Move move = u->move;

    b->side ^= 1;

    if (b->side)
        b->fullmove--;

    int placed = b->squares[move.to];
    int piece = (move.promotion != EMPTY) ? (int)(b->side * 6) : placed;
    int type = piece % 6;
    int captured = u->captured;

    b->squares[move.from] = piece;

    /* en passant: restore captured pawn to its original square */
    if (type == WP && u->ep_file >= 0 &&
        move.to == (b->side ? 16 + u->ep_file : 40 + u->ep_file)) {
        int ep_sq = b->side ? move.to + 8 : move.to - 8;
        b->squares[move.to] = EMPTY;
        b->squares[ep_sq]   = captured;
    } else {
        b->squares[move.to] = captured;
    }

    /* castling: restore the rook */
    if (type == WK) {
        int diff = (int)move.to - (int)move.from;
        if (diff == 2 || diff == -2) {
            int rook = b->side * 6 + WR;
            int rook_from, rook_to;
            if (diff > 0) {
                rook_from = move.to + 1;
                rook_to   = move.from + 1;
            } else {
                rook_from = move.to - 2;
                rook_to   = move.from - 1;
            }
            b->squares[rook_from] = rook;
            b->squares[rook_to]   = EMPTY;
        }
    }

    b->castling    = u->castling;
    b->ep_file     = u->ep_file;
    b->halfmove    = u->halfmove;
    b->repetitions = u->repetitions;
    b->hash        = u->hash;
}

/* --- Legal move generation --- */

static int find_king(const Board *b, int side) {
    int king = side * 6 + WK;
    for (int sq = 0; sq < NUM_SQUARES; sq++)
        if (b->squares[sq] == king)
            return sq;
    return -1;
}

int is_attacked(const Board *b, int sq, int by_side) {
    int r = sq / 8, c = sq % 8;
    int base = by_side * 6;

    /* pawns */
    int pr = r + (by_side ? 1 : -1);
    if (pr >= 0 && pr <= 7) {
        if (c > 0 && b->squares[pr * 8 + c - 1] == base + WP) return 1;
        if (c < 7 && b->squares[pr * 8 + c + 1] == base + WP) return 1;
    }

    /* knights */
    static const int kn[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int i = 0; i < 8; i++) {
        int nr = r + kn[i][0], nc = c + kn[i][1];
        if (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7)
            if (b->squares[nr * 8 + nc] == base + WN) return 1;
    }

    /* king */
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            if (!dr && !dc) continue;
            int nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7)
                if (b->squares[nr * 8 + nc] == base + WK) return 1;
        }

    /* diagonals (bishop, queen) */
    static const int diag[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    for (int d = 0; d < 4; d++) {
        int nr = r + diag[d][0], nc = c + diag[d][1];
        while (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7) {
            int p = b->squares[nr * 8 + nc];
            if (p != EMPTY) {
                if (p == base + WB || p == base + WQ) return 1;
                break;
            }
            nr += diag[d][0];
            nc += diag[d][1];
        }
    }

    /* straights (rook, queen) */
    static const int straight[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (int d = 0; d < 4; d++) {
        int nr = r + straight[d][0], nc = c + straight[d][1];
        while (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7) {
            int p = b->squares[nr * 8 + nc];
            if (p != EMPTY) {
                if (p == base + WR || p == base + WQ) return 1;
                break;
            }
            nr += straight[d][0];
            nc += straight[d][1];
        }
    }

    return 0;
}

static void add_if_legal(Board *b, Move *moves, int *n,
                         int from, int to, int8_t promo, int king_sq) {
    int piece = b->squares[from];
    int type = piece % 6;
    int side = b->side;
    int cap = b->squares[to];

    b->squares[from] = EMPTY;
    b->squares[to] = (promo != EMPTY) ? promo : piece;

    /* en passant: temporarily remove captured pawn */
    int ep_sq = -1, ep_cap = EMPTY;
    if (type == WP && b->ep_file >= 0) {
        int ep_target = side ? (16 + b->ep_file) : (40 + b->ep_file);
        if (to == ep_target) {
            ep_sq = side ? to + 8 : to - 8;
            ep_cap = b->squares[ep_sq];
            b->squares[ep_sq] = EMPTY;
        }
    }

    int ksq = (type == WK) ? to : king_sq;
    int legal = !is_attacked(b, ksq, !side);

    b->squares[from] = piece;
    b->squares[to] = cap;
    if (ep_sq >= 0)
        b->squares[ep_sq] = ep_cap;

    if (legal)
        moves[(*n)++] = (Move){from, to, promo};
}

static void gen_pawn_moves(Board *b, Move *moves, int *n,
                           int sq, int king_sq) {
    int side = b->side;
    int base = side * 6;
    int r = sq / 8, c = sq % 8;
    int fwd = side ? -8 : 8;
    int start_rank = side ? 6 : 1;
    int promo_rank = side ? 0 : 7;

    /* forward */
    int to = sq + fwd;
    if (b->squares[to] == EMPTY) {
        if (to / 8 == promo_rank) {
            for (int p = WQ; p >= WN; p--)
                add_if_legal(b, moves, n, sq, to, base + p, king_sq);
        } else {
            add_if_legal(b, moves, n, sq, to, EMPTY, king_sq);
        }

        /* double push */
        if (r == start_rank) {
            int to2 = to + fwd;
            if (b->squares[to2] == EMPTY)
                add_if_legal(b, moves, n, sq, to2, EMPTY, king_sq);
        }
    }

    /* captures */
    int dc[2] = {-1, 1};
    for (int i = 0; i < 2; i++) {
        int nc = c + dc[i];
        if (nc < 0 || nc > 7) continue;
        to = sq + fwd + dc[i];

        int target = b->squares[to];
        int is_capture = (target != EMPTY && target / 6 != side);
        int is_ep = (b->ep_file >= 0 &&
                     to == (side ? 16 + b->ep_file : 40 + b->ep_file));

        if (is_capture || is_ep) {
            if (to / 8 == promo_rank) {
                for (int p = WQ; p >= WN; p--)
                    add_if_legal(b, moves, n, sq, to, base + p, king_sq);
            } else {
                add_if_legal(b, moves, n, sq, to, EMPTY, king_sq);
            }
        }
    }
}

static void gen_slides(Board *b, Move *moves, int *n, int sq,
                       const int dirs[][2], int ndirs, int king_sq) {
    int side = b->side;
    int r = sq / 8, c = sq % 8;

    for (int d = 0; d < ndirs; d++) {
        int nr = r + dirs[d][0], nc = c + dirs[d][1];
        while (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7) {
            int to = nr * 8 + nc;
            int target = b->squares[to];
            if (target != EMPTY && target / 6 == side) break;
            add_if_legal(b, moves, n, sq, to, EMPTY, king_sq);
            if (target != EMPTY) break;
            nr += dirs[d][0];
            nc += dirs[d][1];
        }
    }
}

int legal_moves(Board *b, Move *moves) {
    int n = 0;
    int side = b->side;
    int opp = !side;
    int king_sq = find_king(b, side);

    static const int kn[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    static const int diag[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int straight[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int all8[8][2] = {
        {1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}
    };

    for (int sq = 0; sq < NUM_SQUARES; sq++) {
        int piece = b->squares[sq];
        if (piece == EMPTY || piece / 6 != side) continue;

        int type = piece % 6;
        int r = sq / 8, c = sq % 8;

        switch (type) {
        case WP:
            gen_pawn_moves(b, moves, &n, sq, king_sq);
            break;

        case WN:
            for (int i = 0; i < 8; i++) {
                int nr = r + kn[i][0], nc = c + kn[i][1];
                if (nr < 0 || nr > 7 || nc < 0 || nc > 7) continue;
                int to = nr * 8 + nc;
                if (b->squares[to] != EMPTY && b->squares[to] / 6 == side)
                    continue;
                add_if_legal(b, moves, &n, sq, to, EMPTY, king_sq);
            }
            break;

        case WB:
            gen_slides(b, moves, &n, sq, diag, 4, king_sq);
            break;

        case WR:
            gen_slides(b, moves, &n, sq, straight, 4, king_sq);
            break;

        case WQ:
            gen_slides(b, moves, &n, sq, all8, 8, king_sq);
            break;

        case WK:
            for (int dr = -1; dr <= 1; dr++)
                for (int dc = -1; dc <= 1; dc++) {
                    if (!dr && !dc) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr > 7 || nc < 0 || nc > 7) continue;
                    int to = nr * 8 + nc;
                    if (b->squares[to] != EMPTY && b->squares[to] / 6 == side)
                        continue;
                    add_if_legal(b, moves, &n, sq, to, EMPTY, king_sq);
                }

            /* castling kingside */
            if (b->castling & (side ? CASTLE_BK : CASTLE_WK))
                if (b->squares[sq + 1] == EMPTY &&
                    b->squares[sq + 2] == EMPTY)
                    if (!is_attacked(b, sq, opp) &&
                        !is_attacked(b, sq + 1, opp) &&
                        !is_attacked(b, sq + 2, opp))
                        moves[n++] = (Move){sq, sq + 2, EMPTY};

            /* castling queenside */
            if (b->castling & (side ? CASTLE_BQ : CASTLE_WQ))
                if (b->squares[sq - 1] == EMPTY &&
                    b->squares[sq - 2] == EMPTY &&
                    b->squares[sq - 3] == EMPTY)
                    if (!is_attacked(b, sq, opp) &&
                        !is_attacked(b, sq - 1, opp) &&
                        !is_attacked(b, sq - 2, opp))
                        moves[n++] = (Move){sq, sq - 2, EMPTY};
            break;
        }
    }

    return n;
}
