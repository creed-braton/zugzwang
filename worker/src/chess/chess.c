#define NUM_PIECES  12
#define NUM_SQUARES 64
#define NUM_CASTLING 4
#define NUM_EP_FILES 8

enum {
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
    EMPTY = -1
};

typedef struct {
    int8_t squares[NUM_SQUARES];
    uint8_t side;
    uint8_t castling;
    int8_t ep_file;
    uint8_t halfmove;
    uint16_t fullmove;
    uint64_t hash;
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
    b->halfmove = 0;
    b->fullmove = 1;
    b->hash     = hash(zt, b);
}
