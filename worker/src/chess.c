#include "chess.h"
#include <string.h>

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

static ZobristTable zt;

void init_zobrist_table(void) {
  rng_seed(42ULL);

  for (int p = 0; p < NUM_PIECES; p++)
    for (int sq = 0; sq < NUM_SQUARES; sq++)
      zt.piece_sq[p][sq] = rng_next();

  zt.side_to_move = rng_next();

  for (int i = 0; i < NUM_CASTLING; i++)
    zt.castling[i] = rng_next();

  for (int i = 0; i < NUM_EP_FILES; i++)
    zt.en_passant[i] = rng_next();
}

static uint64_t hash(const Board *b) {
  uint64_t h = 0;

  for (int sq = 0; sq < NUM_SQUARES; sq++) {
    if (b->squares[sq] != EMPTY)
      h ^= zt.piece_sq[b->squares[sq]][sq];
  }

  if (b->turn)
    h ^= zt.side_to_move;

  for (int i = 0; i < NUM_CASTLING; i++) {
    if (b->castling & (1 << i))
      h ^= zt.castling[i];
  }

  if (b->ep_file >= 0)
    h ^= zt.en_passant[b->ep_file];

  return h;
}

void init_board(Board *b) {
    memset(b, 0, sizeof(*b));

    b->pawns   = 0x00FF00000000FF00ULL;
    b->knights = 0x4200000000000042ULL;
    b->bishops = 0x2400000000000024ULL;
    b->rooks   = 0x8100000000000081ULL;
    b->queens  = 0x0800000000000008ULL;
    b->kings   = 0x1000000000000010ULL;

    b->white   = 0x000000000000FFFFULL;
    b->black   = 0xFFFF000000000000ULL;

    /* Initialize mailbox */
    memset(b->squares, EMPTY, sizeof(b->squares));
    b->squares[0] = WR; b->squares[1] = WN; b->squares[2] = WB;
    b->squares[3] = WQ; b->squares[4] = WK; b->squares[5] = WB;
    b->squares[6] = WN; b->squares[7] = WR;
    for (int i = 8; i < 16; i++) b->squares[i] = WP;
    for (int i = 48; i < 56; i++) b->squares[i] = BP;
    b->squares[56] = BR; b->squares[57] = BN; b->squares[58] = BB;
    b->squares[59] = BQ; b->squares[60] = BK; b->squares[61] = BB;
    b->squares[62] = BN; b->squares[63] = BR;

    b->turn     = 0;
    b->castling = 0xF;   /* KQkq */
    b->ep_file  = -1;
    b->halfmove = 0;
    b->fullmove = 1;
    b->hash = hash(b);
}

#define BB(sq) (1ULL << (sq))
#define RANK(sq) ((sq) >> 3)
#define FILE(sq) ((sq) & 7)

static const uint64_t ROOK_MAGICS[64] = {
  0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL,
  0x0080040800100080ULL, 0x0080020400080080ULL, 0x0080010200040080ULL,
  0x0080008001000200ULL, 0x0080002040800100ULL, 0x0000800020400080ULL,
  0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
  0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL,
  0x0000800040800100ULL, 0x0000208000400080ULL, 0x0000404000201000ULL,
  0x0000808010002000ULL, 0x0000808008001000ULL, 0x0000808004000800ULL,
  0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
  0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL,
  0x0000080080100080ULL, 0x0000040080080080ULL, 0x0000020080040080ULL,
  0x0000010080800200ULL, 0x0000800080004100ULL, 0x0000204000800080ULL,
  0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
  0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL,
  0x0000800040800100ULL, 0x0000204000808000ULL, 0x0000200040008080ULL,
  0x0000100020008080ULL, 0x0000080010008080ULL, 0x0000040008008080ULL,
  0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
  0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL,
  0x0000080010008080ULL, 0x0000040008008080ULL, 0x0000020004008080ULL,
  0x0000800100020080ULL, 0x0000800041000080ULL, 0x00FFFCDDFCED714AULL,
  0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040810002101ULL,
  0x0001000204080011ULL, 0x0001000204000801ULL, 0x0001000082000401ULL,
  0x0001FFFAABFAD1A2ULL,
};
static uint64_t rook_mask[64];
static int      rook_shift[64];

#define ROOK_TABLE_SIZE 262144
static uint64_t rook_table_data[ROOK_TABLE_SIZE];
static uint64_t *rook_table[64];

static void init_rook_magics(void) {
  uint64_t *ptr = rook_table_data;

  for (int sq = 0; sq < 64; sq++) {
    int r = RANK(sq), f = FILE(sq);

    /* --- Build mask: ray squares, excluding board edges --- */
    uint64_t mask = 0;
    for (int i = r + 1; i < 7; i++) mask |= BB(i * 8 + f);
    for (int i = r - 1; i > 0; i--) mask |= BB(i * 8 + f);
    for (int i = f + 1; i < 7; i++) mask |= BB(r * 8 + i);
    for (int i = f - 1; i > 0; i--) mask |= BB(r * 8 + i);

    rook_mask[sq]  = mask;
    int bits       = __builtin_popcountll(mask);
    rook_shift[sq] = 64 - bits;

    /* --- Point this square's table slice into the flat array --- */
    rook_table[sq] = ptr;
    ptr += (1 << bits);

    /* --- Fill table: enumerate every subset of the mask --- */
    uint64_t subset = 0;
    do {
      /* Compute attacks the slow way: walk rays, stop at blockers */
      uint64_t attacks = 0;
      for (int i = r + 1; i <= 7; i++) { uint64_t b = BB(i*8+f); attacks |= b; if (subset & b) break; }
      for (int i = r - 1; i >= 0; i--) { uint64_t b = BB(i*8+f); attacks |= b; if (subset & b) break; }
      for (int i = f + 1; i <= 7; i++) { uint64_t b = BB(r*8+i); attacks |= b; if (subset & b) break; }
      for (int i = f - 1; i >= 0; i--) { uint64_t b = BB(r*8+i); attacks |= b; if (subset & b) break; }

      /* Store at the magic-hashed index */
      int idx = (int)((subset * ROOK_MAGICS[sq]) >> rook_shift[sq]);
      rook_table[sq][idx] = attacks;

      /* Carry-Rippler: next subset of mask */
      subset = (subset - mask) & mask;
    } while (subset);
  }
}

static inline uint64_t rook_attacks(int sq, uint64_t occ) {
  uint64_t masked = occ & rook_mask[sq];
  int idx = (int)((masked * ROOK_MAGICS[sq]) >> rook_shift[sq]);
  return rook_table[sq][idx];
}

static const uint64_t BISHOP_MAGICS[64] = {
  0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL,
  0x0004040080000000ULL, 0x0001104000000000ULL, 0x0000821040000000ULL,
  0x0000410410400000ULL, 0x0000104104104000ULL, 0x0000040404040400ULL,
  0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
  0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL,
  0x0000002082082000ULL, 0x0004000808080800ULL, 0x0002000404040400ULL,
  0x0001000202020200ULL, 0x0000800802004000ULL, 0x0000800400A00000ULL,
  0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
  0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL,
  0x0000404004010200ULL, 0x0000840000802000ULL, 0x0000404002011000ULL,
  0x0000808001041000ULL, 0x0000404000820800ULL, 0x0001041000202000ULL,
  0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
  0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL,
  0x0000808080010400ULL, 0x0000820820004000ULL, 0x0000410410002000ULL,
  0x0000082088001000ULL, 0x0000002011000800ULL, 0x0000080100400400ULL,
  0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
  0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL,
  0x0000000020880000ULL, 0x0000001002020000ULL, 0x0000040408020000ULL,
  0x0004040404040000ULL, 0x0002020202020000ULL, 0x0000104104104000ULL,
  0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
  0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL,
  0x0002020202020200ULL,
};

static uint64_t bishop_mask[64];
static int      bishop_shift[64];

#define BISHOP_TABLE_SIZE 32768
static uint64_t bishop_table_data[BISHOP_TABLE_SIZE];
static uint64_t *bishop_table[64];

static void init_bishop_magics(void) {
  uint64_t *ptr = bishop_table_data;

  for (int sq = 0; sq < 64; sq++) {
    int r = RANK(sq), f = FILE(sq);

    /* --- Build mask: diagonal squares, excluding board edges --- */
    uint64_t mask = 0;
    for (int i=r+1,j=f+1; i<7 && j<7; i++,j++) mask |= BB(i*8+j);
    for (int i=r+1,j=f-1; i<7 && j>0; i++,j--) mask |= BB(i*8+j);
    for (int i=r-1,j=f+1; i>0 && j<7; i--,j++) mask |= BB(i*8+j);
    for (int i=r-1,j=f-1; i>0 && j>0; i--,j--) mask |= BB(i*8+j);

    bishop_mask[sq]  = mask;
    int bits         = __builtin_popcountll(mask);
    bishop_shift[sq] = 64 - bits;

    bishop_table[sq] = ptr;
    ptr += (1 << bits);

    /* --- Fill table: enumerate every subset of the mask --- */
    uint64_t subset = 0;
    do {
      uint64_t attacks = 0;
      for (int i=r+1,j=f+1; i<=7&&j<=7; i++,j++) { uint64_t b=BB(i*8+j); attacks|=b; if(subset&b) break; }
      for (int i=r+1,j=f-1; i<=7&&j>=0; i++,j--) { uint64_t b=BB(i*8+j); attacks|=b; if(subset&b) break; }
      for (int i=r-1,j=f+1; i>=0&&j<=7; i--,j++) { uint64_t b=BB(i*8+j); attacks|=b; if(subset&b) break; }
      for (int i=r-1,j=f-1; i>=0&&j>=0; i--,j--) { uint64_t b=BB(i*8+j); attacks|=b; if(subset&b) break; }

      int idx = (int)((subset * BISHOP_MAGICS[sq]) >> bishop_shift[sq]);
      bishop_table[sq][idx] = attacks;

      subset = (subset - mask) & mask;
    } while (subset);
  }
}

static inline uint64_t bishop_attacks(int sq, uint64_t occ) {
  uint64_t masked = occ & bishop_mask[sq];
  int idx = (int)((masked * BISHOP_MAGICS[sq]) >> bishop_shift[sq]);
  return bishop_table[sq][idx];
}

static inline uint64_t queen_attacks(int sq, uint64_t occ) {
  return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

static uint64_t knight_attacks[64];
static uint64_t king_attacks[64];
static uint64_t pawn_attacks[2][64];

static void init_knight_attacks(void) {
  for (int sq = 0; sq < 64; sq++) {
    int r = RANK(sq), f = FILE(sq);
    uint64_t atk = 0;

    /* 8 possible L-shaped jumps, with bounds checking */
    if (r+2 <= 7 && f+1 <= 7) atk |= BB((r+2)*8 + f+1);
    if (r+2 <= 7 && f-1 >= 0) atk |= BB((r+2)*8 + f-1);
    if (r-2 >= 0 && f+1 <= 7) atk |= BB((r-2)*8 + f+1);
    if (r-2 >= 0 && f-1 >= 0) atk |= BB((r-2)*8 + f-1);
    if (r+1 <= 7 && f+2 <= 7) atk |= BB((r+1)*8 + f+2);
    if (r+1 <= 7 && f-2 >= 0) atk |= BB((r+1)*8 + f-2);
    if (r-1 >= 0 && f+2 <= 7) atk |= BB((r-1)*8 + f+2);
    if (r-1 >= 0 && f-2 >= 0) atk |= BB((r-1)*8 + f-2);

    knight_attacks[sq] = atk;
  }
}

static void init_king_attacks(void) {
  for (int sq = 0; sq < 64; sq++) {
    int r = RANK(sq), f = FILE(sq);
    uint64_t atk = 0;

    /* 8 surrounding squares */
    for (int dr = -1; dr <= 1; dr++) {
      for (int df = -1; df <= 1; df++) {
        if (dr == 0 && df == 0) continue;
        int nr = r + dr, nf = f + df;
        if (nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7)
          atk |= BB(nr * 8 + nf);
      }
    }

    king_attacks[sq] = atk;
  }
}

static void init_pawn_attacks(void) {
  for (int sq = 0; sq < 64; sq++) {
    int r = RANK(sq), f = FILE(sq);

    /* White pawns capture diagonally upward (increasing rank) */
    uint64_t w = 0;
    if (r+1 <= 7 && f+1 <= 7) w |= BB((r+1)*8 + f+1);
    if (r+1 <= 7 && f-1 >= 0) w |= BB((r+1)*8 + f-1);
    pawn_attacks[0][sq] = w;

    /* Black pawns capture diagonally downward (decreasing rank) */
    uint64_t b = 0;
    if (r-1 >= 0 && f+1 <= 7) b |= BB((r-1)*8 + f+1);
    if (r-1 >= 0 && f-1 >= 0) b |= BB((r-1)*8 + f-1);
    pawn_attacks[1][sq] = b;
  }
}

#define MF_NONE      0
#define MF_PROMO_N   (1 << 12)
#define MF_PROMO_B   (2 << 12)
#define MF_PROMO_R   (3 << 12)
#define MF_PROMO_Q   (4 << 12)
#define MF_EP        (5 << 12)
#define MF_CASTLE_K  (6 << 12)
#define MF_CASTLE_Q  (7 << 12)
#define MF_DOUBLE    (8 << 12)

#define ENCODE(from, to, f) ((uint16_t)((from) | ((to) << 6) | (f)))
#define MOVE_FROM(m)  ((m) & 0x3F)
#define MOVE_TO(m)    (((m) >> 6) & 0x3F)
#define MOVE_FLAGS(m) ((m) & 0xF000)

#define RANK1_BB  0x00000000000000FFULL
#define RANK2_BB  0x000000000000FF00ULL
#define RANK3_BB  0x0000000000FF0000ULL
#define RANK6_BB  0x0000FF0000000000ULL
#define RANK7_BB  0x00FF000000000000ULL
#define RANK8_BB  0xFF00000000000000ULL

static uint64_t between_bb[64][64];
static uint64_t line_bb[64][64];

static void init_lines(void) {
  for (int a = 0; a < 64; a++) {
    for (int b = 0; b < 64; b++) {
      uint64_t r = rook_attacks(a, 0);
      uint64_t bi = bishop_attacks(a, 0);
      if (r & BB(b)) {
        between_bb[a][b] = rook_attacks(a, BB(b))
          & rook_attacks(b, BB(a));
        line_bb[a][b] = (r & rook_attacks(b, 0))
          | BB(a) | BB(b);
      } else if (bi & BB(b)) {
        between_bb[a][b] = bishop_attacks(a, BB(b))
          & bishop_attacks(b, BB(a));
        line_bb[a][b] = (bi & bishop_attacks(b, 0))
          | BB(a) | BB(b);
      }
    }
  }
}

void init_lookup_tables(void) {
  init_rook_magics();
  init_bishop_magics();
  init_knight_attacks();
  init_king_attacks();
  init_pawn_attacks();
  init_lines();
}

static inline uint64_t attackers_to(const Board *b, int sq, uint64_t occ) {
  return (pawn_attacks[0][sq] & b->black & b->pawns)
    | (pawn_attacks[1][sq] & b->white & b->pawns)
    | (knight_attacks[sq]  & b->knights)
    | (rook_attacks(sq, occ) & (b->rooks | b->queens))
    | (bishop_attacks(sq, occ) & (b->bishops | b->queens))
    | (king_attacks[sq] & b->kings);
}

int legal_moves(Board *b, uint16_t moves[NUM_MOVES]) {
  const int us      = b->turn;
  const uint64_t occ   = b->white | b->black;
  const uint64_t side  = us == 0 ? b->white : b->black;
  const uint64_t enemy = us == 0 ? b->black : b->white;
  const int ksq = __builtin_ctzll(side & b->kings);

  int count = 0;

  uint64_t checkers = attackers_to(b, ksq, occ) & enemy;

  uint64_t pinned = 0;
  uint64_t pin_ray[64];

  {
    uint64_t cand =
      (rook_attacks(ksq, 0)   & (b->rooks  | b->queens) & enemy) |
      (bishop_attacks(ksq, 0) & (b->bishops | b->queens) & enemy);

    while (cand) {
      int p = __builtin_ctzll(cand);
      cand &= cand - 1;
      uint64_t btwn = between_bb[ksq][p] & occ;
      if (btwn && !(btwn & (btwn - 1))) {
        int sq = __builtin_ctzll(btwn);
        if (BB(sq) & side) {
          pinned |= BB(sq);
          pin_ray[sq] = line_bb[ksq][p];
        }
      }
    }
  }

  uint64_t check_mask;
  if (checkers) {
    if (checkers & (checkers - 1))
      goto king_moves;  /* double check: only king can move */
    int csq = __builtin_ctzll(checkers);
    check_mask = checkers | between_bb[ksq][csq];
  } else {
    check_mask = ~0ULL;
  }

  /* --- Knights --- (pinned knights can never move) */
  {
    uint64_t knights = side & b->knights & ~pinned;
    while (knights) {
      int from = __builtin_ctzll(knights);
      knights &= knights - 1;

      uint64_t atk = knight_attacks[from] & ~side & check_mask;
      while (atk) {
        int to = __builtin_ctzll(atk);
        atk &= atk - 1;
        moves[count++] = ENCODE(from, to, MF_NONE);
      }
    }
  }

  /* --- Bishops --- */
  {
    uint64_t bishops = side & b->bishops;
    while (bishops) {
      int from = __builtin_ctzll(bishops);
      bishops &= bishops - 1;

      uint64_t atk = bishop_attacks(from, occ) & ~side & check_mask;
      if (BB(from) & pinned) atk &= pin_ray[from];
      while (atk) {
        int to = __builtin_ctzll(atk);
        atk &= atk - 1;
        moves[count++] = ENCODE(from, to, MF_NONE);
      }
    }
  }

  /* --- Rooks --- */
  {
    uint64_t rooks = side & b->rooks;
    while (rooks) {
      int from = __builtin_ctzll(rooks);
      rooks &= rooks - 1;

      uint64_t atk = rook_attacks(from, occ) & ~side & check_mask;
      if (BB(from) & pinned) atk &= pin_ray[from];
      while (atk) {
        int to = __builtin_ctzll(atk);
        atk &= atk - 1;
        moves[count++] = ENCODE(from, to, MF_NONE);
      }
    }
  }

  /* --- Queens --- */
  {
    uint64_t queens = side & b->queens;
    while (queens) {
      int from = __builtin_ctzll(queens);
      queens &= queens - 1;

      uint64_t atk = queen_attacks(from, occ) & ~side & check_mask;
      if (BB(from) & pinned) atk &= pin_ray[from];
      while (atk) {
        int to = __builtin_ctzll(atk);
        atk &= atk - 1;
        moves[count++] = ENCODE(from, to, MF_NONE);
      }
    }
  }

  /* --- Pawns --- */
  {
    const uint64_t pw = side & b->pawns;
    const uint64_t empty = ~occ;
    const int dir = us == 0 ? 8 : -8;

    const uint64_t promo_rank = us == 0 ? RANK8_BB : RANK1_BB;
    const uint64_t rank3 = us == 0 ? RANK3_BB : RANK6_BB;

    uint64_t push1 = us == 0 ? (pw << 8) & empty
      : (pw >> 8) & empty;
    uint64_t push1_promo  = push1 &  promo_rank;
    uint64_t push1_normal = push1 & ~promo_rank;

    uint64_t push2 = us == 0 ? ((push1_normal & rank3) << 8) & empty
      : ((push1_normal & rank3) >> 8) & empty;

    /* single pushes (no promo) */
    {
      uint64_t t = push1_normal & check_mask;
      while (t) {
        int to = __builtin_ctzll(t); t &= t - 1;
        int from = to - dir;
        if ((BB(from) & pinned) && !(BB(to) & pin_ray[from]))
          continue;
        moves[count++] = ENCODE(from, to, MF_NONE);
      }
    }

    /* double pushes */
    {
      uint64_t t = push2 & check_mask;
      while (t) {
        int to = __builtin_ctzll(t); t &= t - 1;
        int from = to - 2 * dir;
        if ((BB(from) & pinned) && !(BB(to) & pin_ray[from]))
          continue;
        moves[count++] = ENCODE(from, to, MF_DOUBLE);
      }
    }

    /* push promotions */
    {
      uint64_t t = push1_promo & check_mask;
      while (t) {
        int to = __builtin_ctzll(t); t &= t - 1;
        int from = to - dir;
        if ((BB(from) & pinned) && !(BB(to) & pin_ray[from]))
          continue;
        moves[count++] = ENCODE(from, to, MF_PROMO_Q);
        moves[count++] = ENCODE(from, to, MF_PROMO_R);
        moves[count++] = ENCODE(from, to, MF_PROMO_B);
        moves[count++] = ENCODE(from, to, MF_PROMO_N);
      }
    }

    /* captures (including capture-promotions) */
    {
      uint64_t p = pw;
      while (p) {
        int from = __builtin_ctzll(p); p &= p - 1;
        uint64_t cap = pawn_attacks[us][from] & enemy & check_mask;
        if (BB(from) & pinned) cap &= pin_ray[from];

        uint64_t cap_promo = cap &  promo_rank;
        uint64_t cap_normal = cap & ~promo_rank;

        while (cap_normal) {
          int to = __builtin_ctzll(cap_normal);
          cap_normal &= cap_normal - 1;
          moves[count++] = ENCODE(from, to, MF_NONE);
        }
        while (cap_promo) {
          int to = __builtin_ctzll(cap_promo);
          cap_promo &= cap_promo - 1;
          moves[count++] = ENCODE(from, to, MF_PROMO_Q);
          moves[count++] = ENCODE(from, to, MF_PROMO_R);
          moves[count++] = ENCODE(from, to, MF_PROMO_B);
          moves[count++] = ENCODE(from, to, MF_PROMO_N);
        }
      }
    }

    /* en passant */
    if (b->ep_file >= 0) {
      int ep_to  = us == 0 ? 40 + b->ep_file
        : 16 + b->ep_file;
      int cap_sq = ep_to - dir;

      uint64_t ep_pawns = pawn_attacks[1 - us][ep_to] & pw;
      while (ep_pawns) {
        int from = __builtin_ctzll(ep_pawns);
        ep_pawns &= ep_pawns - 1;

        if ((BB(from) & pinned) && !(BB(ep_to) & pin_ray[from]))
          continue;
        if (checkers && !((BB(ep_to) | BB(cap_sq)) & check_mask))
          continue;

        /* horizontal discovered check: both pawns leave the
          rank, possibly exposing king to rook/queen */
        {
          uint64_t new_occ = (occ ^ BB(from) ^ BB(cap_sq))
            | BB(ep_to);
          if (rook_attacks(ksq, new_occ) &
            (b->rooks | b->queens) & enemy)
            continue;
          if (bishop_attacks(ksq, new_occ) &
            (b->bishops | b->queens) & enemy)
            continue;
        }

        moves[count++] = ENCODE(from, ep_to, MF_EP);
      }
    }
  }

  /* --- Castling --- (only when not in check) */
  if (!checkers) {
    if (us == 0) {
      if ((b->castling & 1) &&
        !(occ & (BB(5) | BB(6))) &&
        !(attackers_to(b, 5, occ) & enemy) &&
        !(attackers_to(b, 6, occ) & enemy))
        moves[count++] = ENCODE(4, 6, MF_CASTLE_K);

      if ((b->castling & 2) &&
        !(occ & (BB(1) | BB(2) | BB(3))) &&
        !(attackers_to(b, 2, occ) & enemy) &&
        !(attackers_to(b, 3, occ) & enemy))
        moves[count++] = ENCODE(4, 2, MF_CASTLE_Q);
    } else {
      if ((b->castling & 4) &&
        !(occ & (BB(61) | BB(62))) &&
        !(attackers_to(b, 61, occ) & enemy) &&
        !(attackers_to(b, 62, occ) & enemy))
        moves[count++] = ENCODE(60, 62, MF_CASTLE_K);

      if ((b->castling & 8) &&
        !(occ & (BB(57) | BB(58) | BB(59))) &&
        !(attackers_to(b, 58, occ) & enemy) &&
        !(attackers_to(b, 59, occ) & enemy))
        moves[count++] = ENCODE(60, 58, MF_CASTLE_Q);
    }
  }

king_moves:
  /* --- King --- (always generated, even in double check) */
  {
    uint64_t occ_no_k = occ ^ BB(ksq);
    uint64_t atk = king_attacks[ksq] & ~side;
    while (atk) {
      int to = __builtin_ctzll(atk);
      atk &= atk - 1;
      if (!(attackers_to(b, to, occ_no_k) & enemy))
        moves[count++] = ENCODE(ksq, to, MF_NONE);
    }
  }

  return count;
}

/* ── Helpers ─────────────────────────────────────────────────────── */

static inline int piece_on(const Board *b, int sq) {
    return b->squares[sq];
}

static inline void toggle_piece(Board *b, int piece, int sq) {
    uint64_t bit = BB(sq);
    if (piece < 6) b->white ^= bit; else b->black ^= bit;
    switch (piece % 6) {
        case 0: b->pawns   ^= bit; break;
        case 1: b->knights ^= bit; break;
        case 2: b->bishops ^= bit; break;
        case 3: b->rooks   ^= bit; break;
        case 4: b->queens  ^= bit; break;
        case 5: b->kings   ^= bit; break;
    }
    b->squares[sq] = ((b->white | b->black) & bit) ? piece : EMPTY;
}

static const uint8_t castle_mask[64] = {
    13, 15, 15, 15, 12, 15, 15, 14, /* rank 1: a1=13 e1=12 h1=14 */
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11, /* rank 8: a8=7  e8=3  h8=11 */
};

/* ── make_move ──────────────────────────────────────────────────── */

void make_move(Board *b, uint16_t move) {
    const int from  = MOVE_FROM(move);
    const int to    = MOVE_TO(move);
    const int flags = MOVE_FLAGS(move);
    const int us    = b->turn;

    /* ---- Save undo state ---- */
    Undo *u     = &b->history[b->ply];
    u->move     = move;
    u->castling = b->castling;
    u->ep_file  = b->ep_file;
    u->halfmove = b->halfmove;
    u->hash     = b->hash;

    uint64_t h = b->hash;

    /* Remove old castling & ep from hash */
    for (int i = 0; i < 4; i++)
        if (b->castling & (1 << i)) h ^= zt.castling[i];
    if (b->ep_file >= 0)
        h ^= zt.en_passant[b->ep_file];

    int piece = piece_on(b, from);

    /* ---- Handle each move type ---- */

    if (flags == MF_CASTLE_K || flags == MF_CASTLE_Q) {
        int rook_from, rook_to;
        if (flags == MF_CASTLE_K) {
            rook_from = us == 0 ? 7  : 63;
            rook_to   = us == 0 ? 5  : 61;
        } else {
            rook_from = us == 0 ? 0  : 56;
            rook_to   = us == 0 ? 3  : 59;
        }
        int rook = us == 0 ? WR : BR;

        toggle_piece(b, piece, from);
        toggle_piece(b, piece, to);
        toggle_piece(b, rook, rook_from);
        toggle_piece(b, rook, rook_to);

        h ^= zt.piece_sq[piece][from] ^ zt.piece_sq[piece][to];
        h ^= zt.piece_sq[rook][rook_from] ^ zt.piece_sq[rook][rook_to];

        u->captured = EMPTY;

    } else if (flags == MF_EP) {
        int cap_sq  = us == 0 ? to - 8 : to + 8;
        int cap_pce = us == 0 ? BP : WP;

        toggle_piece(b, piece, from);
        toggle_piece(b, piece, to);
        toggle_piece(b, cap_pce, cap_sq);

        h ^= zt.piece_sq[piece][from] ^ zt.piece_sq[piece][to];
        h ^= zt.piece_sq[cap_pce][cap_sq];

        u->captured = cap_pce;

    } else if (flags >= MF_PROMO_N && flags <= MF_PROMO_Q) {
        /* Promotion piece: flags>>12 gives 1=N 2=B 3=R 4=Q */
        int promo = (flags >> 12) + (us == 0 ? 0 : 6);
        int captured = piece_on(b, to);
        u->captured  = captured;

        /* Remove captured piece */
        if (captured != EMPTY) {
            toggle_piece(b, captured, to);
            h ^= zt.piece_sq[captured][to];
        }

        /* Remove pawn, place promoted piece */
        toggle_piece(b, piece, from);
        toggle_piece(b, promo, to);

        h ^= zt.piece_sq[piece][from];
        h ^= zt.piece_sq[promo][to];

    } else {
        /* Normal move or double push */
        int captured = piece_on(b, to);
        u->captured  = captured;

        if (captured != EMPTY) {
            toggle_piece(b, captured, to);
            h ^= zt.piece_sq[captured][to];
        }

        toggle_piece(b, piece, from);
        toggle_piece(b, piece, to);

        h ^= zt.piece_sq[piece][from] ^ zt.piece_sq[piece][to];
    }

    /* ---- Update castling rights ---- */
    b->castling &= castle_mask[from] & castle_mask[to];

    /* ---- Update en-passant file ---- */
    if (flags == MF_DOUBLE)
        b->ep_file = FILE(from);
    else
        b->ep_file = -1;

    /* ---- Halfmove clock ---- */
    if (piece % 6 == 0 || u->captured != EMPTY)
        b->halfmove = 0;
    else
        b->halfmove++;

    /* ---- Fullmove counter ---- */
    if (us == 1) b->fullmove++;

    /* ---- Flip side and finalise hash ---- */
    b->turn ^= 1;
    h ^= zt.side_to_move;

    for (int i = 0; i < 4; i++)
        if (b->castling & (1 << i)) h ^= zt.castling[i];
    if (b->ep_file >= 0)
        h ^= zt.en_passant[b->ep_file];

    b->hash = h;
    b->ply++;
}

/* ── undo_move ──────────────────────────────────────────────────── */

void undo_move(Board *b) {
    b->ply--;
    const Undo *u = &b->history[b->ply];

    const uint16_t move  = u->move;
    const int from  = MOVE_FROM(move);
    const int to    = MOVE_TO(move);
    const int flags = MOVE_FLAGS(move);

    /* Flip side back first so "us" is the side that made the move */
    b->turn ^= 1;
    const int us = b->turn;

    b->hash = u->hash;

    /* ---- Restore state ---- */
    b->castling = u->castling;
    b->ep_file  = u->ep_file;
    b->halfmove = u->halfmove;
    if (us == 1) b->fullmove--;

    /* ---- Reverse the move ---- */

    if (flags == MF_CASTLE_K || flags == MF_CASTLE_Q) {
        int rook_from, rook_to;
        if (flags == MF_CASTLE_K) {
            rook_from = us == 0 ? 7  : 63;
            rook_to   = us == 0 ? 5  : 61;
        } else {
            rook_from = us == 0 ? 0  : 56;
            rook_to   = us == 0 ? 3  : 59;
        }
        int king = us == 0 ? WK : BK;
        int rook = us == 0 ? WR : BR;

        toggle_piece(b, king, to);
        toggle_piece(b, king, from);
        toggle_piece(b, rook, rook_to);
        toggle_piece(b, rook, rook_from);

    } else if (flags == MF_EP) {
        int pawn   = us == 0 ? WP : BP;
        int cap_sq = us == 0 ? to - 8 : to + 8;

        toggle_piece(b, pawn, to);
        toggle_piece(b, pawn, from);
        toggle_piece(b, u->captured, cap_sq);

    } else if (flags >= MF_PROMO_N && flags <= MF_PROMO_Q) {
        int pawn  = us == 0 ? WP : BP;
        int promo = (flags >> 12) + (us == 0 ? 0 : 6);

        /* Remove promoted piece, restore pawn */
        toggle_piece(b, promo, to);
        toggle_piece(b, pawn, from);

        /* Restore captured piece */
        if (u->captured != EMPTY)
            toggle_piece(b, u->captured, to);

    } else {
        /* Normal / double-push */
        int piece = piece_on(b, to);

        toggle_piece(b, piece, to);
        toggle_piece(b, piece, from);

        if (u->captured != EMPTY)
            toggle_piece(b, u->captured, to);
    }
}

int insufficient_material(const Board *b) {
  if (b->pawns | b->rooks | b->queens)
    return 0;

  uint64_t knights = b->knights;
  uint64_t bishops = b->bishops;

  /* K vs K */
  if (!knights && !bishops)
    return 1;

  /* Only bishops remain: insufficient iff all on same square color */
  if (!knights) {
    const uint64_t DARK = 0xAA55AA55AA55AA55ULL;
    return !(bishops & DARK) || !(bishops & ~DARK);
  }

  /* Single knight, no bishops: KN vs K */
  if (!bishops && !(knights & (knights - 1)))
    return 1;

  return 0;
}

int repetitions(const Board *b) {
  int reps = 0;
  int limit = b->halfmove;

  for (int i = 2; i <= limit && i <= b->ply; i += 2) {
    if (b->history[b->ply - i].hash == b->hash)
      reps++;
  }

  return reps;
}

int game_result(const Board *b, int num_legal_moves) {
  if (num_legal_moves == 0) {
    uint64_t occ = b->white | b->black;
    uint64_t side = b->turn == 0 ? b->white : b->black;
    uint64_t enemy = b->turn == 0 ? b->black : b->white;
    int ksq = __builtin_ctzll(side & b->kings);
    if (attackers_to(b, ksq, occ) & enemy)
      return -1;  /* side to move is checkmated */
    return 0;
  }

  if (b->halfmove >= 100)
    return 0;

  if (repetitions(b) >= 2)
    return 0;

  if (insufficient_material(b))
    return 0;

  return 2;
}
