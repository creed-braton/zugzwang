#ifndef CHESS_H
#define CHESS_H

#include <stdint.h>

#define NUM_PIECES   12
#define NUM_SQUARES  64
#define NUM_CASTLING 4
#define NUM_EP_FILES 8
#define NUM_MOVES    4168
#define MAX_HISTORY  1024

enum {
  WP, WN, WB, WR, WQ, WK,
  BP, BN, BB, BR, BQ, BK,
  EMPTY = -1
};

enum {
  MT_NORMAL, MT_DOUBLE, MT_CASTLE_K, MT_CASTLE_Q,
  MT_EP, MT_PROMO_Q, MT_PROMO_N, MT_PROMO_B, MT_PROMO_R
};

typedef struct {
    uint16_t move;
    int8_t   captured;
    uint8_t  castling;
    int8_t   ep_file;
    uint8_t  halfmove;
    uint8_t  type;
    uint64_t hash;
} Undo;

typedef struct {
  uint64_t pawns;
  uint64_t knights;
  uint64_t bishops;
  uint64_t rooks;
  uint64_t queens;
  uint64_t kings;

  uint64_t white;
  uint64_t black;

  int8_t   squares[NUM_SQUARES];

  uint8_t  turn;
  uint8_t  castling;
  int8_t   ep_file;
  uint8_t  halfmove;
  uint16_t fullmove;

  uint64_t hash;
  uint16_t ply;
  Undo history[MAX_HISTORY];
} Board;

void init_zobrist_table(void);
void init_lookup_tables(void);
void init_board(Board *b);

int  legal_moves(const Board *b, uint16_t moves[NUM_MOVES]);
void make_move(Board *b, uint16_t move);
void undo_move(Board *b);

int  repetitions(const Board *b);
int  game_result(const Board *b, int num_legal_moves);

#endif
