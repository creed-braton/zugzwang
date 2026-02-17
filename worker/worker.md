# Worker — Remaining Chess Operations

## 1. Game-Over Detection

Used by MCTS leaf evaluation (`search.py`) and the self-play game loop (`dataset.py`).

### `int is_in_check(const Board *b)`
Return whether the side to move is in check.
```c
int king_sq = find_king(b, b->side);
return is_attacked(b, king_sq, !b->side);
```

### `int is_checkmate(Board *b)`
In check and no legal moves.
```c
if (!is_in_check(b)) return 0;
Move moves[MAX_MOVES];
return legal_moves(b, moves) == 0;
```

### `int is_stalemate(Board *b)`
Not in check and no legal moves.
```c
if (is_in_check(b)) return 0;
Move moves[MAX_MOVES];
return legal_moves(b, moves) == 0;
```

### `int is_draw(Board *b)`
Any of:
- **50-move rule**: `b->halfmove >= 100`
- **Threefold repetition**: `b->repetitions >= 3`
- **Insufficient material**: only the following piece combos remain:
  - K vs K
  - K+N vs K
  - K+B vs K
  - K+B vs K+B (bishops on same color square)

Scan the board once, count pieces by type, then check the conditions above.

### `int is_game_over(Board *b)`
```c
return is_checkmate(b) || is_stalemate(b) || is_draw(b);
```

### `float game_result(Board *b)`
Returns outcome from the **current side's perspective** (matches how `search.py` uses it):
- Checkmate: `-1.0` (side to move lost)
- Any draw: `0.0`

---

## 2. Board-to-Tensor Encoding

C equivalent of `encode.py:board_to_tensor()`. Produces the float buffer that gets fed into the ONNX model.

### Layout
- `14 * history_steps + 7` planes, each `8 x 8` floats
- Total size: `(14 * H + 7) * 64` floats (default H=8 gives 7232 floats)

### Per history step `t` (planes `t*14` to `t*14+13`)
| Plane offset | Content |
|---|---|
| 0-5 | Current side's P, N, B, R, Q, K |
| 6-11 | Opponent's P, N, B, R, Q, K |
| 12 | 2-fold repetition (all 1s if `repetitions >= 2`) |
| 13 | 3-fold repetition (all 1s if `repetitions >= 3`) |

For each step, pop one move from history to walk backward. Vertical flip when black is to move (row = 7 - row) so pawn direction is always "up".

### Global planes (after all history steps)
| Plane offset | Content |
|---|---|
| `14*H + 0` | Current side queenside castling rights (all 1s or 0s) |
| `14*H + 1` | Current side kingside castling rights |
| `14*H + 2` | Opponent queenside castling rights |
| `14*H + 3` | Opponent kingside castling rights |
| `14*H + 4` | Halfmove clock (scalar broadcast to all 64) |
| `14*H + 5` | Fullmove number (scalar broadcast to all 64) |
| `14*H + 6` | Unused (zeros) |

Note: `board_to_tensor` has 7 global planes but only 6 are written. The 7th stays zero.

### Suggested signature
```c
void board_to_tensor(Board *b, float *out, int history_steps);
```
- `out` is a caller-provided buffer of `(14 * history_steps + 7) * 64` floats, zeroed beforehand.
- Uses `pop()` to walk history, then `push()` to restore (needs `ZobristTable`).
- Alternative: read directly from `b->history[]` undo entries to avoid push/pop, but the repetition planes require the board state at each step.

### Implementation sketch
```c
void board_to_tensor(Board *b, const ZobristTable *zt, float *out, int history_steps) {
    int flip = b->side;  /* flip when black to move */
    int removed = 0;

    for (int t = 0; t < history_steps; t++) {
        int offset = t * 14 * 64;

        for (int sq = 0; sq < 64; sq++) {
            int piece = b->squares[sq];
            if (piece == EMPTY) continue;

            int r = sq / 8, c = sq % 8;
            if (flip) r = 7 - r;
            int idx = r * 8 + c;

            int color = piece / 6;
            int ptype = piece % 6;
            int plane = (color ^ flip) ? ptype : ptype + 6;
            /* color ^ flip: 0 = current side, 1 = opponent */
            out[offset + plane * 64 + idx] = 1.0f;
        }

        if (b->repetitions >= 2)
            for (int i = 0; i < 64; i++)
                out[offset + 12 * 64 + i] = 1.0f;
        if (b->repetitions >= 3)
            for (int i = 0; i < 64; i++)
                out[offset + 13 * 64 + i] = 1.0f;

        if (b->ply > 0) {
            pop(b);
            removed++;
        } else {
            break;
        }
    }

    /* restore board */
    /* (need zt + stored moves to re-push — or store Moves before popping) */

    /* global planes */
    int base = 14 * history_steps * 64;
    /* castling: use original board state (before pops, restore first) */
    /* ... */
}
```

The tricky part is restoring the board after popping. Two approaches:
1. Save the moves before popping, then re-push them in reverse.
2. Snapshot the entire Board struct before encoding (memcpy), encode from the copy, discard copy.

Option 2 is simpler but Board is ~24KB (due to history array). Option 1 is cleaner if you track the popped move count and re-push.

---

## 3. Move-to-Policy-Index

C equivalent of `encode.py:_move_to_index()`. Maps a `Move` to an index in the `[0, 4168)` policy vector.

### Encoding scheme
```
POLICY_DIM = 64*64 + 3*8*3 = 4096 + 72 = 4168
```

**Normal moves (including queen promotions):**
```
index = from_sq * 64 + to_sq
```
Where `from_sq` and `to_sq` are flipped vertically when black is to move (`row = 7 - row`).

**Underpromotions (knight, bishop, rook):**
```
piece_index = 0 (knight), 1 (bishop), 2 (rook)
from_file   = file of from_sq (after flip)
direction   = file_of(to_sq) - file_of(from_sq) + 1   → 0=left, 1=forward, 2=right

index = 4096 + piece_index * 24 + from_file * 3 + direction
```

### Suggested signature
```c
#define POLICY_DIM 4168

int move_to_index(Move move, int side);
```
- `side`: the side making the move (for vertical flip). 1 = flip.
- Returns index in `[0, POLICY_DIM)`.

### Implementation
```c
int move_to_index(Move move, int side) {
    int fr = move.from / 8, fc = move.from % 8;
    int tr = move.to / 8,   tc = move.to % 8;
    if (side) { fr = 7 - fr; tr = 7 - tr; }
    int from_sq = fr * 8 + fc;
    int to_sq   = tr * 8 + tc;

    if (move.promotion != EMPTY && move.promotion % 6 != WQ) {
        int ptype = move.promotion % 6;
        int piece_index;
        if      (ptype == WN) piece_index = 0;
        else if (ptype == WB) piece_index = 1;
        else                  piece_index = 2;  /* WR */

        int direction = tc - fc + 1;
        return 4096 + piece_index * 24 + fc * 3 + direction;
    }

    return from_sq * 64 + to_sq;
}
```

---

## 4. FEN Parsing / Generation

Needed if the C worker serves HTTP or communicates positions as strings.

### `void board_from_fen(Board *b, const ZobristTable *zt, const char *fen)`

Parse standard FEN string with 6 fields:
1. **Piece placement**: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR`
   - Walk rank 8 down to rank 1, file a to h
   - Digits skip empty squares, letters map to pieces
   - Piece char map: `pnbrqk` → `BP,BN,BB,BR,BQ,BK`, uppercase → white
2. **Side to move**: `w` → 0, `b` → 1
3. **Castling**: `-` or combo of `KQkq` → set bits in `b->castling`
4. **En passant**: `-` or square like `e3` → `b->ep_file = file`, or -1
5. **Halfmove clock**: integer → `b->halfmove`
6. **Fullmove number**: integer → `b->fullmove`

After parsing, set `b->ply = 0`, `b->repetitions = 1`, compute `b->hash = hash(zt, b)`.

### `void board_to_fen(const Board *b, char *fen)`

Write FEN string into caller-provided buffer (at least 92 bytes).

Reverse of parsing:
- Walk ranks 7 down to 0, count consecutive empties as digits
- Append side, castling, ep square, halfmove, fullmove separated by spaces

---

## 5. Move-to-UCI String

### `void move_to_uci(Move move, char *out)`

Write UCI string into caller-provided buffer (at least 6 bytes).

```c
void move_to_uci(Move move, char *out) {
    out[0] = 'a' + move.from % 8;
    out[1] = '1' + move.from / 8;
    out[2] = 'a' + move.to % 8;
    out[3] = '1' + move.to / 8;
    if (move.promotion != EMPTY) {
        out[4] = "  nbrq"[move.promotion % 6];
        out[5] = '\0';
    } else {
        out[4] = '\0';
    }
}
```
