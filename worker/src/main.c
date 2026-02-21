#include "chess.h"

int main(void) {
  init_zobrist_table();
  init_lookup_tables();

  return 0;
}
