#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t s[4];
} Rng;

void     rng_seed(Rng *rng, uint64_t seed);
uint64_t rng_next_u64(Rng *rng);
float    rng_next_float(Rng *rng);

#endif
