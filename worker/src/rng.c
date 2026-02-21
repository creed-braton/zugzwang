#include "rng.h"

static uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

void rng_seed(Rng *rng, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

uint64_t rng_next_u64(Rng *rng) {
    uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
    uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);
    return result;
}

float rng_next_float(Rng *rng) {
    return (float)(rng_next_u64(rng) >> 11) * (1.0f / 9007199254740992.0f);
}
