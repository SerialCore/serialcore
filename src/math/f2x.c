/* f2x.c - Minimal stub to satisfy #include "f2x.c" from xoshiro256ss.c
 *
 * This file is intended to be #included by xoshiro256ss.c .
 * When compiled standalone (via Makefile wildcard), the functions are
 * made static so they don't cause multiple-definition errors at link.
 */

#include <stdint.h>

#define POLY_WORDS 4

static void f2x_jumppoly_ce(uint64_t c, uint32_t e, uint64_t *poly) {
    (void)c; (void)e;
    for (int i = 0; i < POLY_WORDS; i++) poly[i] = 0;
}

static void f2x_jumppoly_n(const uint64_t *jump, int words, uint64_t *poly) {
    (void)jump; (void)words;
    for (int i = 0; i < POLY_WORDS; i++) poly[i] = 0;
}
