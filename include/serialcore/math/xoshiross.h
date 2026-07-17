/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_MATH_XOSHIROSS
#define SERIALCORE_MATH_XOSHIROSS

#include <stdint.h>

/* API provided by xoshiro256ss.c (original author implementation)
 *
 * This header only declares the public symbols.
 * The implementation lives in src/math/xoshiro256ss.c
 *
 * Seeding:
 *   You MUST call xoshiro_seed() with a non-zero value before using next().
 *   The original xoshiro256** authors recommend using splitmix64 to
 *   initialize the internal state from a 64-bit seed.
 *
 *   Example:
 *       xoshiro_seed(42);
 *       uint64_t r = next();
 */

uint64_t next(void);

void jump(void);
void long_jump(void);
void jump_ce(uint64_t c, uint32_t e);
void jump_n(const uint64_t jump[4]);

/* Seed the xoshiro256** generator.
 * Call this once before any call to next(), jump(), etc.
 * A seed of 0 will result in a bad (all-zero) internal state.
 */
void xoshiro_seed(uint64_t seed);

#endif
