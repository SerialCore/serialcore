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
 *   The generator must be seeded before first use of next().
 *   The original code expects you to set the internal state.
 *   In practice, call next() / jump() a number of times derived from
 *   your seed value to get different starting points.
 */

uint64_t next(void);

void jump(void);
void long_jump(void);
void jump_ce(uint64_t c, uint32_t e);
void jump_n(const uint64_t jump[4]);

#endif
