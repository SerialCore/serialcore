/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_IOBIN
#define SERIALCORE_IOBIN

#include <stddef.h>

/* Binary blob I/O. Returns 0 on success, -1 on error. */
int iobin_write_file(const char *filename, const void *data, size_t nbytes);
int iobin_read_file(const char *filename, void *data, size_t nbytes);

/* Contiguous float array helpers on top of the binary blob I/O. */
int iobin_write_floats(const char *filename, const float *data, int n);
int iobin_read_floats(const char *filename, float *data, int n);

#endif
