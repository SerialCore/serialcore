/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/iobin.h>

#include <stdio.h>

int iobin_write_file(const char *filename, const void *data, size_t nbytes)
{
    FILE *f;
    size_t written;

    if (!filename || (!data && nbytes > 0)) return -1;

    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "iobin: cannot open %s for write\n", filename);
        return -1;
    }

    written = nbytes ? fwrite(data, 1, nbytes, f) : 0;
    fclose(f);

    if (written != nbytes) {
        fprintf(stderr, "iobin: short write to %s\n", filename);
        return -1;
    }
    return 0;
}

int iobin_read_file(const char *filename, void *data, size_t nbytes)
{
    FILE *f;
    size_t nread;

    if (!filename || (!data && nbytes > 0)) return -1;

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "iobin: cannot open %s\n", filename);
        return -1;
    }

    nread = nbytes ? fread(data, 1, nbytes, f) : 0;
    fclose(f);

    if (nread != nbytes) {
        fprintf(stderr, "iobin: short read from %s\n", filename);
        return -1;
    }
    return 0;
}

int iobin_write_floats(const char *filename, const float *data, int n)
{
    if (n < 0) return -1;
    return iobin_write_file(filename, data, (size_t)n * sizeof(float));
}

int iobin_read_floats(const char *filename, float *data, int n)
{
    if (n < 0) return -1;
    return iobin_read_file(filename, data, (size_t)n * sizeof(float));
}
