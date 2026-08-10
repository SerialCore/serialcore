/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_IOJSON
#define SERIALCORE_IOJSON

#include <serialcore/cJSON.h>

/* Read entire file into a malloc'd NUL-terminated buffer. NULL on error. */
char *iojson_read_file(const char *filename);

/* Write a text buffer to file. Returns 0 on success, -1 on error. */
int iojson_write_file(const char *filename, const char *text);

/* Parse a JSON file into a cJSON tree. Caller must cJSON_Delete. NULL on error. */
cJSON *iojson_parse_file(const char *filename);

/* Pretty-print a cJSON tree to file. Returns 0 on success, -1 on error. */
int iojson_write_cjson(const char *filename, const cJSON *root);

/* Typed field readers. Return 0 on success, -1 if missing/wrong type. */
int iojson_get_number(const cJSON *obj, const char *key, double *out);
int iojson_get_int(const cJSON *obj, const char *key, int *out);
int iojson_get_float(const cJSON *obj, const char *key, float *out);
int iojson_get_string(const cJSON *obj, const char *key, const char **out);
cJSON *iojson_get_object(const cJSON *obj, const char *key);
cJSON *iojson_get_array(const cJSON *obj, const char *key);

/* Convert between JSON number arrays and float buffers.
 * iojson_array_to_floats returns 0 on success, -1 on size/type mismatch.
 * iojson_floats_to_array returns a new cJSON array (or NULL on OOM). */
int iojson_array_to_floats(const cJSON *arr, float *out, int n);
cJSON *iojson_floats_to_array(const float *data, int n);

#endif
