/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/iojson.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *iojson_read_file(const char *filename)
{
    FILE *f;
    char *buf = NULL;
    long size;

    if (!filename) return NULL;

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "iojson: cannot open %s\n", filename);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) == 0 && (size = ftell(f)) >= 0 &&
        fseek(f, 0, SEEK_SET) == 0 &&
        (buf = (char *)malloc((size_t)size + 1)) != NULL &&
        fread(buf, 1, (size_t)size, f) == (size_t)size) {
        buf[size] = '\0';
        fclose(f);
        return buf;
    }

    fprintf(stderr, "iojson: cannot read %s\n", filename);
    free(buf);
    fclose(f);
    return NULL;
}

int iojson_write_file(const char *filename, const char *text)
{
    FILE *f;
    size_t n, written;

    if (!filename || !text) return -1;

    f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "iojson: cannot open %s for write\n", filename);
        return -1;
    }

    n = strlen(text);
    written = fwrite(text, 1, n, f);
    fclose(f);

    if (written != n) {
        fprintf(stderr, "iojson: short write to %s\n", filename);
        return -1;
    }
    return 0;
}

cJSON *iojson_parse_file(const char *filename)
{
    char *json_text;
    cJSON *root;
    const char *parse_error;

    json_text = iojson_read_file(filename);
    if (!json_text) return NULL;

    root = cJSON_Parse(json_text);
    if (!root) {
        parse_error = cJSON_GetErrorPtr();
        fprintf(stderr, "iojson: invalid JSON in %s", filename);
        if (parse_error) fprintf(stderr, " near: %.40s", parse_error);
        fprintf(stderr, "\n");
        free(json_text);
        return NULL;
    }

    free(json_text);
    return root;
}

int iojson_write_cjson(const char *filename, const cJSON *root)
{
    char *text;
    int rc;

    if (!filename || !root) return -1;

    text = cJSON_Print(root);
    if (!text) {
        fprintf(stderr, "iojson: cJSON_Print failed\n");
        return -1;
    }

    rc = iojson_write_file(filename, text);
    cJSON_free(text);
    return rc;
}

int iojson_get_number(const cJSON *obj, const char *key, double *out)
{
    cJSON *item;

    if (!obj || !key || !out) return -1;
    item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsNumber(item)) return -1;
    *out = item->valuedouble;
    return 0;
}

int iojson_get_int(const cJSON *obj, const char *key, int *out)
{
    double v;
    if (iojson_get_number(obj, key, &v) != 0) return -1;
    *out = (int)v;
    return 0;
}

int iojson_get_float(const cJSON *obj, const char *key, float *out)
{
    double v;
    if (iojson_get_number(obj, key, &v) != 0) return -1;
    *out = (float)v;
    return 0;
}

int iojson_get_string(const cJSON *obj, const char *key, const char **out)
{
    cJSON *item;

    if (!obj || !key || !out) return -1;
    item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsString(item) || item->valuestring == NULL) return -1;
    *out = item->valuestring;
    return 0;
}

cJSON *iojson_get_object(const cJSON *obj, const char *key)
{
    cJSON *item;

    if (!obj || !key) return NULL;
    item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsObject(item) ? item : NULL;
}

cJSON *iojson_get_array(const cJSON *obj, const char *key)
{
    cJSON *item;

    if (!obj || !key) return NULL;
    item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsArray(item) ? item : NULL;
}

int iojson_array_to_floats(const cJSON *arr, float *out, int n)
{
    int i;

    if (!arr || !out || n < 0) return -1;
    if (!cJSON_IsArray(arr)) return -1;
    if (cJSON_GetArraySize(arr) != n) return -1;

    for (i = 0; i < n; i++) {
        cJSON *item = cJSON_GetArrayItem(arr, i);
        if (!cJSON_IsNumber(item)) return -1;
        out[i] = (float)item->valuedouble;
    }
    return 0;
}

cJSON *iojson_floats_to_array(const float *data, int n)
{
    cJSON *arr;
    int i;

    if (!data || n < 0) return NULL;
    arr = cJSON_CreateArray();
    if (!arr) return NULL;

    for (i = 0; i < n; i++) {
        cJSON *num = cJSON_CreateNumber((double)data[i]);
        if (!num) {
            cJSON_Delete(arr);
            return NULL;
        }
        cJSON_AddItemToArray(arr, num);
    }
    return arr;
}
