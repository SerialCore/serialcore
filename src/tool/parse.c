/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/tool/parse.h>
#include <serialcore/tool/cJSON.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_input_file(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    char *buffer;
    long size;

    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(1);
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "Cannot seek %s\n", filename);
        exit(1);
    }

    size = ftell(f);
    if (size < 0) {
        fclose(f);
        fprintf(stderr, "Cannot determine size of %s\n", filename);
        exit(1);
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "Cannot rewind %s\n", filename);
        exit(1);
    }

    buffer = (char *)malloc((size_t)size + 1);
    if (!buffer) {
        fclose(f);
        fprintf(stderr, "Out of memory while reading %s\n", filename);
        exit(1);
    }

    if (fread(buffer, 1, (size_t)size, f) != (size_t)size) {
        free(buffer);
        fclose(f);
        fprintf(stderr, "Cannot read %s\n", filename);
        exit(1);
    }

    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

static cJSON *read_object_item(const cJSON *object, const char *key)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(object, key);
    if (!cJSON_IsObject(item)) {
        fprintf(stderr, "Missing or invalid object: %s\n", key);
        exit(1);
    }
    return item;
}

static cJSON *read_string_item(const cJSON *object, const char *key)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(object, key);
    if (!cJSON_IsString(item) || item->valuestring == NULL) {
        fprintf(stderr, "Missing or invalid string: %s\n", key);
        exit(1);
    }
    return item;
}

static cJSON *read_number_item(const cJSON *object, const char *key)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(object, key);
    if (!cJSON_IsNumber(item)) {
        fprintf(stderr, "Missing or invalid number: %s\n", key);
        exit(1);
    }
    return item;
}

static cJSON *read_array_item(const cJSON *object, const char *key)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(object, key);
    if (!cJSON_IsArray(item)) {
        fprintf(stderr, "Missing or invalid array: %s\n", key);
        exit(1);
    }
    return item;
}

static void parse_task_string(const char *value, argsInput_t *input)
{
    if (strcmp(value, "SPECTRA") == 0) input->task = TASK_SPECTRA;
    else if (strcmp(value, "DECAY3P0") == 0) input->task = TASK_DECAY3P0;
    else if (strcmp(value, "COUPLCHN") == 0) input->task = TASK_COUPLCHN;
    else if (strcmp(value, "SCATTER") == 0) input->task = TASK_SCATTER;
    else {
        fprintf(stderr, "Unknown task: %s\n", value);
        exit(1);
    }
}

static void parse_model_section(const cJSON *root, argsInput_t *input)
{
    cJSON *model_json = read_object_item(root, "model");
    const char *type = read_string_item(model_json, "type")->valuestring;
    const char *param = read_string_item(model_json, "param")->valuestring;

    if (strcmp(type, "GISTRING") == 0) input->model = MODEL_GISTRING;
    else if (strcmp(type, "GISCREEN") == 0) input->model = MODEL_GISCREEN;
    else {
        fprintf(stderr, "Unknown model type: %s\n", type);
        exit(1);
    }

    if (input->model == MODEL_GISTRING) {
        if (strcmp(param, "GISTRING_MESON") == 0) input->param = PARAM_GISTRING_MESON;
        else if (strcmp(param, "GISTRING_CUSTOM") == 0) input->param = PARAM_GISTRING_CUSTOM;
        else {
            fprintf(stderr, "Unknown GISTRING parameter set: %s\n", param);
            exit(1);
        }
    }

    if (input->model == MODEL_GISCREEN) {
        if (strcmp(param, "GISCREEN_MESON") == 0) input->param = PARAM_GISCREEN_MESON;
        else if (strcmp(param, "GISCREEN_BBBAR") == 0) input->param = PARAM_GISCREEN_BBBAR;
        else if (strcmp(param, "GISCREEN_BCBAR") == 0) input->param = PARAM_GISCREEN_BCBAR;
        else if (strcmp(param, "GISCREEN_BSBAR") == 0) input->param = PARAM_GISCREEN_BSBAR;
        else if (strcmp(param, "GISCREEN_CCBAR") == 0) input->param = PARAM_GISCREEN_CCBAR;
        else if (strcmp(param, "GISCREEN_CSBAR") == 0) input->param = PARAM_GISCREEN_CSBAR;
        else if (strcmp(param, "GISCREEN_CUSTOM") == 0) input->param = PARAM_GISCREEN_CUSTOM;
        else {
            fprintf(stderr, "Unknown GISCREEN parameter set: %s\n", param);
            exit(1);
        }
    }

    if (input->param == PARAM_GISTRING_CUSTOM || input->param == PARAM_GISCREEN_CUSTOM) {
        const char *file = read_string_item(model_json, "file")->valuestring;
        if (strlen(file) > 0) {
            strncpy(input->param_file, file, 256);
            input->param_file[255] = '\0';
        }
        else {
            fprintf(stderr, "Missing parameter file for custom parameter set\n");
            exit(1);
        }
    }
}

static void parse_system_section(const cJSON *root, argsInput_t *input)
{
    cJSON *system_json = read_object_item(root, "system");
    const char *type = read_string_item(system_json, "type")->valuestring;

    if (strcmp(type, "MESON") == 0) input->system = SYSTEM_MESON;
    else if (strcmp(type, "BARYON") == 0) input->system = SYSTEM_BARYON;
    else {
        fprintf(stderr, "Unsupported system type: %s\n", type);
        exit(1);
    }

    if (input->system == SYSTEM_MESON) {
        input->f1 = read_number_item(system_json, "f1")->valueint;
        input->f2 = read_number_item(system_json, "f2")->valueint;
        input->S = read_number_item(system_json, "S")->valuedouble;
        input->L = read_number_item(system_json, "L")->valuedouble;
        input->J = read_number_item(system_json, "J")->valuedouble;
    }
    else if (input->system == SYSTEM_BARYON) {
        input->f1 = read_number_item(system_json, "f1")->valueint;
        input->f2 = read_number_item(system_json, "f2")->valueint;
        input->f3 = read_number_item(system_json, "f3")->valueint;
        input->J = read_number_item(system_json, "J")->valuedouble;
        input->P = read_number_item(system_json, "P")->valueint;
        input->f12 = read_number_item(system_json, "sym12")->valueint;
        input->Lmax = read_number_item(system_json, "Lmax")->valueint;
    }
}

static void parse_basis_section(const cJSON *root, argsInput_t *input)
{
    cJSON *basis_json = read_object_item(root, "basis");
    const char *type = read_string_item(basis_json, "type")->valuestring;

    if (strcmp(type, "GEM") == 0) input->orbit = ORBIT_GEM;
    else if (strcmp(type, "CRG") == 0) input->orbit = ORBIT_CRG;
    else if (strcmp(type, "SHO") == 0) input->orbit = ORBIT_SHO;
    else {
        fprintf(stderr, "Unknown basis type: %s\n", type);
        exit(1);
    }

    if (input->orbit == ORBIT_GEM || input->orbit == ORBIT_CRG) {
        input->nmax = read_number_item(basis_json, "nmax")->valueint;
        input->rmax = read_number_item(basis_json, "rmax")->valuedouble;
        input->rmin = read_number_item(basis_json, "rmin")->valuedouble;
    }

    if (input->orbit == ORBIT_CRG) {
        input->omega = read_number_item(basis_json, "omega")->valuedouble;
    }

    if (input->orbit == ORBIT_SHO) {
        input->beta = read_number_item(basis_json, "beta")->valuedouble;
    }
}

static void parse_print_section(const cJSON *root, argsInput_t *input)
{
    cJSON *print_json = read_object_item(root, "print");
    const char *pot_str = read_string_item(print_json, "pot")->valuestring;
    const char *wfn_str = read_string_item(print_json, "wfn")->valuestring;

    input->print_pot = (strcmp(pot_str, "true") == 0) ? 1 : 0;
    input->print_wfn = (strcmp(wfn_str, "true") == 0) ? 1 : 0;
}

void parse_input_file(const char *filename, argsInput_t *input)
{
    char *json_text = read_input_file(filename);
    const char *parse_error = NULL;
    cJSON *root = cJSON_Parse(json_text);

    if (!root) {
        parse_error = cJSON_GetErrorPtr();
        fprintf(stderr, "Invalid JSON input in %s", filename);
        if (parse_error) fprintf(stderr, " near: %.40s", parse_error);
        fprintf(stderr, "\n");
        free(json_text);
        exit(1);
    }

    strncpy(input->project, read_string_item(root, "project")->valuestring, 256);
    input->project[255] = '\0';

    parse_task_string(read_string_item(root, "task")->valuestring, input);
    parse_model_section(root, input);
    parse_system_section(root, input);
    parse_basis_section(root, input);
    parse_print_section(root, input);

    cJSON_Delete(root);
    free(json_text);
}

void parse_param_GISTRING(const char *filename, argsGIModel_t *args_model)
{
    char *json_text = read_input_file(filename);
    const char *parse_error = NULL;
    cJSON *root = cJSON_Parse(json_text);
    cJSON *param_json;

    if (!root) {
        parse_error = cJSON_GetErrorPtr();
        fprintf(stderr, "Invalid GISTRING parameter JSON in %s", filename);
        if (parse_error) fprintf(stderr, " near: %.40s", parse_error);
        fprintf(stderr, "\n");
        free(json_text);
        exit(1);
    }

    param_json = read_object_item(root, "param");
    args_model->mn = read_number_item(param_json, "mn")->valuedouble;
    args_model->ms = read_number_item(param_json, "ms")->valuedouble;
    args_model->mc = read_number_item(param_json, "mc")->valuedouble;
    args_model->mb = read_number_item(param_json, "mb")->valuedouble;
    args_model->b = read_number_item(param_json, "b")->valuedouble;
    args_model->c = read_number_item(param_json, "c")->valuedouble;
    args_model->sigma_0 = read_number_item(param_json, "sigma_0")->valuedouble;
    args_model->s = read_number_item(param_json, "s")->valuedouble;
    args_model->epsilon_cont = read_number_item(param_json, "epsilon_cont")->valuedouble;
    args_model->epsilon_sov = read_number_item(param_json, "epsilon_sov")->valuedouble;
    args_model->epsilon_sos = read_number_item(param_json, "epsilon_sos")->valuedouble;
    args_model->epsilon_tens = read_number_item(param_json, "epsilon_tens")->valuedouble;
    args_model->mu = 0.0;

    cJSON_Delete(root);
    free(json_text);
}

void parse_param_GISCREEN(const char *filename, argsGIModel_t *args_model)
{
    char *json_text = read_input_file(filename);
    const char *parse_error = NULL;
    cJSON *root = cJSON_Parse(json_text);
    cJSON *param_json;

    if (!root) {
        parse_error = cJSON_GetErrorPtr();
        fprintf(stderr, "Invalid GISCREEN parameter JSON in %s", filename);
        if (parse_error) fprintf(stderr, " near: %.40s", parse_error);
        fprintf(stderr, "\n");
        free(json_text);
        exit(1);
    }

    param_json = read_object_item(root, "param");
    args_model->mn = read_number_item(param_json, "mn")->valuedouble;
    args_model->ms = read_number_item(param_json, "ms")->valuedouble;
    args_model->mc = read_number_item(param_json, "mc")->valuedouble;
    args_model->mb = read_number_item(param_json, "mb")->valuedouble;
    args_model->b = read_number_item(param_json, "b")->valuedouble;
    args_model->mu = read_number_item(param_json, "mu")->valuedouble;
    args_model->c = read_number_item(param_json, "c")->valuedouble;
    args_model->sigma_0 = read_number_item(param_json, "sigma_0")->valuedouble;
    args_model->s = read_number_item(param_json, "s")->valuedouble;
    args_model->epsilon_cont = read_number_item(param_json, "epsilon_cont")->valuedouble;
    args_model->epsilon_sov = read_number_item(param_json, "epsilon_sov")->valuedouble;
    args_model->epsilon_sos = read_number_item(param_json, "epsilon_sos")->valuedouble;
    args_model->epsilon_tens = read_number_item(param_json, "epsilon_tens")->valuedouble;

    cJSON_Delete(root);
    free(json_text);
}

void parse_meson_state(const char *filename, array_t *mass, array_t *radius, matrix_t *eigenvector)
{
    if (filename == NULL || mass == NULL || radius == NULL || eigenvector == NULL) {
        fprintf(stderr, "Error: Invalid input parameters to parse_meson_state()\n");
        exit(1);
    }

    if (mass->value == NULL || radius->value == NULL || eigenvector->value == NULL) {
        fprintf(stderr, "Error: Arrays must be pre-allocated (non-NULL values)\n");
        exit(1);
    }

    char *json_text = read_input_file(filename);
    const char *parse_error = NULL;
    cJSON *root = cJSON_Parse(json_text);

    if (!root) {
        parse_error = cJSON_GetErrorPtr();
        fprintf(stderr, "Invalid JSON in state file %s", filename);
        if (parse_error) fprintf(stderr, " near: %.40s", parse_error);
        fprintf(stderr, "\n");
        free(json_text);
        exit(1);
    }

    /* Get states array using read_array_item */
    cJSON *states_json = read_array_item(root, "states");
    int num_states = cJSON_GetArraySize(states_json);
    if (num_states <= 0) {
        fprintf(stderr, "Error: Empty states array in %s\n", filename);
        cJSON_Delete(root);
        free(json_text);
        exit(1);
    }

    /* Verify array dimensions match */
    if (num_states != mass->len || num_states != radius->len || num_states != eigenvector->row) {
        fprintf(stderr, "Error: State count mismatch (expected %d, got %d states in file)\n", 
                mass->len, num_states);
        cJSON_Delete(root);
        free(json_text);
        exit(1);
    }

    /* Get eigenvector length from first state */
    cJSON *first_state = cJSON_GetArrayItem(states_json, 0);
    cJSON *first_eigenvector = read_array_item(first_state, "eigenvector");
    int eigenvector_len = cJSON_GetArraySize(first_eigenvector);
    if (eigenvector_len != eigenvector->col) {
        fprintf(stderr, "Error: Eigenvector length mismatch (expected %d, got %d)\n",
                eigenvector->col, eigenvector_len);
        cJSON_Delete(root);
        free(json_text);
        exit(1);
    }

    /* Parse states data */
    for (int n = 0; n < num_states; n++) {
        cJSON *state = cJSON_GetArrayItem(states_json, n);
        if (!cJSON_IsObject(state)) {
            fprintf(stderr, "Error: Invalid state object at index %d\n", n);
            cJSON_Delete(root);
            free(json_text);
            exit(1);
        }

        /* Parse mass */
        cJSON *mass_item = read_number_item(state, "mass");
        mass->value[n] = mass_item->valuedouble;

        /* Parse rms_radius */
        cJSON *radius_item = read_number_item(state, "rms_radius");
        radius->value[n] = radius_item->valuedouble;

        /* Parse eigenvector using read_array_item */
        cJSON *eigen_array = read_array_item(state, "eigenvector");
        int current_eigen_len = cJSON_GetArraySize(eigen_array);
        if (current_eigen_len != eigenvector_len) {
            fprintf(stderr, "Error: Eigenvector length mismatch in state %d (expected %d, got %d)\n",
                    n, eigenvector_len, current_eigen_len);
            cJSON_Delete(root);
            free(json_text);
            exit(1);
        }

        /* Extract eigenvector coefficients */
        for (int d = 0; d < eigenvector_len; d++) {
            cJSON *coeff = cJSON_GetArrayItem(eigen_array, d);
            if (!cJSON_IsNumber(coeff)) {
                fprintf(stderr, "Error: Invalid eigenvector coefficient at state %d, index %d\n", n, d);
                cJSON_Delete(root);
                free(json_text);
                exit(1);
            }
            eigenvector->value[n][d] = coeff->valuedouble;
        }
    }

    cJSON_Delete(root);
    free(json_text);
}
