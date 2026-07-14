/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/sonn/activaton.h>

#include <stdio.h>
#include <string.h>

float activaton_func(float x, activaton_t a)
{
    switch (a) {
        case Sigmoid:
            return af_Sigmoid(x);
        case Tanh:
            return af_Tanh(x);
        case ReLU:
            return af_ReLU(x);
        case Leaky:
            return af_Leaky(x);
        case SELU:
            return af_SELU(x);
        case GELU:
            return af_GELU(x);
        case Swish:
            return af_Swish(x);
        case Sin:
            return af_Sin(x);
        case Cos:
            return af_Cos(x);

        default:
            return af_GELU(x);
    }
}

float gradient_func(float x, activaton_t a)
{
    switch (a) {
        case Sigmoid:
            return gf_Sigmoid(x);
        case Tanh:
            return gf_Tanh(x);
        case ReLU:
            return gf_ReLU(x);
        case Leaky:
            return gf_Leaky(x);
        case SELU:
            return gf_SELU(x);
        case GELU:
            return gf_GELU(x);
        case Swish:
            return gf_Swish(x);
        case Sin:
            return gf_Sin(x);
        case Cos:
            return gf_Cos(x);

        default:
            return gf_GELU(x);
    }
}

char *activaton_name(activaton_t a)
{
    switch (a) {
        case Sigmoid:
            return "Sigmoid";
        case Tanh:
            return "Tanh";
        case ReLU:
            return "ReLU";
        case Leaky:
            return "Leaky";
        case SELU:
            return "SELU";
        case GELU:
            return "GELU";
        case Swish:
            return "Swish";
        case Sin:
            return "Sin";
        case Cos:
            return "Cos";

        default:
            return "GELU";
    }
}

activaton_t activaton_type(char *s)
{
    if (strcmp(s, "Sigmoid") == 0) return Sigmoid;
    if (strcmp(s, "Tanh") == 0) return Tanh;
    if (strcmp(s, "ReLU") == 0) return ReLU;
    if (strcmp(s, "Leaky") == 0) return Leaky;
    if (strcmp(s, "SELU") == 0) return SELU;
    if (strcmp(s, "GELU") == 0) return GELU;
    if (strcmp(s, "Swish") == 0) return Swish;
    if (strcmp(s, "Sin") == 0) return Sin;
    if (strcmp(s, "Cos") == 0) return Cos;

    return GELU;
}
