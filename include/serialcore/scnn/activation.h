/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SCNN_ACTIVATION
#define SERIALCORE_SCNN_ACTIVATION

#include <math.h>

typedef enum ACTIVATION {
    Sigmoid,
    Tanh,
    ReLU,
    Leaky,
    SELU,
    GELU,
    Swish,
    Sin,
    Cos,
}ACTIVATION_T;

float activate(float x, ACTIVATION_T a);
float gradient(float x, ACTIVATION_T a);
char *activation_name(ACTIVATION_T a);
ACTIVATION_T activation_type(char *s);

static inline float af_Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
static inline float gf_Sigmoid(float x)
{
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

static inline float af_Tanh(float x)
{
    return tanh(x);
}
static inline float gf_Tanh(float x)
{
    return 1 - tanh(x) * tanh(x);
}

static inline float af_ReLU(float x)
{
    return (x >= 0) ? x : 0;
}
static inline float gf_ReLU(float x)
{
    return (x >= 0) ? 1 : 0;
}

static inline float af_Leaky(float x)
{
    return (x >= 0) ? x : 0.01*x;
}
static inline float gf_Leaky(float x)
{
    return (x >= 0) ? 1 : 0.01;
}

static inline float af_SELU(float x)
{
    return (x >= 0) ? 1.0507*x : 1.0507*1.67326*(exp(x)-1);
}
static inline float gf_SELU(float x)
{
    return (x >= 0) ? 1.0507 : 1.0507*1.67326*exp(x);
}

static inline float af_GELU(float x)
{
    return x * (1 / (1 + exp(-1.702*x)));
}
static inline float gf_GELU(float x)
{
    return 1 / (1 + exp(-1.702*x)) + 1.702 * x * (1 / (1 + exp(-1.702*x))) * (1 - (1 / (1 + exp(-1.702*x))));
}

static inline float af_Swish(float x)
{
    return x / (1 + exp(-x));
}
static inline float gf_Swish(float x)
{
    return 1 / (1 + exp(-x)) + x * exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

static inline float af_Sin(float x)
{
    return sin(x);
}
static inline float gf_Sin(float x)
{
    return cos(x);
}

static inline float af_Cos(float x)
{
    return cos(x);
}
static inline float gf_Cos(float x)
{
    return -sin(x);
}

#endif
