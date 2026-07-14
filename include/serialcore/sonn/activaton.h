/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_SONN_ACTIVATON
#define SERIALCORE_SONN_ACTIVATON

#include <math.h>

/* Supported activation function types. */
typedef enum activaton {
    Sigmoid,
    Tanh,
    ReLU,
    Leaky,
    SELU,
    GELU,
    Swish,
    Sin,
    Cos,
} activaton_t;

/* Compute the activation output for input x using the specified activation type. */
float activaton_func(float x, activaton_t a);

/* Compute the gradient (derivative) of the activation for input x. */
float gradient_func(float x, activaton_t a);

/* Return a human-readable string name for the given activation type. */
char *activaton_name(activaton_t a);

/* Convert a string name to the corresponding activation type. */
activaton_t activaton_type(char *s);

/* Sigmoid activation and its derivative. */
static inline float af_Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
static inline float gf_Sigmoid(float x)
{
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

/* Hyperbolic tangent (Tanh) activation and its derivative. */
static inline float af_Tanh(float x)
{
    return tanh(x);
}
static inline float gf_Tanh(float x)
{
    return 1 - tanh(x) * tanh(x);
}

/* Rectified Linear Unit (ReLU) activation and its derivative. */
static inline float af_ReLU(float x)
{
    return (x >= 0) ? x : 0;
}
static inline float gf_ReLU(float x)
{
    return (x >= 0) ? 1 : 0;
}

/* Leaky ReLU activation and its derivative. */
static inline float af_Leaky(float x)
{
    return (x >= 0) ? x : 0.01*x;
}
static inline float gf_Leaky(float x)
{
    return (x >= 0) ? 1 : 0.01;
}

/* Scaled Exponential Linear Unit (SELU) activation and its derivative. */
static inline float af_SELU(float x)
{
    return (x >= 0) ? 1.0507*x : 1.0507*1.67326*(exp(x)-1);
}
static inline float gf_SELU(float x)
{
    return (x >= 0) ? 1.0507 : 1.0507*1.67326*exp(x);
}

/* Gaussian Error Linear Unit (GELU) activation and its derivative. */
static inline float af_GELU(float x)
{
    return x * (1 / (1 + exp(-1.702*x)));
}
static inline float gf_GELU(float x)
{
    return 1 / (1 + exp(-1.702*x)) + 1.702 * x * (1 / (1 + exp(-1.702*x))) * (1 - (1 / (1 + exp(-1.702*x))));
}

/* Swish (SiLU) activation and its derivative. */
static inline float af_Swish(float x)
{
    return x / (1 + exp(-x));
}
static inline float gf_Swish(float x)
{
    return 1 / (1 + exp(-x)) + x * exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

/* Sine activation and its derivative. */
static inline float af_Sin(float x)
{
    return sin(x);
}
static inline float gf_Sin(float x)
{
    return cos(x);
}

/* Cosine activation and its derivative. */
static inline float af_Cos(float x)
{
    return cos(x);
}
static inline float gf_Cos(float x)
{
    return -sin(x);
}

#endif
