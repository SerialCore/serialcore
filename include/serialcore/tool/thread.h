/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef SERIALCORE_TOOL_THREAD
#define SERIALCORE_TOOL_THREAD

#include <unistd.h>
#include <pthread.h>

struct argsThread;
typedef struct argsThread argsThread_t;

/* entry function, argsThread_t *args = void *args */
typedef void (*thread_fun)(argsThread_t *args);

struct argsThread {
	int index;					/* index of this thread */
	int length;					/* total number of tasks */
	int lock;					/* the lock state */
	pthread_mutex_t *mutex;		/* mutex for claiming tasks */
	int *next;					/* next task index to be claimed */
	thread_fun f;				/* function pointer */
	void *args;					/* user-defined arguements */
};

/* Get number of cores */
int getNumCores();

/* Thread running function */
void *thread_run(void *args);

/* Load threads, 
 * fun: entry function
 * args: args for function
 * length: number of tasks
 * threads: number of threads to be created
 */
void thread_load(thread_fun fun, void *args, int length, int threads);

#endif