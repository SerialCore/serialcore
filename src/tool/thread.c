/*
 * Copyright (C) 2026, Wen-Xuan Zhang <serialcore@outlook.com>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <serialcore/tool/thread.h>

#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

int getNumCores()
{
    return sysconf(_SC_NPROCESSORS_ONLN);
}

void *thread_run(void *args)
{
	argsThread_t *args_thread = (argsThread_t *)args;
	int length = args_thread->length;
	int *next = args_thread->next;
	pthread_mutex_t *mutex = args_thread->mutex;
	thread_fun f = args_thread->f;
	int pos;	/* the task index just claimed */
	
	/* try up to length times */
    for (int i = 0; i < length; i++) {
		pthread_mutex_lock(mutex);
		if (*next < length) {
			pos = *next;			/* claim a task */
			*next = *next + 1;		/* update next task index */
		}
        else {
			pos = -1;				/* no task to be claimed */
		}
		pthread_mutex_unlock(mutex);
		if (pos >= 0) {
			f(&args_thread[pos]);	/* execute the claimed task */
		}
	}

	return NULL;
}

void thread_load(thread_fun fun, void *args, int length, int threads)
{
	pthread_mutex_t mutex;
	pthread_t *thread_id = (pthread_t*)malloc(sizeof(pthread_t)*threads);
	argsThread_t *arg_thread = (argsThread_t*)malloc(sizeof(argsThread_t)*length);
	int next = 0;
	
	pthread_mutex_init(&mutex, NULL);
	
	for (int i = 0; i < length; i++) {
		arg_thread[i].index = i;
		arg_thread[i].length = length;
		arg_thread[i].lock = 0;
		arg_thread[i].mutex = &mutex;
		arg_thread[i].next = &next;
		arg_thread[i].f = fun;
		arg_thread[i].args = args;
	}
	for (int i = 0; i < threads; i++) {
		pthread_create(&thread_id[i], NULL, thread_run, &arg_thread[i]);
	}
	for (int i = 0; i < threads; i++) {
		pthread_join(thread_id[i], NULL);
	}

	pthread_mutex_destroy(&mutex);
	
	free(thread_id);
	free(arg_thread);
}
