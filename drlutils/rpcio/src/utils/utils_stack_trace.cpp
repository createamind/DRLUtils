//
// Created by Robin Huang on 12/14/16.
//
#include "stdafx.h"
#include "utils_stack_trace.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The code below was obtained from:
// http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes/1925461#1925461
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif

#ifdef LINUX

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

/* This structure mirrors the one found in /usr/include/asm/ucontext.h */
typedef struct _sig_ucontext {
    unsigned long     uc_flags;
    struct ucontext   *uc_link;
    stack_t           uc_stack;
    struct sigcontext uc_mcontext;
    sigset_t          uc_sigmask;
} sig_ucontext_t;

static void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext) {
    void *             array[50];
    void *             caller_address;
    char **            messages;
    int                size, i;
    sig_ucontext_t *   uc;

    uc = (sig_ucontext_t *)ucontext;

    /* Get the address at the time the signal was raised */
#if defined(__i386__) // gcc specific
    caller_address = (void *) uc->uc_mcontext.eip; // EIP: x86 specific
#elif defined(__x86_64__) // gcc specific
    caller_address = (void *) uc->uc_mcontext.rip; // RIP: x86_64 specific
#else
#error Unsupported architecture. // TODO: Add support for other arch.
#endif

    fprintf(stderr, "\n");
    FILE * backtraceFile;

    // In this example we write the stacktrace to a file. However, we can also just fprintf to stderr (or do both).
//    QString backtraceFilePath = "/home/my_user/stacktrace.txt;
//    backtraceFile = fopen(backtraceFilePath.tostd(),"w");
    backtraceFile = stderr;

    if (sig_num == SIGSEGV)
        fprintf(backtraceFile, "signal %d (%s), address is %p from %p\n",sig_num, strsignal(sig_num), info->si_addr,(void *)caller_address);
    else
        fprintf(backtraceFile, "signal %d (%s)\n",sig_num, strsignal(sig_num));

    size = backtrace(array, 50);
    /* overwrite sigaction with caller's address */
    array[1] = caller_address;
    messages = backtrace_symbols(array, size);
    /* skip first stack frame (points here) */
    for (i = 1; i < size && messages != NULL; ++i) {
        fprintf(backtraceFile, "[bt]: (%d) %s\n", i, messages[i]);
    }

//    fclose(backtraceFile);
    free(messages);

    exit(EXIT_FAILURE);
}

void installSignal(int __sig) {
    struct sigaction sigact;
    sigact.sa_sigaction = crit_err_hdlr;
    sigact.sa_flags = SA_RESTART | SA_SIGINFO;
    if (sigaction(__sig, &sigact, (struct sigaction *)NULL) != 0) {
        fprintf(stderr, "error setting signal handler for %d (%s)\n",__sig, strsignal(__sig));
        exit(EXIT_FAILURE);
    }
}


#endif

void init_stack_trace() {
#ifdef LINUX
    installSignal(SIGSEGV);
#endif
}
