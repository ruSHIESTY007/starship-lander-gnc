/**
 * @file test_framework.h
 * @brief Minimal dependency-free unit test harness for the flight software.
 */
#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H
#include <math.h>
#include <stdio.h>

extern int g_tests_run, g_tests_failed;

#define CHECK(cond) do { ++g_tests_run; if (!(cond)) { ++g_tests_failed; \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_NEAR(a, b, tol) do { ++g_tests_run; { double _a=(a), _b=(b); \
    if (!(fabs(_a-_b) <= (tol))) { ++g_tests_failed; \
    printf("  FAIL %s:%d: %s=%.7g vs %s=%.7g (tol %.3g)\n", __FILE__, __LINE__, #a, _a, #b, _b, (double)(tol)); } } } while (0)
#define RUN(fn) do { printf("[ RUN  ] %s\n", #fn); fn(); } while (0)

#endif
