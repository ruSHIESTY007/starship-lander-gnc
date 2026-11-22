/** @file test_main.c  @brief Unit test entry point. */
#include <stdio.h>
#include "test_framework.h"

int g_tests_run = 0, g_tests_failed = 0;

void run_math_tests(void);
void run_nav_tests(void);
void run_wind_tests(void);

int main(void)
{
    run_math_tests();
    run_nav_tests();
    run_wind_tests();
    printf("\n%d checks, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
