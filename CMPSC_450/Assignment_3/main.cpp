/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <iostream>
#include <cmath>
#include <random>
#include <sys/time.h>
#include <cstdlib>
#include <fstream>

void get_walltime(long double *wcTime)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (long double) (tp.tv_sec + tp.tv_usec / 1000000.0);
}


double binary_sum(double *array, int array_size, long double *time)
{
    long double start_time = 0.0, end_time = 0.0, scratch_time = 0.0;
    double answer = 0.0, *results;
    int scratch_size, num_elements;
    if (array_size == 1)
    {
        return (array[0]);
    }
    if (array_size == 2)
    {
        return (array[0] + array[1]);
    }

    if (array_size % 2 == 1)
    {
        scratch_size = (array_size + 1) / 2;
        results = new double[scratch_size]();
    } else
    {
        scratch_size = array_size / 2;
        results = new double[scratch_size]();
    }
    get_walltime(&start_time);
    #pragma omp parallel
    {
        int threads, thread_id, start, end;
        threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        #pragma omp single
        {
            if (array_size % 2 == 1)
            {
                results[scratch_size - 1] = array[array_size - 1];
            } else
            {
                results[scratch_size - 1] = array[array_size - 1] + array[array_size - 2];
            }
            num_elements = (int) ceil((double) (scratch_size) / (double) threads);
        }
        start = num_elements * thread_id;
        end = start + num_elements;
        for (start; start < end && start < scratch_size - 1; start++)
        {
            results[start] = array[start * 2] + array[start * 2 + 1];
        }
    }
    answer = binary_sum(results, scratch_size, &scratch_time);
    get_walltime(&end_time);
    *time = end_time - start_time;
    free(results);
    return answer;
}


double parallel_sum(double *array, int array_size, long double *time)
{
    double *scratch, sum;
    long double start_time = 0.0, end_time = 0.0;
    int scratch_size, num_elements;
    sum = 0.0;
    get_walltime(&start_time);
    #pragma omp parallel
    {
        int threads, thread_id, start, end;
        threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        #pragma omp single
        {
            scratch = new double[threads]();
            scratch_size = threads;
            num_elements = (int) ceil((double) array_size / (double) threads);
        }
        start = num_elements * thread_id;
        end = start + num_elements;

        for (start; start < end && start < array_size; start++)
        {
            scratch[thread_id] += array[start];
        }
    }
    for (int i = 0; i < scratch_size; i++)
    {
        sum += scratch[i];
    }
    free(scratch);
    get_walltime(&end_time);
    *time = end_time - start_time;
    return sum;
}

double fast_sum(double *array, int array_size, long double *time)
{
    double sum;
    long double start_time = 0.0, end_time = 0.0;
    sum = 0.0;
    get_walltime(&start_time);
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < array_size; i++)
    {
        sum += array[i];
    }
    get_walltime(&end_time);
    *time = end_time - start_time;
    return sum;
}


int main()
{
    std::ofstream log;
    log.open("log.txt", std::ofstream::out | std::ofstream::app);

    log << "ArraySize\tBinarySumTime\tBinarySumTotal\tParallelSumTime\tParallelSumTotal\tFastSumTime\t"
            "FastSumTotal\tBinParDiff\tBinFastDiff\tParFastDiff\n";
    for (int array_size = 1000000; array_size <= 10000000; array_size += 100000)
    {
        auto *x = new double[array_size]();
        long double parallel_time = 0.0, fast_time = 0.0, binary_time = 0.0;
        for (int i = 0; i < array_size; i++)
        {
            x[i] = (double) (rand()) / (double) (RAND_MAX) * 5.0;
        }
        double binary_total = binary_sum(x, array_size, &binary_time);
        double parallel_total = parallel_sum(x, array_size, &parallel_time);
        double fast_total = fast_sum(x, array_size, &fast_time);
        log << array_size << "\t" << binary_time << "\t" << binary_total << "\t" << parallel_time << "\t"
            << parallel_total << "\t" << fast_time << "\t" << fast_total << "\t" << binary_total - parallel_total
            << "\t" << binary_total - fast_total << "\t" << parallel_total - fast_total << std::endl;
        free(x);
    }
    log.close();

    return 0;
}
