/******************************************************************************
Compile command: g++ -std=c++0x -fopenmp -o assignment3 assignment3.cpp -O3
******************************************************************************/
#include <omp.h>
#include <iostream>
#include <cmath>
#include <random>
#include <sys/time.h>
#include <fstream>

void get_walltime(long double *wcTime)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (long double) (tp.tv_sec + tp.tv_usec / 1000000.0);
}

/** binary_sum
 *
 * Calculates the sum of an array in a recursive binary-tree PRAM fashion.
 *
 * @param *array - the array to be computed
 * @param array_size - necessary for proper assignment
 * @param *time - reference to a variable to store the walltime
 *
 * @return - sum of array as a double
*/
double binary_sum(double *array, int array_size, long double *time)
{
    long double start_time = 0.0, end_time = 0.0, scratch_time = 0.0;
    double answer = 0.0, *results;
    int scratch_size, num_elements;

    // Handle edge cases of 1 or 2 element arrays
    if (array_size == 1)
    {
        return (array[0]);
    }
    if (array_size == 2)
    {
        return (array[0] + array[1]);
    }

    /** Create a scratch array to put temporary results in. If the input array is not divisible by 2, add 1 so that
     * scratch array can be a round integer. */
    if (array_size % 2 == 1)
    {
        scratch_size = (array_size + 1) / 2;
        results = new double[scratch_size]();
    } else
    {
        scratch_size = array_size / 2;
        results = new double[scratch_size]();
    }

    // Gets wall time and start parallel block.
    get_walltime(&start_time);
    #pragma omp parallel
    {
        int threads, thread_id, start, end;
        threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        #pragma omp single
        {
            /** Sets variables shared between threads and handles the final element in arrays first dependent on
             * whether or not the scratch array size is even or odd. */
            if (array_size % 2 == 1)
            {
                results[scratch_size - 1] = array[array_size - 1]; // Dynamic Allocation
            } else
            {
                results[scratch_size - 1] = array[array_size - 1] + array[array_size - 2]; // Dynamic Allocation
            }
            // divide and conquer b-tree adds based on number of threads
            num_elements = (int) ceil((double) (scratch_size) / (double) threads);
        }
        // Compute start and end values for each thread
        start = num_elements * thread_id;
        end = start + num_elements;

        // Ends at scratch array - 1 because final element is handled above.
        for (start; start < end && start < scratch_size - 1; start++)
        {
            // start*2 is used to access the correct array elements:
            // results[i] = array[2*i] + array[2*i + 1]
            results[start] = array[start * 2] + array[start * 2 + 1];

        }
    }
    // RECURSE! Scratch time is a dummy variable created in place of overloading the function call
    answer = binary_sum(results, scratch_size, &scratch_time);
    get_walltime(&end_time);
    *time = end_time - start_time;
    free(results); // frees dynamically allocated memory
    return answer;
}


/** parallel_sum
 *
 * Calculates the sum of an array by dividing array into subsections and then summing those on different threads.
 *
 * @param *array - the array to be computed
 * @param array_size - necessary for proper assignment
 * @param *time - reference to a variable to store the walltime
 *
 * @return - sum of array as a double
*/
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
            /** Allocate variables necessary to share between threads. Scratch space is allocated based on the
             * number of threads obtained by OpenMP (one array element per thread). */
            scratch = new double[threads](); // Dynamic allocation
            scratch_size = threads;

            // Divide array up into a set number of elements per thread.
            num_elements = (int) ceil((double) array_size / (double) threads);
        }

        // Set start and endpoints for each thread
        start = num_elements * thread_id;
        end = start + num_elements;

        // Sum up elements into scratch array
        for (start; start < end && start < array_size; start++)
        {
            scratch[thread_id] += array[start];
        }
    }

    // Go through scratch array and calculate final sum.
    for (int i = 0; i < scratch_size; i++)
    {
        sum += scratch[i];
    }
    free(scratch); // Free dynamically allocated memory
    get_walltime(&end_time);
    *time = end_time - start_time;
    return sum;
}


/** fast_sum
 *
 * Calculates the sum of an array using the OpenMP reduction method.
 *
 * @param *array - the array to be computed
 * @param array_size - necessary for proper assignment
 * @param *time - reference to a variable to store the walltime
 *
 * @return - sum of array as a double
*/
double fast_sum(double *array, int array_size, long double *time)
{
    double sum;
    long double start_time = 0.0, end_time = 0.0;
    sum = 0.0;
    get_walltime(&start_time);
    #pragma omp parallel for reduction(+:sum)
    // declares sum to be the accumulator variable for all threads 
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
    int set_threads = 10; // Threads to be requested by OpenMP
    char filename[11];
    sprintf(filename, "log_%02d.txt", set_threads);
    omp_set_num_threads(set_threads); // Set global variable (shell script did not appear to actually set value)

    std::ofstream log;
    log.open(filename, std::ofstream::out | std::ofstream::app); // Open file
    log << "ArraySize\tBinarySumTime\tBinarySumTotal\tParallelSumTime\tParallelSumTotal\tFastSumTime\t"
            "FastSumTotal\tBinParDiff\tBinFastDiff\tParFastDiff\n";

    for (int array_size = 1000000; array_size <= 10000000; array_size += 100000)
    {
        auto *x = new double[array_size](); // Dynamic allocation
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
        free(x); // Free dynamically allocated memory
    }
    log.close(); // Close file
    return 0;
}
