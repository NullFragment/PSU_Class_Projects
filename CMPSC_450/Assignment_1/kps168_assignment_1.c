#include <stdio.h>
#include <sys/time.h>

/*
MEMORY READ BENCHMARK

System Info:
    CPU: Intel Core i7-6700HQ, 2.6 GHz
    GPU: GeForce GTX 1060
    Memory: 16GB
    Operating System: Ubuntu 16.04 LTS 64-bit
    Compiler: GCC

compiled using: gcc -Wall kps168_assignment_1.c -o benchmark -O2; ./benchmark
*/

void get_walltime(double *wcTime)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double) (tp.tv_sec + tp.tv_usec / 1000000.0);
}

void dummy(double A, double B, double C, double D)
{}

double benchmark(long int R, long int N)
{
    double A[N], B[N], C[N], D[N], S, E, MFLOPS;
    for (int i = 0; i < N; i++)
    {
        A[i] = 0.0;
        B[i] = 1.0;
        C[i] = 2.0;
        D[i] = 3.0;
    }
    get_walltime(&S);
    for (int j = 1; j < R; j++)
    {
        for (int i = 1; i < N; i++)
        {
            A[i] = B[i] + C[i] * D[i];
        }
        if (A[2] < 0)
        {
            dummy(0.0, 0.0, 0.0, 0.0);
        }
    }

    get_walltime(&E);
    MFLOPS = (R * N * 2.0) / ((E - S) * 1000000.0);
    return (MFLOPS);
}

int main()
{
    double mflops;
    long int m, n, r;
    m = 20;
    r = 100000;
    printf("R,N,MFLOPS\n");
    for (int i = 0; i < m; i++)
    {
        n = 5000 + 5000 * i;
        mflops = benchmark(r, n);
        printf("%ld,%ld,%f\n", r, n, mflops);
    }
    return 0;
}