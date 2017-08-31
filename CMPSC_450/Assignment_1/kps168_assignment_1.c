#include <stdio.h>
#include <sys/time.h>

/*
MEMORY READ BENCHMARK
*/

void get_walltime(double *wcTime)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double) (tp.tv_sec + tp.tv_usec / 1000000.0);
}

void dummy(double A, double B, double C, double D)
{

}

double benchmark(int R, int N)
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
    mflops = benchmark(1000, 1000);
    printf("MFLOPS: %G", mflops);
    return 0;
}