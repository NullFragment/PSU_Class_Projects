/*
Compiled using:

g++ main.cpp -o main_default
g++ main.cpp -o main_O1 -O1
g++ main.cpp -o main_O2 -O2
g++ main.cpp -o main_O3 -O3
g++ main.cpp -o main_O1_mavx -O1 -mavx
g++ main.cpp -o main_O2_mavx -O2 -mavx
g++ main.cpp -o main_O3_mavx -O3 -mavx

===============================================

main_default   Elapsed time: 0.190453
main_O1        Elapsed time: 0.029483
main_O2        Elapsed time: 0.020534
main_O3        Elapsed time: 0.018077
main_O1_mavx   Elapsed time: 0.026962
main_O2_mavx   Elapsed time: 0.017295
main_O3_mavx   Elapsed time: 0.016406
*/

#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <cmath>
#include <stdlib.h>

void get_walltime(double *wcTime)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double) (tp.tv_sec + tp.tv_usec / 1000000.0);
}

// complex algorithm for evaluation
void myfunc(std::vector<std::vector<double> > &v_s,
            std::vector<std::vector<double> > &v_mat, std::vector<int> &i_v,
            std::vector<double> &value_map)
{
    int d_val;
    for (int j = 0; j < v_s.size(); j++)
    {
        for (int i = 0; i < v_s[0].size(); i++)
        {
            d_val = i_v[i] % 256;
            v_mat[i][j] = v_s[i][j] * (value_map[d_val]);
        }
    }
}

int main(int argc, char *argv[])
{
    // this should be called as> ./slow_code <i_R> <i_N>
    int i_R = 1000;
    int i_N = 100;

    double d_S, d_E;
    // parse input parameters
    if (argc >= 2)
    {
        i_R = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        i_N = atoi(argv[2]);
    }

    // some declarations
    std::vector<std::vector<double> > vd_s(i_N, std::vector<double>(i_N));
    std::vector<std::vector<double> > vd_mat(i_N, std::vector<double>(i_N));
    std::vector<int> vi_v(i_N);
    std::vector<double> vd_difference(256);
    int sin_val;
    double sine, cosine;
    // populate memory with some random data
    for (int i=0;i<256;i++)
    {
        sine = sin(i);
        cosine = cos(i);
        vd_difference[i]= sine*sine - cosine*cosine;
    }

    for (int i = 0; i < i_N; i++)
    {
        vi_v[i] = i * i;
        for (int j = 0; j < i_N; j++)
        {
            vd_s[i][j] = j + i;
        }
    }

    // start benchmark
    get_walltime(&d_S);

    // iterative test loop
    for (int i = 0; i < i_R; i++)
    {
        myfunc(vd_s, vd_mat, vi_v, vd_difference);
    }

    // end benchmark
    get_walltime(&d_E);

    // report results
    printf("Elapsed time: %f\n", d_E - d_S);

    return 0;
}
