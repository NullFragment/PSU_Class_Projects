#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>


#define USE_MPI 1
#define COMM_TIMER 0

#if USE_MPI

    #include <mpi.h>

#endif

static double timer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

int main(int argc, char **argv)
{

    int rank, num_tasks;

    /* Initialize MPI */
    #if USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status topsendrcv, botsendrcv;
    int max_thread = num_tasks - 1;

    #else
    rank = 0;
    num_tasks = 1;
    #endif

    if (argc != 3)
    {
        if (rank == 0)
        {
            fprintf(stderr, "%s <m> <k>\n", argv[0]);
            fprintf(stderr, "Program for parallel Game of Life\n");
            fprintf(stderr, "with 1D grid partitioning\n");
            fprintf(stderr, "<m>: grid dimension (an mxm grid is created)\n");
            fprintf(stderr, "<k>: number of time steps\n");
            fprintf(stderr, "(initial pattern specified inside code)\n");

            #if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
            #else
            exit(1);
            #endif
        }
    }

    int m, k;

    m = atoi(argv[1]);
    assert(m > 2);
    assert(m <= 10000);

    k = atoi(argv[2]);
    assert(k > 0);
    assert(k <= 1000);

    /* ensure that m is a multiple of num_tasks */
    m = (m / num_tasks) * num_tasks;

    int m_p = (m / num_tasks);

    /* print new m to let user know n has been modified */
    char *std_filename = calloc(14, sizeof(char));
    FILE *std_log;
    if (rank == 0)
    {
        strcat(std_filename, "std_times.txt");
        std_log = fopen(std_filename, "a+");
        fprintf(stderr, "Using m: %d, m_p: %d, k: %d\n", m, m_p, k);
        fprintf(stderr, "Requires %3.6lf MB of memory per task\n",
                ((2 * 4.0 * m_p) * m / 1e6));
    }

    /* Linearizing 2D grids to 1D using row-major ordering */
    /* grid[i][j] would be grid[i*n+j] */
    int *grid_current;
    int *grid_next;
    int *boundary_top;
    int *boundary_bottom;

    grid_current = (int *) calloc((size_t) m_p * m, sizeof(int));
    assert(grid_current != 0);

    grid_next = (int *) calloc((size_t) m_p * m, sizeof(int));
    assert(grid_next != 0);

    boundary_bottom = (int *) calloc((size_t) m, sizeof(int));
    assert(boundary_bottom != 0);

    boundary_top = (int *) calloc((size_t) m, sizeof(int));
    assert(boundary_top != 0);

    int i, j, t;

    /* static initalization, so that we can verify output */
    /* using very simple initialization right now */
    /* this isn't a good check for parallel debugging */
    #ifdef _OPENMP
        #pragma omp parallel for private(i,j)
    #endif
    for (i = 0; i < m_p; i++)
    {
        for (j = 0; j < m; j++)
        {
            grid_current[i * m + j] = 0;
            grid_next[i * m + j] = 0;
        }
    }

    /* initializing some cells in the middle */
    assert((m * m_p / 2 + m / 2 + 3) < m_p * m);
    grid_current[m * m_p / 2 + m / 2 + 0] = 1;
    grid_current[m * m_p / 2 + m / 2 + 1] = 1;
    grid_current[m * m_p / 2 + m / 2 + 2] = 1;
    grid_current[m * m_p / 2 + m / 2 + 3] = 1;

    #if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    double elt = 0.0;
    if (rank == 0)
        elt = timer();

    double comm_start = 0.0, comm_time = 0.0;
    char *time_filename = calloc(25, sizeof(char));
    FILE *time_log;

    if (COMM_TIMER == 1)
    {
        sprintf(time_filename, "rank_%02d_comm_times.txt", rank);
        time_log = fopen(time_filename, "a+");
    }

    #if USE_MPI
    int *full_grid, *full_grid_next;
    if (rank == 0)
    {
        full_grid = (int *) calloc(m_p * m * num_tasks, sizeof(int));
        full_grid_next = (int *) calloc(m_p * m * num_tasks, sizeof(int));
    }

    for (t = 0; t < k; t++)
    {
        if (COMM_TIMER == 1) comm_start = timer();
        if (rank > 0)
        {
            MPI_Sendrecv(&grid_current[0], m, MPI_INT, rank - 1, 0,
                         &boundary_top[0], m, MPI_INT, rank - 1, 0,
                         MPI_COMM_WORLD, &topsendrcv);
        } // Exchange Top
        if (rank < max_thread)
        {
            MPI_Sendrecv(&grid_current[m * m_p + 1], m, MPI_INT, rank + 1, 0,
                         &boundary_bottom[0], m, MPI_INT, rank + 1, 0,
                         MPI_COMM_WORLD, &botsendrcv);
        } // Exchange Bottom
        if (COMM_TIMER == 1) comm_time += timer() - comm_start;


        MPI_Barrier(MPI_COMM_WORLD);

        for (i = 1; i < m_p; i++)
        {
            for (j = 1; j < m - 1; j++)
            {
                /* avoiding conditionals inside inner loop */
                int prev_state = grid_current[i * m + j];
                int num_alive = 0;
                if (rank != 0 & i == 0)
                {
                    num_alive =
                            grid_current[(i) * m + j - 1] +
                            grid_current[(i) * m + j + 1] +
                            grid_current[(i - 1) * m + j - 1] +
                            grid_current[(i - 1) * m + j] +
                            grid_current[(i - 1) * m + j + 1] +
                            boundary_top[j - 1] +
                            boundary_top[j] +
                            boundary_top[j + 1];

                } else if (rank != max_thread & i == m_p)
                {
                    num_alive =
                            grid_current[(i) * m + j - 1] +
                            grid_current[(i) * m + j + 1] +
                            boundary_bottom[j - 1] +
                            boundary_bottom[j] +
                            boundary_bottom[j + 1] +
                            grid_current[(i + 1) * m + j - 1] +
                            grid_current[(i + 1) * m + j] +
                            grid_current[(i + 1) * m + j + 1];

                } else if (i != 0 && i != m_p)
                {
                    num_alive =
                            grid_current[(i) * m + j - 1] +
                            grid_current[(i) * m + j + 1] +
                            grid_current[(i - 1) * m + j - 1] +
                            grid_current[(i - 1) * m + j] +
                            grid_current[(i - 1) * m + j + 1] +
                            grid_current[(i + 1) * m + j - 1] +
                            grid_current[(i + 1) * m + j] +
                            grid_current[(i + 1) * m + j + 1];
                }


                grid_next[i * m + j] =
                        prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);

            }
        }
        if (COMM_TIMER == 1) comm_start = timer();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(grid_current, m_p * m, MPI_INT, full_grid, m_p * m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(grid_next, m_p * m, MPI_INT, full_grid_next, m_p * m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (COMM_TIMER == 1) comm_time += timer() - comm_start;
        int *grid_tmp = grid_next;
        grid_next = grid_current;
        grid_current = grid_tmp;
    }


    #else
    /* serial code */
    /* considering only internal cells */
    for (t = 0; t < k; t++)
    {
        for (i = 1; i < m - 1; i++)
        {
            for (j = 1; j < m - 1; j++)
            {
                /* avoiding conditionals inside inner loop */
                int prev_state = grid_current[i * m + j];
                int num_alive =
                        grid_current[(i  )*m+j-1] +
                        grid_current[(i  )*m+j+1] +
                        grid_current[(i-1)*m+j-1] +
                        grid_current[(i-1)*m+j  ] +
                        grid_current[(i-1)*m+j+1] +
                        grid_current[(i+1)*m+j-1] +
                        grid_current[(i+1)*m+j  ] +
                        grid_current[(i+1)*m+j+1];

                grid_next[i * m + j] =
                        prev_state * ((num_alive == 2) + (num_alive == 3)) + (1 - prev_state) * (num_alive == 3);
            }
        }
        /* swap current and next */
        int *grid_tmp = grid_next;
        grid_next = grid_current;
        grid_current = grid_tmp;
    }
    #endif

    if (rank == 0)
        elt = timer() - elt;

    if (rank == 0 && COMM_TIMER == 0)
    {
        double updates = ((1.0 * m * m) * k / (elt * 1e9));
        fprintf(std_log,"p\t%d\tm\t%d\tm_p\t%d\tk\t%d\ttime\t%f\tupdates\t%f\n", num_tasks, m, m_p, k, elt, updates);
    }
    if (COMM_TIMER == 1)
    {
        fprintf(time_log, "p\t%d\tm\t%d\tm_p\t%d\tk\t%d\trank\t%d\tcomm_time\t%f\n", num_tasks, m, m_p, k,rank, comm_time);
    }

    /* free memory */
    free(grid_current);
    free(grid_next);

/* Shut down MPI */
#if USE_MPI

    MPI_Finalize();

#endif

    if (rank == 0)
    {
        fclose(std_log);
    }
    if (COMM_TIMER == 1)
    {
        fclose(time_log);
    }

    return 0;
}
