#include <mpi.h>
#include <stdio.h>

#include "monte_carlo.h"
#include "util.h"

 #define MIN(a,b) ((a) < (b) ? (a) : (b))
 #define MAX(a,b) ((a) > (b) ? (a) : (b))

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  int count = 0;

  // TODO: Parallelize the code using mpi_world_size processes (1 process per
  // node.
  // In total, (mpi_world_size * threads_per_process) threads will collaborate
  // to compute pi.
  int local_count = 0;
  omp_set_num_threads(threads_per_process);
  int step = num_points / mpi_world_size;
  int last_step = 0;
  MPI_Bcast(xs, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ys, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  #pragma parallel for
  for (int i = mpi_rank * step; i < MIN((mpi_rank + 1) * step, num_points); i++) {
    double x = xs[i];
    double y = ys[i];
    if (x*x + y*y <= 1)
      local_count++;
  }

  MPI_Reduce(&local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return (double) 4 * count / num_points;
}
