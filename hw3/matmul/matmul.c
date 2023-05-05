#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

__m512 __fma(__m512 a, __m512 b, __m512 c, __m512 curr) {

  return _mm512_add_ps(curr, _mm512_fmadd_ps(a, b, c));
}

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE
  // A: M x K
  // B: K x N
  // C: M x N

  // Bp: N x K
  float *Bp;
  alloc_mat(&Bp, N, K);
  if (mpi_rank == 0) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        Bp[k + K * n] = B[n + N * k];
      }
    }
  }

  // Data Transfer: A
  MPI_Bcast(Bp, N * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // Data Transfer: B
  int chunksize = M / mpi_world_size;
  float *Ac, *Cc;
  __m512 *Ccc = aligned_alloc(512, N * chunksize * sizeof(__m512));
  alloc_mat(&Ac, K, chunksize);
  alloc_mat(&Cc, N, chunksize);
  MPI_Scatter(A, K * chunksize, MPI_FLOAT, Ac, K * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);

  __m512 a;
  __m512 b;
  __m512 c;
  
  omp_set_num_threads(threads_per_process);
  #pragma omp parallel private(a, b, c)
  {
    #pragma omp for nowait
    for (int m = 0; m < chunksize; ++m) {
      for (int n = 0; n < N; ++n) {
        c = _mm512_set1_ps(Cc[n + N * m]);
        for (int k = 0; k < K; k += 16) {
          a = _mm512_load_ps(&Ac[k + K * m]);
          b = _mm512_load_ps(&Bp[k + K * n]);
          Ccc[n + N * m] = __fma(a, b, c, Ccc[n + N * m]);;
        }
      }
    }

    #pragma omp for nowait
    for (int i = 0; i < chunksize * N; ++i) {
      Cc[i] = _mm512_reduce_add_ps(Ccc[i]);
    }
  }
  free(Ccc);
  MPI_Gather(Cc, N * chunksize, MPI_FLOAT, C, N * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
