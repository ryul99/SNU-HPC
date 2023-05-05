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

__m512 __vectordot(float *A, float *B, __m512 c, int K) {
  __m512 curr = _mm512_set1_ps(0);
  __m512 a, b;

  for (int k = 0; k < K; k += 16) {
    a = _mm512_load_ps(&A[k]);
    b = _mm512_load_ps(&B[k]);
    curr = __fma(a, b, c, curr);;
  }
  return curr;
}

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE
  omp_set_num_threads(threads_per_process);
  __m512 a;
  __m512 b;
  __m512 c;
  __m512 curr;
  int chunksize = M / mpi_world_size;
  MPI_Request r[2];
  float *Ac, *Bp, *Cc;
  alloc_mat(&Ac, K, chunksize);
  alloc_mat(&Bp, N, K);
  alloc_mat(&Cc, N, chunksize);
  MPI_Iscatter(A, K * chunksize, MPI_FLOAT, Ac, K * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD, &r[1]);

  // A: M x K
  // B: K x N
  // C: M x N

  // Bp: N x K
  if (mpi_rank == 0) {
    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        Bp[k + K * n] = B[n + N * k];
      }
    }
  }

  // Data Transfer
  MPI_Ibcast(Bp, N * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &r[0]);
  MPI_Waitall(2, r, MPI_STATUS_IGNORE);

  
  #pragma omp parallel private(a, b, c, curr)
  {
    #pragma omp for nowait
    for (int m = 0; m < chunksize; ++m) {
      for (int n = 0; n < N; ++n) {
        c = _mm512_set1_ps(Cc[n + N * m]);
        curr =  __vectordot(&Ac[K * m], &Bp[K * n], c, K);
        Cc[n + N * m] = _mm512_reduce_add_ps(curr);
      }
    }
  }
  MPI_Gather(Cc, N * chunksize, MPI_FLOAT, C, N * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  free(Ac);
  free(Bp);
  free(Cc);
}
