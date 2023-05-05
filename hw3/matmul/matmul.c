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

float hadd(__m512 v) {
  //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
  // -> 4 3 2 1 8 7 6 5
  __m512 curr = _mm512_shuffle_ps(v, v, 0b00011011);
  // 5 5 5 5 13 13 13 13
  curr = _mm512_add_ps(v, curr);
  // 10 26 10 26 10 26 10 26
  curr = _mm512_hadd_ps(curr, curr);
  // 36 36 36 36 36 36 36 36 
  curr = _mm512_hadd_ps(curr, curr);

  float ret;
  _mm512_mask_store_ps(&ret, 1, curr);
  return ret;
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
  MPI_Bcast(&Bp, N * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // Data Transfer: B
  int chunksize = M / mpi_world_size;
  float *Ac, *Cc;
  __m512 Ccc[N * chunksize];
  alloc_mat(&Ac, K, chunksize);
  alloc_mat(&Cc, N, chunksize);
  MPI_Scatter(A, K * chunksize, MPI_FLOAT, Ac, K * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);

  __m512 a;
  __m512 b;
  __m512 c;
  __m512 curr;
  
  omp_set_num_threads(threads_per_process);
  #pragma omp parallel private(a, b, c, curr)
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
  }

  // TODO Convert Ccc to Cc

  MPI_Gather(Cc, N * chunksize, MPI_FLOAT, C, N * chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
