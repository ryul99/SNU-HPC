#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

__m512 __fma(__m512 a, __m512 b, __m512 c) {

  return _mm512_fmadd_ps(a, b, c);
}

__m512 __vectordot(float *A, float *B, int K, int mpi_world_size) {
  __m512 curr = _mm512_set1_ps(0);
  __m512 a, b;

  for (int k = 0; k < K; k += 16) {
    a = _mm512_load_ps(&A[k]);
    b = _mm512_load_ps(&B[k]);
    curr = __fma(a, b, curr);;
  }
  return curr;
}

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE
  omp_set_num_threads(threads_per_process);
  __m512 curr;
  // int chunkM = M / mpi_world_size;
  int chunkK = K / mpi_world_size;
  float *Bp, *Bc, *Cc;
  // alloc_mat(&Ac, K, chunkM);
  alloc_mat(&Bp, N, chunkK);
  alloc_mat(&Bc, chunkK, N);
  alloc_mat(&Cc, N, M);

  // A: M x K
  // B: K x N
  // C: M x N

  // Ac: chunkM x K
  // Bc: chunkK x N
  // Cc: M x N

  // Bp: N x chunkK
  // MPI_Scatter(A, K * chunkM, MPI_FLOAT, Ac, K * chunkM, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(A, M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(B, chunkK * N, MPI_FLOAT, Bc, chunkK * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #pragma omp parallel for
  for (int k = 0; k < chunkK; ++k) {
    for (int n = 0; n < N; ++n) {
      Bp[k + chunkK * n] = Bc[n + N * k];
    }
  }
  #pragma omp barrier
  #pragma omp parallel private(curr)
  {
    #pragma omp for
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        curr = __vectordot(&A[mpi_rank * chunkK + K * m], &Bp[chunkK * n], chunkK, mpi_world_size);
        Cc[n + N * m] = _mm512_reduce_add_ps(curr);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(Cc, C, N * M, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  // free(Ac);
  free(Bp);
  free(Cc);
}
