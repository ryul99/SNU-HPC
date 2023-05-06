#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

 #define MIN(a,b) ((a) < (b) ? (a) : (b))
 #define MAX(a,b) ((a) > (b) ? (a) : (b))

void transpose(float *src, float *dst, int I, int J) {
  // I x J -> J x I
  #pragma omp for
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      dst[i + I * j] = src[j + J * i];
    }
  }
}

__m512 __vectordot(float *A, float *B, int K, int mpi_world_size) {
  __m512 curr = _mm512_set1_ps(0);
  __m512 a, b;

  for (int k = 0; k < K; k += 16) {
    a = _mm512_loadu_ps(&A[k]);
    b = _mm512_loadu_ps(&B[k]);
    curr = _mm512_fmadd_ps(a, b, curr);;
  }
  return curr;
}

void matmul(float *A, float *B, float *C, int M, int N, int K,
    int threads_per_process, int mpi_rank, int mpi_world_size) {
  // TODO: FILL_IN_HERE
  omp_set_num_threads(threads_per_process);
  __m512 curr;
  int chunkM = M / mpi_world_size;
  // int chunkK = K / mpi_world_size;
  int sz = 32;
  float *Ac, *Cc;
  float *Ap, *Bp;
  alloc_mat(&Ac, K, chunkM);
  // alloc_mat(&Bc, chunkK, N);
  alloc_mat(&Ap, chunkM, K);
  alloc_mat(&Bp, N, K);
  alloc_mat(&Cc, N, M);

  // A: M x K
  // B: K x N
  // C: M x N

  // Ac: chunkM x K
  // Bc: chunkK x N
  // Cc: M x N

  // Bp: N x chunkK
  // MPI_Scatter(A, K * chunkM, MPI_FLOAT, Ac, K * chunkM, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(A, chunkM * K, MPI_FLOAT, Ac, chunkM * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #pragma omp parallel private(curr)
  {
    transpose(B, Bp, K, N);
    #pragma omp for
    for (int mm = 0; mm < chunkM; mm += sz) {
      for (int nn = 0; nn < N; nn += sz) {
        for (int m = mm; m < MIN(mm + sz, chunkM); ++m) {
          for (int n = nn; n < MIN(nn + sz, N); ++n) {
            curr = __vectordot(&Ac[K * m], &Bp[K * n], K, mpi_world_size);
            Cc[n + N * m] = _mm512_reduce_add_ps(curr);
          }
        }
      }
    }
  }
  MPI_Gather(Cc, chunkM * N, MPI_FLOAT, C, chunkM * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // free(Ac);
  // free(Bp);
  // free(Bc);
  // free(Cc);
}
