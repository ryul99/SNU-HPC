#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {
  // TODO: FILL_IN_HERE
  // A: M x K
  // B: K x N
  // C: M x N
  // printf("HI\n");
  omp_set_num_threads(num_threads);
  float a;
  float sh[N * num_threads];
  #pragma omp parallel private(a)
  {
    int tid = omp_get_thread_num();
    int pos = tid * N;
    #pragma omp for nowait
    for (int m = 0; m < M; ++m) {
      memset(&sh[pos], 0, N * sizeof(float));
      for (int k = 0; k < K; ++k) {
        a = A[k + K * m];
        for (int n = 0; n < N; ++n) {
          sh[n + pos] += a * B[n + N * k];
        }
      }
      memcpy(&C[N*m], &sh[pos], N * sizeof(float));
    }
  }
}
