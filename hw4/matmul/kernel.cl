#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K
  const int m = get_local_size(0) * get_group_id(0) + get_local_id(0);
  const int n = get_local_size(1) * get_group_id(1) + get_local_id(1);
  float c = 0.0;
  for (int k = 0; k < K; ++k) {
    c += A[k + K * m] * B[n + N * k];
  }
  C[n + N * m] = c;
}

__kernel void transpose(__global float *src, __global float *dst, int M, int N) {
  // M x N => N x M
  const int mn = get_local_size(0) * get_group_id(0) + get_local_id(0);
  const int n = mn % N;
  const int m = mn / N;
  dst[m + M * n] = src[n + N * m];
}