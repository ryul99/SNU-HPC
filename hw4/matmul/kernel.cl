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
