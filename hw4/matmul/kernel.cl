#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K
  
  // 0 ... row_size
  const int row = get_local_id(0);
  // 0 ... col_size
  const int col = get_local_id(1);
  const int row_size = get_local_size(0);
  const int col_size = get_local_size(1);
  // m - row idx
  const int global_row = row_size * get_group_id(0) + row;
  // n - col idx
  const int global_col = col_size * get_group_id(1) + col;


  float c = 0.0;
  for (int k = 0; k < K; ++k) {
    c += A[global_row + M * k] * B[global_col + N * k];
  }
  C[global_col + N * global_row] = c;
}

__kernel void transpose(__global float *src, __global float *dst, int M, int N) {
  // M x N => N x M
  const int m = get_global_id(0);
  const int n = get_global_id(1);
  dst[m + M * n] = src[n + N * m];
}