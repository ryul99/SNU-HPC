#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define TS 32

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K

  // 0 ... col_size
  const int col = get_local_id(0);
  // 0 ... row_size
  const int row = get_local_id(1);
  const int col_size = get_local_size(0);
  const int row_size = get_local_size(1);
  // n - col idx
  const int global_col = TS * get_group_id(0) + col;
  // m - row idx
  const int global_row = TS * get_group_id(1) + row;

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];


  float c = 0.0;
  int t = 0;
  const int numTiles = K / TS;
  for (int t = 0; t < numTiles; ++t) {
    const int tiledRow = TS * t + row;
    const int tiledCol = TS * t + col;
    Asub[row][col] = A[tiledCol + K * global_row];
    Bsub[row][col] = B[global_col + N * tiledRow];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int k = 0; k < TS; k++) {
      c += Asub[row][k] * Bsub[k][col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[global_col + N * global_row] = c;
}


__kernel void sgemm_naive(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K

  // 0 ... col_size
  const int col = get_local_id(0);
  // 0 ... row_size
  const int row = get_local_id(1);
  const int col_size = get_local_size(0);
  const int row_size = get_local_size(1);
  // n - col idx
  const int global_col = get_local_size(0) * get_group_id(0) + col;
  // m - row idx
  const int global_row = get_local_size(1) * get_group_id(1) + row;

  float c = 0.0;
  for (int k = 0; k < K; ++k) {
    c += A[k + K * global_row] * B[global_col + N * k];
  }
  C[global_col + N * global_row] = c;
}

__kernel void transpose(__global float *src, __global float *dst, int M, int N) {
  // M x N => N x M
  const int m = get_global_id(0);
  const int n = get_global_id(1);
  dst[m + M * n] = src[n + N * m];
}