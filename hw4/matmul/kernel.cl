#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__kernel void sgemm(const __global float *A, const __global float *B, __global float *C, int M, int N, int K, int TS, __local float *sA, __local float *sB) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int row_size = get_local_size(0);
  const int col_size = get_local_size(1);
  // m - row idx
  const int global_row = row_size * get_group_id(0) + row;
  // n - col idx
  const int global_col = col_size * get_group_id(1) + col;
  
  float c = 0.0;

  const int num_tiles = K / TS;
  for (int t = 0; t < num_tiles; ++t) {
    const int tiledRow = TS * t + row;
    const int tiledCol = TS * t + col;
    sA[col + row * TS] = A[tiledCol + K * global_row];
    sB[col + row * TS] = B[global_col + N * tiledRow];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TS; k++) {
        c += sA[k + TS * row] * sB[col + TS * k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[global_col + N * global_row] = c;
}
