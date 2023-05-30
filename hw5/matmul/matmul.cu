#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }



#define NUM_ELEM 4096
#define NUM_BUFFER_ELEM 32
#define TS 16
#define NUM_GPU 4
#define NUM_NODE 4
#define NUM_THREAD 256

float *h_A, *h_B, *h_C;
float *d_A[NUM_GPU], *d_B[NUM_GPU], *d_C[NUM_GPU];
cudaStream_t s_d[NUM_GPU];
cudaEvent_t ev_d[NUM_GPU];
int mpi_rank, mpi_world_size;


__global__ void matmul_cal(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE

  // A: M x K
  // B: K x N
  // C: M x N

  // Ap: K x M
  // Bp: N x K

  // 0 ... col_size
  const int col = threadIdx.x;
  // 0 ... row_size
  const int row = threadIdx.y;
  // const int col_size = blockDim.x;
  // const int row_size = blockDim.y;
  // n - col idx
  const int global_col = TS * blockIdx.x + col;
  // m - row idx
  const int global_row = TS * blockIdx.y + row;

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  // if (row == 0 && col == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   for (int row = 0; row < TS; ++row) {
  //     for (int col = 0; col < TS; ++col) {
  //       Asub[row][col] = 0;
  //     }
  //   }
  // }

  float c = 0.0;
  const int numTiles = K / TS;
  for (int t = 0; t < numTiles; ++t) {
    const int tiledRow = TS * t + row;
    const int tiledCol = TS * t + col;
    Asub[row][col] = A[tiledCol + K * global_row];
    Bsub[row][col] = B[global_col + N * tiledRow];

    __syncthreads();

    // if (row == 0 && col == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   for (int r = 0; r < TS; ++r) {
    //     for (int c = 0; c < TS; ++c) {
    //       if (Asub[r][c] == 0) {
    //         printf("%d %d\n", r, c);
    //         // printf("%f\n", Asub[row][col]);
    //       }
    //     }
    //   }
    // }

    for(int k = 0; k < TS; k++) {
      c += Asub[row][k] * Bsub[k][col];
    }

    __syncthreads();
  }

  C[global_col + N * global_row] = c;
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  // A: M x K
  // B: K x N
  // C: M x N

  h_B = (float *) B;

  MPI_Bcast(h_B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter((float *) A, M * K / NUM_NODE, MPI_FLOAT, h_A, M * K / NUM_NODE, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    
    const int perM = M / NUM_NODE / NUM_GPU;
    CUDA_CALL(cudaMemcpyAsync(
      d_A[d], &h_A[d * perM * K], sizeof(float) * perM * K, cudaMemcpyHostToDevice, s_d[d]
    ));
    CUDA_CALL(cudaMemcpyAsync(
      d_B[d], h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice, s_d[d]
    ));

    dim3 dimBlock(TS, TS);
    dim3 dimGrid(N / TS, perM / TS);
    matmul_cal<<<dimGrid, dimBlock, 0, s_d[d]>>>(d_A[d], d_B[d], d_C[d], perM, N, K);
    CUDA_CALL(cudaMemcpyAsync(
      &h_C[d * perM * N], d_C[d],
      sizeof(float) * perM * N, cudaMemcpyDeviceToHost,
      s_d[d]
    ));
  }

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaStreamSynchronize(s_d[d]));
    CUDA_CALL(cudaDeviceSynchronize());
  }

  MPI_Gather(h_C, M * N / NUM_NODE, MPI_FLOAT, C, M * N / NUM_NODE, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void matmul_initialize(int M, int N, int K) {
  // TODO: FILL_IN_HERE
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CUDA_CALL(cudaMallocHost(&h_A, sizeof(float) * M * K / NUM_NODE));
  CUDA_CALL(cudaMallocHost(&h_B, sizeof(float) * K * N));
  CUDA_CALL(cudaMallocHost(&h_C, sizeof(float) * M * N / NUM_NODE));

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaMalloc(&d_A[d], sizeof(float) * M * K / NUM_NODE / NUM_GPU));
    CUDA_CALL(cudaMalloc(&d_B[d], sizeof(float) * K * N));
    CUDA_CALL(cudaMalloc(&d_C[d], sizeof(float) * M * N / NUM_NODE / NUM_GPU));  
    
    CUDA_CALL(cudaStreamCreate(&s_d[d]));
    CUDA_CALL(cudaEventCreate(&ev_d[d]));
  }
}

void matmul_finalize() {
  // TODO: FILL_IN_HERE

  CUDA_CALL(cudaFreeHost(h_A));
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C));

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaFree(d_A[d]));
    CUDA_CALL(cudaFree(d_B[d]));
    CUDA_CALL(cudaFree(d_C[d]));

    CUDA_CALL(cudaStreamDestroy(s_d[d]));
    CUDA_CALL(cudaEventDestroy(ev_d[d]));
  }
}
