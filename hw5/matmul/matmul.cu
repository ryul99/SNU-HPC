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


#define DEBUG 0
#define NUM_ELEM 4096
#define NUM_BUFFER_ELEM 32
#define TS 16
#define BLOCK_ROWS 4
#define NUM_GPU 4
#define NUM_NODE 4
#define NUM_THREAD 256
#define NUM_OUTER_LOOP 4

float *h_A[NUM_OUTER_LOOP], *h_B, *h_C;
float *d_A[NUM_OUTER_LOOP][NUM_GPU], *d_B[NUM_GPU], *d_C[NUM_OUTER_LOOP][NUM_GPU];
cudaStream_t s_d[NUM_GPU];
cudaEvent_t ev_d[NUM_GPU];
int mpi_rank, mpi_world_size;
MPI_Request req[NUM_OUTER_LOOP];


__global__ void transposeFineGrained(float *dst, const float *src, const int width, const int height)
{
  // ref: https://developer.download.nvidia.com/assets/cuda/files/MatrixTranspose.pdf
  // x: TS, y: BLOCK_ROWS
  __shared__ float block[TS][TS + 1];
  int xIndex = blockIdx.x * TS + threadIdx.x;
  int yIndex = blockIdx.y * TS + threadIdx.y;
  int index = xIndex + (yIndex) * width;

  for (int i=0; i < TS; i += BLOCK_ROWS) {
    block[threadIdx.y+i][threadIdx.x] = src[index+i*width];
  }
  __syncthreads();
  for (int i=0; i < TS; i += BLOCK_ROWS) {
    dst[index+i*height] = block[threadIdx.x][threadIdx.y+i];
  }
}

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

  #if DEBUG
  if (row == 0 && col == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int row = 0; row < TS; ++row) {
      for (int col = 0; col < TS; ++col) {
        Asub[row][col] = 0;
      }
    }
  }
  #endif

  float c = 0.0;
  const int numTiles = K / TS;
  for (int t = 0; t < numTiles; ++t) {
    const int tiledRow = TS * t + row;
    const int tiledCol = TS * t + col;
    Asub[row][col] = A[tiledCol + K * global_row];
    Bsub[row][col] = B[global_col + N * tiledRow];

    __syncthreads();

    #if DEBUG
    if (row == 0 && col == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      for (int r = 0; r < TS; ++r) {
        for (int c = 0; c < TS; ++c) {
          if (Asub[r][c] == 0) {
            printf("%d %d\n", r, c);
            // printf("%f\n", Asub[row][col]);
          }
        }
      }
    }
    #endif

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

  const int nodeM = M / NUM_NODE / NUM_OUTER_LOOP;

  MPI_Bcast(h_B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Iscatter(
    &A[0], K * nodeM, MPI_FLOAT,
    h_A[0], K * nodeM, MPI_FLOAT,
    0, MPI_COMM_WORLD, &req[0]
  );

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    #if DEBUG
    if (mpi_rank == 0) {
      printf("l: %d\n", l);
    }
    #endif

    if (l < NUM_OUTER_LOOP - 1) {
      MPI_Iscatter(
        &A[K * nodeM * NUM_NODE * (l + 1)], K * nodeM, MPI_FLOAT,
        h_A[(l + 1)], K * nodeM, MPI_FLOAT,
        0, MPI_COMM_WORLD, &req[(l + 1)]
      );
    }
    
    MPI_Wait(&req[l], MPI_STATUS_IGNORE);
    
    for (int d = 0; d < NUM_GPU; ++d) {
      CUDA_CALL(cudaSetDevice(d));
      
      const int perM = nodeM / NUM_GPU;

      CUDA_CALL(cudaMemcpyAsync(
        d_A[l][d], &h_A[l][d * perM * K], sizeof(float) * perM * K, cudaMemcpyHostToDevice, s_d[d]
      ));
      CUDA_CALL(cudaMemcpyAsync(
        d_B[d], h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice, s_d[d]
      ));


      dim3 dimBlock(TS, TS);
      dim3 dimGrid(N / TS, perM / TS);

      #if DEBUG
      if (mpi_rank == 0) {
        printf("dimBlock: %d %d\n", TS, TS);
        printf("dimGrid: %d %d\n", N / TS, perM / TS);
      }
      #endif

      matmul_cal<<<dimGrid, dimBlock, 0, s_d[d]>>>(d_A[l][d], d_B[d], d_C[l][d], perM, N, K);
      CUDA_CALL(cudaMemcpyAsync(
        &h_C[d * perM * N + l * nodeM * N], d_C[l][d],
        sizeof(float) * perM * N, cudaMemcpyDeviceToHost,
        s_d[d]
      ));
    }
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

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaMallocHost(&h_A[l], sizeof(float) * M * K / NUM_NODE / NUM_OUTER_LOOP));
  }
  CUDA_CALL(cudaMallocHost(&h_B, sizeof(float) * K * N));
  CUDA_CALL(cudaMallocHost(&h_C, sizeof(float) * M * N / NUM_NODE));

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaMalloc(&d_B[d], sizeof(float) * K * N));
    for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
      CUDA_CALL(cudaMalloc(&d_A[l][d], sizeof(float) * M * K / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));
      CUDA_CALL(cudaMalloc(&d_C[l][d], sizeof(float) * M * N / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));  
    }  
    CUDA_CALL(cudaStreamCreate(&s_d[d]));
    CUDA_CALL(cudaEventCreate(&ev_d[d]));
  }
}

void matmul_finalize() {
  // TODO: FILL_IN_HERE
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaFreeHost(h_A[l]));
  }
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C));

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaFree(d_B[d]));
    for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
      CUDA_CALL(cudaFree(d_A[l][d]));
      CUDA_CALL(cudaFree(d_C[l][d]));
    }

    CUDA_CALL(cudaStreamDestroy(s_d[d]));
    CUDA_CALL(cudaEventDestroy(ev_d[d]));
  }
}
