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


#define MIN(a, b) ((a) < (b) ? (a) : (b))


#define DEBUG 0
#define KERNEL_DEBUG 0
#define SUMMARY 0
#define TS 32
#define BLOCK_ROWS 4
#define NUM_GPU 4
#define NUM_NODE 4
#define NUM_OUTER_LOOP 16
#define DIV_STREAM 1024
#define NUM_FUSION 1
#define NUM_MPI 2
// NUM_MPI should be devidor of NUM_OUTER_LOOP
#if NUM_MPI >= 1
  #define MPI_TS (NUM_OUTER_LOOP / NUM_MPI)
#else
  #define MPI_TS 1
#endif

#if NUM_MPI >= 1
float *h_A_buff[NUM_MPI];
#endif

float *h_A[NUM_OUTER_LOOP], *h_B, *h_C;
float *d_A[NUM_OUTER_LOOP / NUM_FUSION][NUM_GPU], *d_B[NUM_GPU], *d_C[NUM_OUTER_LOOP / NUM_FUSION][NUM_GPU];
cudaStream_t s_d[NUM_GPU][NUM_OUTER_LOOP / NUM_FUSION][3];
cudaEvent_t ev_buff[NUM_GPU][NUM_OUTER_LOOP / NUM_FUSION][2];
int mpi_rank, mpi_world_size;
MPI_Request req[NUM_OUTER_LOOP], reqB;


struct matmul_args {
  int M;
  int N;
  int K;
  float *C;
};



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

  // print global_col global_row
  // #if KERNEL_DEBUG
  // if (global_row > 2048)
  //   printf("global_col: %d, global_row: %d\n", global_col, global_row);
  // #endif

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  #if KERNEL_DEBUG
  if (row == 0 && col == 0 && global_row == 0 && global_col == 0) {
    for (int row = 0; row < TS; ++row) {
      for (int col = 0; col < TS; ++col) {
        Asub[row][col] = 0;
        Bsub[row][col] = 0;
      }
    }
    // check bound
    if (global_row >= M || global_col >= N) {
      printf("global_row: %d, global_col: %d\n", global_row, global_col);
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

    #if KERNEL_DEBUG
    if (Asub[row][col] == 0) {
      printf("Asub[%d][%d] = %f and Grid Idx: [%d][%d]\n", row, col, Asub[row][col], blockIdx.x * TS, blockIdx.y * TS);
    }
    if (Bsub[row][col] == 0) {
      printf("Bsub[%d][%d] = %f and Grid Idx: [%d][%d]\n", row, col, Bsub[row][col], blockIdx.x * TS, blockIdx.y * TS);
    }
    #endif

    for(int k = 0; k < TS; k++) {
      c += Asub[row][k] * Bsub[k][col];
    }

    __syncthreads();
  }
  #if KERNEL_DEBUG
  if (c == 0) {
    printf("C[%d][%d] = %f\n", global_row, global_col, c);
  }
  #endif
  C[global_col + N * global_row] = c;
}


void createEvent() {
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      for (int i = 0; i < 2; ++i)
        CUDA_CALL(cudaEventCreate(&ev_buff[d][l][i]));
    }
  }
}

void createStream() {
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      for (int i = 0; i < 3; ++i)
        CUDA_CALL(cudaStreamCreate(&s_d[d][l][i]));
    }
  }
}

void destroyEvent() {
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      for (int i = 0; i < 2; ++i)
        CUDA_CALL(cudaEventDestroy(ev_buff[d][l][i]));
    }
  }
}

void destroyStream() {
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      for (int i = 0; i < 3; ++i)
        CUDA_CALL(cudaStreamDestroy(s_d[d][l][i]));
    }
  }
}


void loadA(int K, int perM, int d, int l) {
  CUDA_CALL(cudaSetDevice(d));
  int d_A_d_idx = ((d + l * NUM_GPU) / NUM_FUSION) % NUM_GPU;
  #if DEBUG
  printf("d_A index: %d, in array index: %d, %d\n", l / NUM_FUSION, d_A_d_idx, ((d % NUM_FUSION)));
  printf("h_A index: %d, in array index: %d\n", l, d);
  #endif
  CUDA_CALL(cudaMemcpyAsync(
    &d_A[l / NUM_FUSION][d_A_d_idx][(d % NUM_FUSION) * perM * K],
    &h_A[l][d * perM * K],
    sizeof(float) * perM * K, cudaMemcpyHostToDevice, s_d[d_A_d_idx][l / NUM_FUSION][0]
  ));
  #if DEBUG
  printf("ev_buff[%d][%d][0]\n", d_A_d_idx, l / NUM_FUSION);
  printf("s_d[%d][%d][0]\n", d_A_d_idx, l / NUM_FUSION);
  printf("\n\n");
  #endif
  CUDA_CALL(cudaEventRecord(ev_buff[d_A_d_idx][l / NUM_FUSION][0], s_d[d_A_d_idx][l / NUM_FUSION][0]));
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  // A: M x K
  // B: K x N
  // C: M x N
  //
  // M: Outer -> Node -> GPU

  const int nodeM = M / NUM_NODE / NUM_OUTER_LOOP;
  const int perM = nodeM / NUM_GPU;
  #if NUM_MPI > 1
  int perNodeM = M / NUM_NODE / NUM_MPI;
  #endif

  #if SUMMARY
  if (mpi_rank == 0) {
    const int check = NUM_NODE * NUM_OUTER_LOOP * NUM_GPU * NUM_FUSION;
    printf("dimBlock: %d %d\n", TS, TS);
    printf("dimGrid: %d %d\n", N / TS, NUM_FUSION * perM / TS);
    printf("NUM_FUSION: %d\n", NUM_FUSION);
    printf("perM: %d\n", perM);
    printf("NUM_OUTER_LOOP: %d\n", NUM_OUTER_LOOP);
    printf("M should be bigger than %d and... validation: %d\n", check, M / check);
  }
  #endif

  createEvent();
  createStream();

  h_B = (float *) B;
  // memset(h_C, 0, sizeof(float) * M * N);

  #if NUM_MPI > 0
  MPI_Ibcast(h_B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &reqB);
  #if NUM_MPI == 1
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    h_A[l] = (float *) &h_A_buff[K * nodeM * l];
  }
  MPI_Iscatter(
    A, M * K / NUM_NODE, MPI_FLOAT,
    h_A_buff[0], M * K / NUM_NODE, MPI_FLOAT,
    0, MPI_COMM_WORLD, &req[0]
  );
  #else
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    h_A[l] = (float *) &h_A_buff[l / MPI_TS][K * nodeM * (l % MPI_TS)];
  }
  for (int l = 0; l < NUM_MPI; ++l) {
    MPI_Iscatter(
      &A[K * perNodeM * NUM_NODE * l], K * perNodeM, MPI_FLOAT,
      h_A_buff[l], K * perNodeM, MPI_FLOAT,
      0, MPI_COMM_WORLD, &req[l]
    );
  }
  #endif
  #else
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    h_A[l] = (float *) &A[K * nodeM * l];
  }
  h_C = C;
  #endif
  #if NUM_MPI > 0
  MPI_Wait(&reqB, MPI_STATUS_IGNORE);
  #if NUM_MPI == 1
  MPI_Wait(&req[0], MPI_STATUS_IGNORE);
  #endif
  #endif
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    
    CUDA_CALL(cudaMemcpyAsync(
      d_B[d], h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice, s_d[d][0][0]
    ));
  }
  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaStreamSynchronize(s_d[d][0][0]));
  }

  #if NUM_MPI > 0
  #pragma omp parallel for
  #endif
  for (int t = 0; t < NUM_OUTER_LOOP / NUM_FUSION; t += MPI_TS) {
    #if NUM_MPI
    #if (MPI_TS) >= NUM_FUSION
    MPI_Wait(
      &req[t / ((MPI_TS) / NUM_FUSION)], MPI_STATUS_IGNORE
    );
    #endif
    #endif
    for (int l = t; l < MIN(t + MPI_TS, NUM_OUTER_LOOP / NUM_FUSION); ++l) {
      #if NUM_MPI > 0
      #if (MPI_TS) < NUM_FUSION
      MPI_Waitall(
        NUM_FUSION / (MPI_TS),
        &req[l * NUM_FUSION / (MPI_TS)], MPI_STATUSES_IGNORE
      );
      #endif
      #endif

      for (int d = 0; d < NUM_GPU; ++d) {
        CUDA_CALL(cudaSetDevice(d));
        for (int i = 0; i < NUM_FUSION; ++i) {
          loadA(K, perM, d, l * NUM_FUSION + i);
        }
        #if DEBUG
        printf("s_d_Cal index: %d\n", l % DIV_STREAM);
        #endif
        dim3 dimBlock(TS, TS);
        dim3 dimGrid(N / TS, NUM_FUSION * perM / TS);
        CUDA_CALL(cudaStreamWaitEvent(s_d[d][l % DIV_STREAM][1], ev_buff[d][l][0]));

        matmul_cal<<<dimGrid, dimBlock, 0, s_d[d][l % DIV_STREAM][1]>>>(
          d_A[l][d], d_B[d], d_C[l][d], perM * NUM_FUSION, N, K
        );
        CUDA_CALL(cudaGetLastError());

        #if DEBUG
        printf("d_C index: %d, %d\n", l, d);
        #endif

        CUDA_CALL(cudaEventRecord(ev_buff[d][l][1], s_d[d][l % DIV_STREAM][1]));

        CUDA_CALL(cudaStreamWaitEvent(s_d[d][l % DIV_STREAM][2], ev_buff[d][l][1]));
        CUDA_CALL(cudaMemcpyAsync(
          &h_C[((d * perM + (l) * nodeM) * NUM_FUSION) * N], d_C[l][d],
          sizeof(float) * perM * N * NUM_FUSION, cudaMemcpyDeviceToHost,
          s_d[d][l % DIV_STREAM][2]
        ));
        #if DEBUG
        printf("h_C index: %d\n", ((d * perM + (l) * nodeM) * NUM_FUSION));
        #endif

        #if DEBUG
        printf("\n\n");
        #endif
      }
    }
  }
  #if NUM_MPI > 0
  #if NUM_MPI == 1
  MPI_Gather(
    h_C, M * N / NUM_NODE, MPI_FLOAT,
    C, M * N / NUM_NODE, MPI_FLOAT,
    0, MPI_COMM_WORLD
  );
  #else
  #pragma omp parallel for
  for (int l = 0; l < NUM_MPI; ++l) {
    for (int ll = l * MPI_TS; ll < MIN(NUM_OUTER_LOOP, (l + 1) * MPI_TS); ++ll) {
      int lll = ll / NUM_FUSION;
      for (int d = 0; d < NUM_GPU; ++d) {
        CUDA_CALL(cudaSetDevice(d));
        CUDA_CALL(cudaStreamSynchronize(s_d[d][lll % DIV_STREAM][2]));
      }
    }
    if (mpi_rank == 0) {
      for (int r = 1; r < mpi_world_size; ++r) {
        MPI_Recv(
          &C[(l * NUM_NODE + r) * perNodeM *  N], perNodeM * N, MPI_FLOAT,
          r, r + l * mpi_world_size, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
      }
      memcpy(&C[l * NUM_NODE * perNodeM * N], &h_C[l * perNodeM * N], perNodeM * N * sizeof(float));
    } else {
      MPI_Send(
        &h_C[l * perNodeM * N], perNodeM * N, MPI_FLOAT,
        0, mpi_rank + l * mpi_world_size, MPI_COMM_WORLD
      );
    } 
  }
  #endif
  #else
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    for (int d = 0; d < NUM_GPU; ++d) {
      CUDA_CALL(cudaSetDevice(d));
      CUDA_CALL(cudaStreamSynchronize(s_d[d][l / NUM_FUSION % DIV_STREAM][2]));
    }
  }
  #endif
  destroyEvent();
  destroyStream();
}

void warn_values() {
  // check NUM_NODE, NUM_GPU, NUM_FUSION, NUM_OUTER_LOOP with each rank
  if (NUM_NODE != mpi_world_size)
    printf("[WARN] (rank: %d) NODE => set: %d / current active: %d\n", mpi_rank, NUM_NODE, mpi_world_size);
  int num_gpu;
  CUDA_CALL(cudaGetDeviceCount(&num_gpu));
  if (NUM_GPU != num_gpu)
    printf("[WARN] (rank: %d) GPU => set: %d / current active: %d\n", mpi_rank, NUM_GPU, num_gpu);
  if (NUM_OUTER_LOOP % NUM_FUSION != 0 | NUM_OUTER_LOOP < NUM_FUSION)
    printf("[WARN] (rank: %d) NUM_OUTER_LOOP: %d, NUM_FUSION: %d\n", mpi_rank, NUM_OUTER_LOOP, NUM_FUSION);
  #if NUM_MPI
  if (NUM_OUTER_LOOP % (NUM_MPI * NUM_NODE) != 0 | NUM_OUTER_LOOP < NUM_MPI * NUM_NODE)
    printf("[WARN] (rank: %d) NUM_OUTER_LOOP: %d, NUM_MPI: %d, NUM_NODE: %d\n", mpi_rank, NUM_OUTER_LOOP, NUM_MPI, NUM_NODE);
  if (! (NUM_FUSION % MPI_TS == 0 || MPI_TS % NUM_FUSION == 0))
    printf("[WARN] (rank: %d) NUM_FUSION: %d, MPI_TS: %d\n", mpi_rank, NUM_FUSION, MPI_TS);
  #endif
}

void matmul_initialize(int M, int N, int K) {
  // TODO: FILL_IN_HERE
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  warn_values();

  // print NUM GPU
  #if SUMMARY
  int num_gpu;
  CUDA_CALL(cudaGetDeviceCount(&num_gpu));
  printf("(rank: %d) NUM_GPU: %d\n", mpi_rank, num_gpu);
  #endif
  #if NUM_MPI > 0
  CUDA_CALL(cudaMallocHost(&h_C, sizeof(float) * M * N / NUM_NODE));
  for (int i = 0; i < NUM_MPI; ++i)
    CUDA_CALL(cudaMallocHost(&h_A_buff[i], sizeof(float) * M * K / NUM_NODE / NUM_MPI));
  #endif

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaMalloc(&d_B[d], sizeof(float) * K * N));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      CUDA_CALL(cudaMalloc(&d_A[l][d], sizeof(float) * M * K  * NUM_FUSION / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));
      CUDA_CALL(cudaMalloc(&d_C[l][d], sizeof(float) * M * N  * NUM_FUSION / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));
    }
  }
}

void matmul_finalize() {
  // TODO: FILL_IN_HERE
  #if NUM_MPI > 0
  CUDA_CALL(cudaFreeHost(h_C));
  for (int i = 0; i < NUM_MPI; ++i)
    CUDA_CALL(cudaFreeHost(h_A_buff[i]));
  #endif

  for (int d = 0; d < NUM_GPU; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    CUDA_CALL(cudaFree(d_B[d]));
    for (int l = 0; l < NUM_OUTER_LOOP / NUM_FUSION; ++l) {
      CUDA_CALL(cudaFree(d_A[l][d]));
      CUDA_CALL(cudaFree(d_C[l][d]));
    }
  }
}
