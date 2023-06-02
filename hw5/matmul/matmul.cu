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
#define SUMMARY 1
#define NUM_ELEM 4096
#define NUM_BUFFER_ELEM 32
#define TS 32
#define BLOCK_ROWS 4
#define NUM_GPU 4
#define NUM_NODE 4
#define NUM_THREAD 256
#define NUM_OUTER_LOOP 8

float *h_A[NUM_OUTER_LOOP], *h_B, *h_C;
float *d_A[NUM_OUTER_LOOP], *d_B, *d_C[NUM_OUTER_LOOP];
cudaStream_t s_d[NUM_OUTER_LOOP];
// cudaEvent_t ev_buff[NUM_GPU][NUM_INNER_LOOP][2];
cudaEvent_t ev_d[NUM_OUTER_LOOP];
int mpi_rank, mpi_world_size, device_id;
MPI_Request req[NUM_OUTER_LOOP];

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

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  #if DEBUG
  if (row == 0 && col == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int row = 0; row < TS; ++row) {
      for (int col = 0; col < TS; ++col) {
        Asub[row][col] = 0;
        Bsub[row][col] = 0;
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

    for(int k = 0; k < TS; k++) {
      c += Asub[row][k] * Bsub[k][col];
    }

    __syncthreads();
  }

  C[global_col + N * global_row] = c;
  #if DEBUG
  printf("%d %d %f\n", global_row, global_col, c);
  #endif
}


void* gather_func(void *args) {
  struct matmul_args *arg = (struct matmul_args *) args;
  const int M = arg->M;
  const int N = arg->N;
  const int K = arg->K;
  float *C = arg->C;

  const int perM = M / NUM_GPU / NUM_NODE / NUM_OUTER_LOOP;

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    // spinlock
    while (cudaEventQuery(ev_d[l]) != cudaSuccess);
    MPI_Gather(
      &h_C[l * perM * N], perM * N, MPI_FLOAT,
      &C[l * perM * NUM_GPU * NUM_NODE * N], perM * N, MPI_FLOAT,
      0, MPI_COMM_WORLD
    );
  }
  return NULL;
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  // A: M x K
  // B: K x N
  // C: M x N
  //
  // M: Outer -> Node -> GPU

  #if SUMMARY
  if (mpi_rank == 0) {
    const int perM = 4 * NUM_ELEM / NUM_NODE / NUM_OUTER_LOOP / NUM_GPU;
    printf("dimBlock: %d %d\n", TS, TS);
    printf("dimGrid: %d %d\n", NUM_ELEM / TS, perM / TS);
    printf("perM: %d\n", perM);
    printf("NUM_OUTER_LOOP: %d\n", NUM_OUTER_LOOP);
  }
  #endif

  CUDA_CALL(cudaSetDevice(device_id));
  
  // create event & stream
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
      CUDA_CALL(cudaEventCreate(&ev_d[l]));
  }
  for (int st = 0; st < NUM_OUTER_LOOP; ++st) {
    CUDA_CALL(cudaStreamCreate(&s_d[st]));
  }

  h_B = (float *) B;
  // memset(h_C, 0, sizeof(float) * M * N);


  const int nodeM = M / NUM_NODE / NUM_OUTER_LOOP;
  const int perM = nodeM / NUM_GPU;

  MPI_Bcast(h_B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    MPI_Iscatter(
      &A[K * nodeM * NUM_NODE * l], K * perM, MPI_FLOAT,
      h_A[l], K * perM, MPI_FLOAT,
      0, MPI_COMM_WORLD, &req[l]
    );
  }
  
  CUDA_CALL(cudaMemcpyAsync(
    d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice, s_d[0]
  ));
  CUDA_CALL(cudaStreamSynchronize(s_d[0]));

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    #if DEBUG
    if (mpi_rank == 0) {
      printf("l: %d\n", l);
    }
    #endif
    
    MPI_Wait(&req[l], MPI_STATUSES_IGNORE);

    CUDA_CALL(cudaMemcpyAsync(
      d_A[l],
      h_A[l],
      sizeof(float) * perM * K, cudaMemcpyHostToDevice, s_d[l]
    ));


    dim3 dimBlock(TS, TS);
    dim3 dimGrid(N / TS, perM / TS);
    matmul_cal<<<dimGrid, dimBlock, 0, s_d[l]>>>(
      d_A[l], d_B, d_C[l], perM , N, K
    );
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaMemcpyAsync(
      &h_C[l * perM * N], d_C[l],
      sizeof(float) * perM * N, cudaMemcpyDeviceToHost,
      s_d[l]
    ));

    CUDA_CALL(cudaEventRecord(ev_d[l], s_d[l]));
  }
  pthread_t gather_thread;
  struct matmul_args *args = (struct matmul_args *) malloc(sizeof(struct matmul_args));
  args->M = M;
  args->N = N;
  args->K = K;
  args->C = C;
  pthread_create(&gather_thread, NULL, gather_func, args);
  pthread_join(gather_thread, NULL);
  
  // destroy event
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
      CUDA_CALL(cudaEventDestroy(ev_d[l]));
  }
  // destroy stream
    for (int st = 0; st < NUM_OUTER_LOOP; ++st) {
      CUDA_CALL(cudaStreamDestroy(s_d[st]));
    }
  
  #if DEBUG
  if (mpi_rank == 0) {
    printf("dimBlock: %d %d\n", TS, TS);
    printf("dimGrid: %d %d\n", N / TS, perM / TS);
    printf("perM: %d\n", perM);
  }
  #endif
}

void matmul_initialize(int M, int N, int K) {
  // TODO: FILL_IN_HERE
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  device_id = mpi_rank % NUM_GPU;
  CUDA_CALL(cudaSetDevice(device_id));

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaMallocHost(&h_A[l], sizeof(float) * M * K / NUM_GPU / NUM_NODE / NUM_OUTER_LOOP));
  }
  CUDA_CALL(cudaMallocHost(&h_B, sizeof(float) * K * N));
  CUDA_CALL(cudaMallocHost(&h_C, sizeof(float) * M * N / NUM_GPU / NUM_NODE));

  CUDA_CALL(cudaMalloc(&d_B, sizeof(float) * K * N));
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaMalloc(&d_A[l], sizeof(float) * M * K / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));
    CUDA_CALL(cudaMalloc(&d_C[l], sizeof(float) * M * N / NUM_NODE / NUM_GPU / NUM_OUTER_LOOP));
  }
}

void matmul_finalize() {
  // TODO: FILL_IN_HERE

  CUDA_CALL(cudaSetDevice(device_id));

  #if SUMMARY
  if (mpi_rank == 0) {
    const int perM = 4 * NUM_ELEM / NUM_NODE / NUM_OUTER_LOOP / NUM_GPU;
    printf("dimBlock: %d %d\n", TS, TS);
    printf("dimGrid: %d %d\n", NUM_ELEM / TS, perM / TS);
    printf("perM: %d\n", perM);
  }
  #endif

  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaFreeHost(h_A[l]));
  }
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C));

  CUDA_CALL(cudaFree(d_B));
  for (int l = 0; l < NUM_OUTER_LOOP; ++l) {
    CUDA_CALL(cudaFree(d_A[l]));
    CUDA_CALL(cudaFree(d_C[l]));
  }
}
