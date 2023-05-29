#include <stdio.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }


#define NUM_ELEM (1ul << 28)
#define NUM_BUFFER_ELEM (1ul << 20)

#define EPS (1e-6)

static float *h_A, *h_B, *h_C_naive, *h_C_buffered;
static float *d_A, *d_B, *d_C;
static cudaStream_t s0, s1, s2;
static cudaEvent_t ev0, ev1;

__global__ void fakeVecAdd(float *A, float *B, float *C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  C[i] = 0;
  for (int j = 0; j < 128; ++j) {
    C[i] += A[i] + B[i];
  }
}


void fakeVecAdd_naive()
{
  CUDA_CALL(cudaMemcpyAsync(
    d_A, h_A, sizeof(float) * NUM_ELEM, cudaMemcpyHostToDevice, s0));
  CUDA_CALL(cudaMemcpyAsync(
    d_B, h_B, sizeof(float) * NUM_ELEM, cudaMemcpyHostToDevice, s0));

  size_t numThreads = 256;
  size_t numBlocks = (NUM_ELEM + numThreads - 1) / numThreads;
  fakeVecAdd<<<numBlocks, numThreads, 0, s0>>>(d_A, d_B, d_C, NUM_ELEM);
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpyAsync(
    h_C_naive, d_C, sizeof(float) * NUM_ELEM, cudaMemcpyDeviceToHost, s0));

  CUDA_CALL(cudaStreamSynchronize(s0));
}

void fakeVecAdd_double_buffered()
{
  for (int off = 0; off < NUM_ELEM; off += NUM_BUFFER_ELEM) {
    CUDA_CALL(cudaMemcpyAsync(
        &d_A[off], &h_A[off], sizeof(float) * NUM_BUFFER_ELEM, cudaMemcpyHostToDevice, s0));
    CUDA_CALL(cudaMemcpyAsync(
        &d_B[off], &h_B[off], sizeof(float) * NUM_BUFFER_ELEM, cudaMemcpyHostToDevice, s0));

    CUDA_CALL(cudaEventRecord(ev0, s0));

    CUDA_CALL(cudaStreamWaitEvent(s1, ev0));

    size_t numThreads = 256;
    size_t numBlocks = (NUM_BUFFER_ELEM + numThreads - 1) / numThreads;
    fakeVecAdd<<<numBlocks, numThreads, 0, s1>>>(&d_A[off], &d_B[off], &d_C[off], NUM_BUFFER_ELEM);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(ev1, s1));

    CUDA_CALL(cudaStreamWaitEvent(s2, ev1));
    CUDA_CALL(cudaMemcpyAsync(
        &h_C_buffered[off], &d_C[off], sizeof(float) * NUM_BUFFER_ELEM, cudaMemcpyDeviceToHost, s2));
  }

  CUDA_CALL(cudaStreamSynchronize(s0));
  CUDA_CALL(cudaStreamSynchronize(s1));
  CUDA_CALL(cudaStreamSynchronize(s2));
}

int main(int argc, char *argv[])
{
  CUDA_CALL(cudaMallocHost(&h_A, sizeof(float) * NUM_ELEM));
  CUDA_CALL(cudaMallocHost(&h_B, sizeof(float) * NUM_ELEM));
  CUDA_CALL(cudaMallocHost(&h_C_naive, sizeof(float) * NUM_ELEM));
  CUDA_CALL(cudaMallocHost(&h_C_buffered, sizeof(float) * NUM_ELEM));

  CUDA_CALL(cudaMalloc(&d_A, sizeof(float) * NUM_ELEM));
  CUDA_CALL(cudaMalloc(&d_B, sizeof(float) * NUM_ELEM));
  CUDA_CALL(cudaMalloc(&d_C, sizeof(float) * NUM_ELEM));

  CUDA_CALL(cudaStreamCreate(&s0));
  CUDA_CALL(cudaStreamCreate(&s1));
  CUDA_CALL(cudaStreamCreate(&s2));


  CUDA_CALL(cudaEventCreate(&ev0));
  CUDA_CALL(cudaEventCreate(&ev1));

  for (size_t i = 0; i < NUM_ELEM; ++i) {
    h_A[i] = (float)rand() / RAND_MAX - 0.5;
    h_B[i] = (float)rand() / RAND_MAX - 0.5;
  }

  struct timespec s, e;

  //
  // 1. naive vector addition
  //
  clock_gettime(CLOCK_MONOTONIC, &s);
  fakeVecAdd_naive();
  clock_gettime(CLOCK_MONOTONIC, &e);
  printf("VecAdd_naive: %f ms\n",
      (e.tv_sec - s.tv_sec) * 1000.  + (e.tv_nsec - s.tv_nsec) / 1000000.);

  //
  // 2. double-buffered vector addition
  //
  clock_gettime(CLOCK_MONOTONIC, &s);
  fakeVecAdd_double_buffered();
  clock_gettime(CLOCK_MONOTONIC, &e);
  printf("fakeVecAdd_double_buffered: %f ms\n",
      (e.tv_sec - s.tv_sec) * 1000.  + (e.tv_nsec - s.tv_nsec) / 1000000.);


  printf("Validating ...\n");
  for (size_t i = 0; i < NUM_ELEM; ++i) {
    if (fabs(h_C_naive[i] - h_C_buffered[i]) >= EPS) {
      printf("[%lu] Validation failed: %f %f\n",
          i, h_C_naive[i], h_C_buffered[i]);
      exit(1);
    }
  }
  printf("Validation done\n");
  
  CUDA_CALL(cudaFreeHost(h_A));
  CUDA_CALL(cudaFreeHost(h_B));
  CUDA_CALL(cudaFreeHost(h_C_naive));
  CUDA_CALL(cudaFreeHost(h_C_buffered));

  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));

  CUDA_CALL(cudaStreamDestroy(s0));
  CUDA_CALL(cudaStreamDestroy(s1));
  CUDA_CALL(cudaStreamDestroy(s2));

  CUDA_CALL(cudaEventDestroy(ev0));
  CUDA_CALL(cudaEventDestroy(ev1));


  return 0;
}
