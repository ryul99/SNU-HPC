#define _GNU_SOURCE
#include "matmul.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
    exit(EXIT_FAILURE);                                                        \
  }

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel, kernel_naive; //, kernel2;
static cl_mem a_d, b_d, c_d; //, ta_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  // float elapsed_time;
  const int SZ = 32;
  int isNaive = N % SZ || M % SZ || K % SZ;
  // if (N % SZ || M % SZ || K % SZ) {
  //   isNaive = 1;
  //   SZ = 1;
  // }

  // A: M x K
  // B: K x N
  // C: M x N

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &a_d);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &b_d);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &c_d);
  clSetKernelArg(kernel, 3, sizeof(int), (void*) &M);
  clSetKernelArg(kernel, 4, sizeof(int), (void*) &N);
  clSetKernelArg(kernel, 5, sizeof(int), (void*) &K);

  clSetKernelArg(kernel_naive, 0, sizeof(cl_mem), (void*) &a_d);
  clSetKernelArg(kernel_naive, 1, sizeof(cl_mem), (void*) &b_d);
  clSetKernelArg(kernel_naive, 2, sizeof(cl_mem), (void*) &c_d);
  clSetKernelArg(kernel_naive, 3, sizeof(int), (void*) &M);
  clSetKernelArg(kernel_naive, 4, sizeof(int), (void*) &N);
  clSetKernelArg(kernel_naive, 5, sizeof(int), (void*) &K);

  // clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*) &a_d);
  // clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*) &ta_d);
  // clSetKernelArg(kernel2, 2, sizeof(int), (void*) &M);
  // clSetKernelArg(kernel2, 3, sizeof(int), (void*) &K);

  clEnqueueWriteBuffer(queue, a_d, CL_FALSE, 0, M * K * sizeof(float), A, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, b_d, CL_FALSE, 0, N * K * sizeof(float), B, 0, NULL, NULL);

  const size_t global_work_size[2] = { N, M };
  const size_t local_work_size[2] = { SZ, SZ };
  const size_t local_work_size_naive[2] = { 1, 1 };
  // const size_t global_work_size2[2] = { K, N };
  // const size_t local_work_size2[2] = { SZ, SZ };
  
  // clFinish(queue);

  // printf("\n");
  // timer_start(1);
  // clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global_work_size2, local_work_size2, 0, NULL, NULL);
  // clFinish(queue);
  // printf("TP: %f sec\n", timer_stop(1));

  // timer_start(1);
  if (isNaive) {
    // printf("\nNaive\n");
    clEnqueueNDRangeKernel(queue, kernel_naive, 2, NULL, global_work_size, local_work_size_naive, 0, NULL, NULL);    
  } else {
    // printf("Tiling");
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);    
  }
  // clFinish(queue);
  // printf("CAL: %f sec\n", timer_stop(1));

  // timer_start(1);
  clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);
  // clFinish(queue);
  // printf("READ: %f sec\n", timer_stop(1));

  clFinish(queue);
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *)malloc(source_size + 1);
  size_t ntotal = 0;
  while (ntotal < source_size) {
    int nread = fread(source_code, sizeof(char), source_size, file);
    ntotal += nread;
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));
    char *log = (char *)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void matmul_initialize(int M, int N, int K) {
  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device (only 1)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "sgemm", &err);
  CHECK_ERROR(err);
  kernel_naive = clCreateKernel(program, "sgemm_naive", &err);
  CHECK_ERROR(err);
  // kernel2 = clCreateKernel(program, "transpose", &err);
  // CHECK_ERROR(err);

  // Create GPU buffers
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  // ta_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
  //                      &err);
  // CHECK_ERROR(err);
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
}

void matmul_finalize() {}
