#include "translator.h"
#include "util.h"
#include <mpi.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define SANITY_CHECK(t,d,s_idx)                                                     \
  {                                                                           \
    int ret = ((t)->check_values(d,s_idx));                                         \
    if (ret != 0) {                                                           \
      fprintf(stderr, "[%s:%d] Sanity check failed at [%s]\n", __FILE__,      \
              __LINE__, #t);                                                  \
      exit(1);                                                                \
    }                                                                         \
  }                                                                           \

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define NUM_STREAMS 4
#define NUM_GPUS 4
#define DEBUG 0

#define SOS_token 0
#define EOS_token 1
#define HIDDEN_SIZE 256
#define NUM_THREADS 128
#define INPUT_VOCAB_SIZE 4345
#define OUTPUT_VOCAB_SIZE 2803
cudaStream_t stream[NUM_STREAMS][NUM_GPUS];
pthread_t thread[NUM_STREAMS][NUM_GPUS];

/*
 * Tensor 
 * @brief : A multi-dimensional matrix containing elements of a single data type.
 *
 * @member buf    : Data buffer containing elements
 * @member shape  : Shape of tensor from outermost dimension to innermost dimension
                    - e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) { 
    shape[i] = shape_[i]; 
  }
  int N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));
  CUDA_CALL(cudaMalloc(&d_buf, N_ * sizeof(float)));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) { 
    shape[i] = shape_[i]; 
  }
  int N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  CUDA_CALL(cudaMalloc(&d_buf, N_ * sizeof(float)));
  memcpy(buf, buf_, N_ * sizeof(float));
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
  if (d_buf != nullptr) CUDA_CALL(cudaFree(d_buf));
}

int Tensor::num_elem() {
  int sz = 1;
  for (int i=0; i<ndim; ++i){
    sz *= shape[i];
  }
  return sz;
}

void Tensor::fill_zeros(int d) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = num_elem();
  memset(buf, 0, N_ * sizeof(float));
  CUDA_CALL(cudaMemset(d_buf, 0, N_ * sizeof(float)));
}

void Tensor::to_device(int d) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = num_elem();
  CUDA_CALL(cudaMemcpy(d_buf, buf, N_ * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::to_host(int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = num_elem();
  CUDA_CALL(cudaMemcpyAsync(buf, d_buf, N_ * sizeof(float), cudaMemcpyDeviceToHost, stream[s_idx][d]));
  CUDA_CALL(cudaStreamSynchronize(stream[s_idx][d]));
}

void Tensor::copy_from(Tensor *src) {
  int N_ = num_elem();
  if (N_ != src->num_elem()) {
    fprintf(stderr, "Tensor::copy_from() : size mismatch\n");
    exit(1);
  }
  // memcpy(buf, src->buf, N_ * sizeof(float));
  buf = src->buf;
}

void Tensor::print_device_value(int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  CUDA_CALL(cudaStreamSynchronize(stream[s_idx][d]));
  int N_ = num_elem();
  float *tmp = (float *)malloc(N_ * sizeof(float));
  CUDA_CALL(cudaMemcpy(tmp, d_buf, N_ * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i=0; i<N_; ++i) {
    printf("%f ", tmp[i]);
  }
  printf("\n");
  free(tmp);
}

int Tensor::check_values(int d, int s_idx) {
  // check if buf and buf_d is same
  CUDA_CALL(cudaSetDevice(d));
  CUDA_CALL(cudaStreamSynchronize(stream[s_idx][d]));
  int N_ = num_elem();
  float *tmp = (float *)malloc(N_ * sizeof(float));
  CUDA_CALL(cudaMemcpy(tmp, d_buf, N_ * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i=0; i<N_; ++i) {
    if (abs(tmp[i] - buf[i]) > 1e-3) {
      fprintf(stderr, "ERROR: buf: %f, d_buf: %f\n", buf[i], tmp[i]);
      return 1;
    }
  }
  return 0;
}

// Parameters
Tensor *eW_emb_raw;
Tensor *eW_ir_raw;
Tensor *eW_iz_raw;
Tensor *eW_in_raw;
Tensor *eW_hr_raw;
Tensor *eW_hz_raw;
Tensor *eW_hn_raw;
Tensor *eb_ir_raw;
Tensor *eb_iz_raw;
Tensor *eb_in_raw;
Tensor *eb_hr_raw;
Tensor *eb_hz_raw;
Tensor *eb_hn_raw;
Tensor *dW_emb_raw;
Tensor *dW_ir_raw;
Tensor *dW_iz_raw;
Tensor *dW_in_raw;
Tensor *dW_hr_raw;
Tensor *dW_hz_raw;
Tensor *dW_hn_raw;
Tensor *db_ir_raw;
Tensor *db_iz_raw;
Tensor *db_in_raw;
Tensor *db_hr_raw;
Tensor *db_hz_raw;
Tensor *db_hn_raw;
Tensor *dW_attn_raw;
Tensor *db_attn_raw;
Tensor *dW_attn_comb_raw;
Tensor *db_attn_comb_raw;
Tensor *dW_out_raw;
Tensor *db_out_raw;

MPI_Request req[33];

Tensor *eW_emb[NUM_GPUS];
Tensor *eW_ir[NUM_GPUS], *eW_iz[NUM_GPUS], *eW_in[NUM_GPUS];
Tensor *eW_hr[NUM_GPUS], *eW_hz[NUM_GPUS], *eW_hn[NUM_GPUS];
Tensor *eb_ir[NUM_GPUS], *eb_iz[NUM_GPUS], *eb_in[NUM_GPUS];
Tensor *eb_hr[NUM_GPUS], *eb_hz[NUM_GPUS], *eb_hn[NUM_GPUS];
Tensor *dW_emb[NUM_GPUS];
Tensor *dW_ir[NUM_GPUS], *dW_iz[NUM_GPUS], *dW_in[NUM_GPUS];
Tensor *dW_hr[NUM_GPUS], *dW_hz[NUM_GPUS], *dW_hn[NUM_GPUS];
Tensor *db_ir[NUM_GPUS], *db_iz[NUM_GPUS], *db_in[NUM_GPUS];
Tensor *db_hr[NUM_GPUS], *db_hz[NUM_GPUS], *db_hn[NUM_GPUS];
Tensor *dW_attn[NUM_GPUS], *db_attn[NUM_GPUS], *dW_attn_comb[NUM_GPUS], *db_attn_comb[NUM_GPUS], *dW_out[NUM_GPUS], *db_out[NUM_GPUS];

// Encoder Activations
Tensor *encoder_hidden[NUM_STREAMS][NUM_GPUS], *encoder_outputs[NUM_STREAMS][NUM_GPUS];
Tensor *encoder_embedded[NUM_STREAMS][NUM_GPUS];
Tensor *encoder_rtmp2[NUM_STREAMS][NUM_GPUS], *encoder_rtmp4[NUM_STREAMS][NUM_GPUS], *encoder_rt[NUM_STREAMS][NUM_GPUS];
Tensor *encoder_ztmp2[NUM_STREAMS][NUM_GPUS], *encoder_ztmp4[NUM_STREAMS][NUM_GPUS], *encoder_zt[NUM_STREAMS][NUM_GPUS];
Tensor *encoder_ntmp2[NUM_STREAMS][NUM_GPUS], *encoder_ntmp4[NUM_STREAMS][NUM_GPUS], *encoder_ntmp5[NUM_STREAMS][NUM_GPUS], *encoder_nt[NUM_STREAMS][NUM_GPUS];
Tensor *encoder_htmp1[NUM_STREAMS][NUM_GPUS], *encoder_htmp2[NUM_STREAMS][NUM_GPUS], *encoder_htmp3[NUM_STREAMS][NUM_GPUS], *encoder_ht[NUM_STREAMS][NUM_GPUS];

// Decoder Activations
Tensor *decoder_input[NUM_STREAMS][NUM_GPUS], *decoder_output[NUM_STREAMS][NUM_GPUS], *decoder_hidden[NUM_STREAMS][NUM_GPUS], *decoded_words[NUM_STREAMS][NUM_GPUS], *decoder_embedded[NUM_STREAMS][NUM_GPUS], *decoder_embhid[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_attn[NUM_STREAMS][NUM_GPUS], *decoder_attn_weights[NUM_STREAMS][NUM_GPUS], *decoder_attn_applied[NUM_STREAMS][NUM_GPUS], *decoder_embattn[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_attn_comb[NUM_STREAMS][NUM_GPUS], *decoder_relu[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_rtmp2[NUM_STREAMS][NUM_GPUS], *decoder_rtmp4[NUM_STREAMS][NUM_GPUS], *decoder_rt[NUM_STREAMS][NUM_GPUS]; 
Tensor *decoder_ztmp2[NUM_STREAMS][NUM_GPUS], *decoder_ztmp4[NUM_STREAMS][NUM_GPUS], *decoder_zt[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_ntmp2[NUM_STREAMS][NUM_GPUS], *decoder_ntmp4[NUM_STREAMS][NUM_GPUS], *decoder_ntmp5[NUM_STREAMS][NUM_GPUS], *decoder_nt[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_htmp1[NUM_STREAMS][NUM_GPUS], *decoder_htmp2[NUM_STREAMS][NUM_GPUS], *decoder_htmp3[NUM_STREAMS][NUM_GPUS], *decoder_ht[NUM_STREAMS][NUM_GPUS];
Tensor *decoder_out[NUM_STREAMS][NUM_GPUS], *decoder_logsoftmax[NUM_STREAMS][NUM_GPUS];

// Operations
void embedding(int ei, Tensor *weight, Tensor *output, int d, int s_idx);
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, int d, int s_idx);
void linear2_add_sigmoid(Tensor *input1, Tensor *input2, Tensor *weight1, Tensor *weight2, Tensor *bias1, Tensor *bias2, Tensor *output, int d, int s_idx);
void elemwise_add_sigmoid(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx);
void elemwise_add_tanh(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx);
void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx);
void elemwise_oneminus(Tensor *input, Tensor *output, int d, int s_idx);
void copy_encoder_outputs(Tensor *input, Tensor *output, int i, int d, int s_idx);
void concat(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx);
void softmax(Tensor *input, Tensor *output, int d, int s_idx);
void bmm(Tensor *input, Tensor *weight, Tensor *output, int d, int s_idx);
void relu(Tensor *input, Tensor *output, int d, int s_idx);
int  log_top_one(Tensor *input, int d, int s_idx);
void load_parameters(int d);
void n_t_part(Tensor *input1, Tensor *input2, Tensor *weight1, Tensor *weight2, Tensor *bias1, Tensor *bias2, Tensor *input_mul, Tensor *output, int d, int s_idx);
void h_t_part(Tensor *input_zt, Tensor *input_nt, Tensor *hidden, int d, int s_idx);

__global__ void linear_cal(const float *input, const float *weight, const float *bias, float *output, int M, int N);
__global__ void bmm_cal(const float *input, const float *weight, float *output, int K, int N);
__global__ void elemwise_add_cal(const float *input1, const float *input2, float *output, int M);
__global__ void elemwise_add_sigmoid_cal(const float *input1, const float *input2, float *output, int M);
__global__ void elemwise_add_tanh_cal(const float *input1, const float *input2, float *output, int M);
__global__ void elemwise_mult_cal(const float *input1, const float *input2, float *output, int M);
__global__ void elemwise_oneminus_cal(const float *input, float *output, int M);
__global__ void softmax_cal(const float *input, float *output, float *sum, int M);
__global__ void relu_cal(const float *input, float *output, int M);
__global__ void linear2_add_sigmoid_cal(const float *input1, const float *input2, const float *weight1, const float *weight2, const float *bias1, const float *bias2, float *output, int M, int K);
__global__ void n_t_part_cal(const float *input1, const float *input2, const float *weight1, const float *weight2, const float *bias1, const float *bias2, const float *input_mul, float *output, int M, int K);
__global__ void h_t_part_cal(const float *input_zt, const float *input_nt, float *hidden, int M);

typedef struct {
  int d;
  int s_idx;
  int start;
  int end;
  Tensor *input;
  Tensor *output;
} trans_args;


template<bool is_first>
__global__ void reduce_sum_cal(const float *input, float *output, int M) {
  __shared__ float smem[NUM_THREADS];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // load input into __shared__ memory
  if (is_first) {
    smem[tid] = (i < M) ? expf(input[i]) : 0;
  } else {
    smem[tid] = input[i];
  }
  __syncthreads();
  for (int s = (blockDim.x >> 1); s > 0; s = s >> 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  // result of this block
  if (tid == 0) output[blockIdx.x] += smem[tid];
}

template __global__ void reduce_sum_cal<true>(const float *input, float *output, int M);
template __global__ void reduce_sum_cal<false>(const float *input, float *output, int M);

void *translate(void *args) {
  trans_args *arg = (trans_args *) args;
  int d = arg->d;
  int s_idx = arg->s_idx;
  int start = arg->start;
  int end = arg->end;
  Tensor *input = arg->input;
  Tensor *output = arg->output;

  CUDA_CALL(cudaSetDevice(d));

  for (int n = start; n < end; ++n) {
    // Encoder init
    int input_length = 0;
    for (int i=0; i<MAX_LENGTH; ++i, ++input_length) {
      if (input->buf[n * MAX_LENGTH + i] == 0.0) break;
    }
    encoder_hidden[s_idx][d]->fill_zeros(d);
    encoder_outputs[s_idx][d]->fill_zeros(d);

    // Encoder
    for (int i=0; i<input_length; ++i) {
      
      // Embedding
      int ei = input->buf[n * MAX_LENGTH + i];
      embedding(ei, eW_emb[d], encoder_embedded[s_idx][d], d, s_idx);
      // GRU
      // r_t
      linear2_add_sigmoid(encoder_embedded[s_idx][d], encoder_hidden[s_idx][d], eW_ir[d], eW_hr[d], eb_ir[d], eb_hr[d], encoder_rt[s_idx][d], d, s_idx);

      // z_t
      linear2_add_sigmoid(encoder_embedded[s_idx][d], encoder_hidden[s_idx][d], eW_iz[d], eW_hz[d], eb_iz[d], eb_hz[d], encoder_zt[s_idx][d], d, s_idx);

      // n_t
      n_t_part(encoder_embedded[s_idx][d], encoder_hidden[s_idx][d], eW_in[d], eW_hn[d], eb_in[d], eb_hn[d], encoder_rt[s_idx][d], encoder_nt[s_idx][d], d, s_idx);

      // h_t
      h_t_part(encoder_zt[s_idx][d], encoder_nt[s_idx][d], encoder_hidden[s_idx][d], d, s_idx);

      copy_encoder_outputs(encoder_hidden[s_idx][d], encoder_outputs[s_idx][d], i, d, s_idx);
    } // end Encoder loop

    // Decoder init
    decoder_hidden[s_idx][d] = encoder_hidden[s_idx][d];
    decoder_input[s_idx][d]->buf[0] = SOS_token; 
    int di = (int)decoder_input[s_idx][d]->buf[0];
    // Decoder
    for (int i=0; i<MAX_LENGTH; ++i) {

      // Embedding
      embedding(di, dW_emb[d], decoder_embedded[s_idx][d], d, s_idx);
      // Attention
      concat(decoder_embedded[s_idx][d], decoder_hidden[s_idx][d], decoder_embhid[s_idx][d], d, s_idx);
      linear(decoder_embhid[s_idx][d], dW_attn[d], db_attn[d], decoder_attn[s_idx][d], d, s_idx);
      softmax(decoder_attn[s_idx][d], decoder_attn_weights[s_idx][d], d, s_idx);
      bmm(decoder_attn_weights[s_idx][d], encoder_outputs[s_idx][d], decoder_attn_applied[s_idx][d], d, s_idx);
      concat(decoder_embedded[s_idx][d], decoder_attn_applied[s_idx][d], decoder_embattn[s_idx][d], d, s_idx);
      linear(decoder_embattn[s_idx][d], dW_attn_comb[d], db_attn_comb[d], decoder_attn_comb[s_idx][d], d, s_idx);
      relu(decoder_attn_comb[s_idx][d], decoder_relu[s_idx][d], d, s_idx);

      // GRU
      // r_t
      linear2_add_sigmoid(decoder_relu[s_idx][d], decoder_hidden[s_idx][d], dW_ir[d], dW_hr[d], db_ir[d], db_hr[d], decoder_rt[s_idx][d], d, s_idx);

      // z_t
      linear2_add_sigmoid(decoder_relu[s_idx][d], decoder_hidden[s_idx][d], dW_iz[d], dW_hz[d], db_iz[d], db_hz[d], decoder_zt[s_idx][d], d, s_idx);
      
      // n_t
      n_t_part(decoder_relu[s_idx][d], decoder_hidden[s_idx][d], dW_in[d], dW_hn[d], db_in[d], db_hn[d], decoder_rt[s_idx][d], decoder_nt[s_idx][d], d, s_idx);

      // h_t
      h_t_part(decoder_zt[s_idx][d], decoder_nt[s_idx][d], decoder_hidden[s_idx][d], d, s_idx);

      // Select output token
      linear(decoder_hidden[s_idx][d], dW_out[d], db_out[d], decoder_out[s_idx][d], d, s_idx);
      softmax(decoder_out[s_idx][d], decoder_logsoftmax[s_idx][d], d, s_idx);
      int topi = log_top_one(decoder_logsoftmax[s_idx][d], d, s_idx);
      if (topi != EOS_token) {
        output->buf[n * MAX_LENGTH + i] = topi;
        di = topi;
      }
      else {
        output->buf[n * MAX_LENGTH + i] = EOS_token;
        break;
      }
    } // end Decoder loop
  }
  return NULL;
}

/*
 * translator 
 * @brief : French to English translator. 
 *          Translate N sentences in French into N sentences in English
 *
 * @param [in1] input  : a tensor of size [N x MAX_LENGTH]. French tokens are stored in this tensor.
 * @param [out] output : a tensor of size [N x MAX_LENGTH]. English tokens will be stored in this tensor.
 */
void translator(Tensor *input, Tensor *output, int N){
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  Tensor *mpi_input = new Tensor({N, MAX_LENGTH});
  Tensor *mpi_output = new Tensor({N, MAX_LENGTH});

  int cnt = 0;

  MPI_Ibcast(eW_emb_raw->buf, eW_emb_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_ir_raw->buf, eW_ir_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_iz_raw->buf, eW_iz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_in_raw->buf, eW_in_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_hr_raw->buf, eW_hr_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_hz_raw->buf, eW_hz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eW_hn_raw->buf, eW_hn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_ir_raw->buf, eb_ir_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_iz_raw->buf, eb_iz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_in_raw->buf, eb_in_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_hr_raw->buf, eb_hr_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_hz_raw->buf, eb_hz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(eb_hn_raw->buf, eb_hn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_emb_raw->buf, dW_emb_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_ir_raw->buf, dW_ir_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_iz_raw->buf, dW_iz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_in_raw->buf, dW_in_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_hr_raw->buf, dW_hr_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_hz_raw->buf, dW_hz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_hn_raw->buf, dW_hn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_ir_raw->buf, db_ir_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_iz_raw->buf, db_iz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_in_raw->buf, db_in_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_hr_raw->buf, db_hr_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_hz_raw->buf, db_hz_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_hn_raw->buf, db_hn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_attn_raw->buf, dW_attn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_attn_raw->buf, db_attn_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_attn_comb_raw->buf, dW_attn_comb_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_attn_comb_raw->buf, db_attn_comb_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(dW_out_raw->buf, dW_out_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);
  MPI_Ibcast(db_out_raw->buf, db_out_raw->num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);

  MPI_Iscatter(input->buf, input->num_elem() / mpi_size, MPI_FLOAT, mpi_input->buf, mpi_input->num_elem() / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD, &req[cnt++]);

  MPI_Waitall(cnt, req, MPI_STATUSES_IGNORE);

  for (int d = 0; d < NUM_GPUS; ++d) {
    load_parameters(d);
    mpi_input->to_device(d);

    int start = d * N / mpi_size / NUM_GPUS;
    int end = (d + 1) * N / mpi_size / NUM_GPUS;
    for (int s_idx = 0; s_idx < NUM_STREAMS; ++s_idx) {
      int s_start = s_idx * (end - start) / NUM_STREAMS + start;
      int s_end = (s_idx + 1) * (end - start) / NUM_STREAMS + start;
      trans_args *args = new trans_args();
      args->d = d;
      args->s_idx = s_idx;
      args->start = s_start;
      args->end = s_end;
      args->input = mpi_input;
      args->output = mpi_output;
      pthread_create(&thread[s_idx][d], NULL, translate, args);
    }
  }
  for (int s_idx = 0; s_idx < NUM_STREAMS; ++s_idx) {
    for (int d = 0; d < NUM_GPUS; ++d) {
      pthread_join(thread[s_idx][d], NULL);
    }
  }
  MPI_Gather(mpi_output->buf, mpi_output->num_elem() / mpi_size, MPI_FLOAT, output->buf, output->num_elem() / mpi_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

/*
 * embedding 
 * @brief : A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * @param [in1] ei     : embedding index
 * @param [in2] weight : a matrix of size [M x H_]
 * @param [out] output : a vector of size [H_]
 */
void embedding(int ei, Tensor *weight, Tensor *output, int d, int s_idx){
  CUDA_CALL(cudaSetDevice(d));

  int H_ = weight->shape[1];
  #if DEBUG
  memcpy(output->buf, &weight->buf[ei * H_], H_ * sizeof(float));
  #endif
  CUDA_CALL(cudaMemcpyAsync(output->d_buf, &weight->d_buf[ei * H_], H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream[s_idx][d]));
}

/*
 * elemwise_add
 * @brief : Element-by-element addition of tensors
 *
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx){
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input1->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    output->buf[n] = input1->buf[n] + input2->buf[n];
  }
  #endif
  elemwise_add_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, output->d_buf, N_);
}

/*
 * elemwise_add_sigmoid
 * @brief : Element-by-element addition of tensors and sigmoid
 *        output = sigmoid(input1 + input2)
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
void elemwise_add_sigmoid(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx){
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input1->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    output->buf[n] = 1.0 / (1.0 + exp(-(input1->buf[n] + input2->buf[n])));
  }
  #endif
  elemwise_add_sigmoid_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, output->d_buf, N_);
}

/*
 * elemwise_add_tanh
 * @brief : Element-by-element addition of tensors and tanh
 *       output = tanh(input1 + input2)
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
void elemwise_add_tanh(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx){
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input1->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    output->buf[n] = tanhf(input1->buf[n] + input2->buf[n]);
  }
  #endif
  elemwise_add_tanh_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, output->d_buf, N_);
}

/*
 * linear
 * @brief : Perform a matrix-vector product of the matrix and the vector and add a vector
 *         to the result
 * @param [in1] input  : a vector of size [K_]
 * @param [in2] weight  : a matrix of size [M_ x K_]
 * @param [in3] bias  : a vector of size [M_]
 * @param [out] output  : a vector of size [M_]
 * @return : output = weight * input + bias
 */
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int M_ = weight->shape[0];
  int K_ = weight->shape[1];
  
  #if DEBUG
  for (int m=0; m<M_; ++m) {
    float c = 0.0;
    for (int k=0; k<K_; ++k) {
      float w = weight->buf[m*K_+k];
      float i1 = input->buf[k];
      c += w*i1;
    }
    output->buf[m] = c + bias->buf[m];
  }
  #endif

  linear_cal<<<(M_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, weight->d_buf, bias->d_buf, output->d_buf, M_, K_);
}

void linear2_add_sigmoid(Tensor *input1, Tensor *input2, Tensor *weight1, Tensor *weight2, Tensor *bias1, Tensor *bias2, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int M_ = weight1->shape[0];
  int K_ = weight1->shape[1];

  #if DEBUG
  for (int m=0; m<M_; ++m) {
    float c = 0.0;
    for (int k=0; k<K_; ++k) {
      float w1 = weight1->buf[m*K_+k];
      float w2 = weight2->buf[m*K_+k];
      float i1 = input1->buf[k];
      float i2 = input2->buf[k];
      c += w1*i1 + w2*i2;
    }
    output->buf[m] = 1.0 / (1.0 + exp(-(c + bias1->buf[m] + bias2->buf[m])));
  }
  #endif

  linear2_add_sigmoid_cal<<<(M_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, weight1->d_buf, weight2->d_buf, bias1->d_buf, bias2->d_buf, output->d_buf, M_, K_);
}

void n_t_part(Tensor *input1, Tensor *input2, Tensor *weight1, Tensor *weight2, Tensor *bias1, Tensor *bias2, Tensor *input_mul, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int M_ = weight1->shape[0];
  int K_ = weight1->shape[1];

  #if DEBUG
  for (int m=0; m<M_; ++m) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    for (int k=0; k<K_; ++k) {
      float w1 = weight1->buf[m*K_+k];
      float w2 = weight2->buf[m*K_+k];
      float i1 = input1->buf[k];
      float i2 = input2->buf[k];
      sum1 += w1*i1
      sum2 += w2*i2;
    }
    sum1 += bias1->buf[m];
    sum2 += bias2->buf[m];
    sum2 *= input_mul->buf[m];
    output->buf[m] = tanhf(sum1 + sum2);
  }
  #endif

  n_t_part_cal<<<(M_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, weight1->d_buf, weight2->d_buf, bias1->d_buf, bias2->d_buf, input_mul->d_buf, output->d_buf, M_, K_);
}

void h_t_part(Tensor *input_zt, Tensor *input_nt, Tensor *hidden, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int M_ = input_zt->shape[0];

  #if DEBUG
  for (int m=0; m<M_; ++m) {
    float zt = input_zt->buf[m];
    float nt = input_nt->buf[m];
    float h = hidden->buf[m];
    hidden->buf[m] = zt*h + (1.0 - zt)*nt;
  }
  #endif

  h_t_part_cal<<<(M_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input_zt->d_buf, input_nt->d_buf, hidden->d_buf, M_);
}

/*
 * elemwise_mult
 * @brief : Element-by-element multiplication of tensors.
 *
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input1->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    float x1 = input1->buf[n];
    float x2 = input2->buf[n];
    output->buf[n] = x1 * x2;
  }
  #endif
  elemwise_mult_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input1->d_buf, input2->d_buf, output->d_buf, N_);
}

/*
 * elemwise_oneminus
 * @brief : Apply the element-wise oneminus function. oneminus(x) = 1.0 - x
 *
 * @param [in1] input
 * @param [out] output
 */
void elemwise_oneminus(Tensor *input, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    float x = input->buf[n];
    output->buf[n] = 1.0 - x;
  }
  #endif
  elemwise_oneminus_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, output->d_buf, N_);
}

/*
 * copy_encoder_outputs
 * @brief : Copy input vector into i-th row of the output matrix
 *
 * @param [in1] input  : a vector of size [N_]
 * @param [in2] i      : row index
 * @param [out] output : a matrix of size [MAX_LENGTH x N_]
 */
void copy_encoder_outputs(Tensor *input, Tensor *output, int i, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input->num_elem();
  
  #if DEBUG
  memcpy(&output->buf[i * HIDDEN_SIZE], input->buf, N_ * sizeof(float));
  #endif
  CUDA_CALL(cudaMemcpyAsync(&output->d_buf[i * HIDDEN_SIZE], input->d_buf, N_ * sizeof(float), cudaMemcpyDeviceToDevice, stream[s_idx][d]));
}

/*
 * concat
 * @brief : Concatenate the two input tensors
 *
 * @param [in1] input1 : a vector of size [N_]
 * @param [in2] input2 : a vector of size [N_]
 * @param [out] output : a vector of size [2*N_]
 */
void concat(Tensor *input1, Tensor *input2, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input1->num_elem();

  #if DEBUG
  memcpy(output->buf, input1->buf, N_ * sizeof(float));
  memcpy(&output->buf[N_], input2->buf, N_ * sizeof(float));
  #endif
  CUDA_CALL(cudaMemcpyAsync(output->d_buf, input1->d_buf, N_ * sizeof(float), cudaMemcpyDeviceToDevice, stream[s_idx][d]));
  CUDA_CALL(cudaMemcpyAsync(&output->d_buf[N_], input2->d_buf, N_ * sizeof(float), cudaMemcpyDeviceToDevice, stream[s_idx][d]));
}

/*
 * softmax
 * @brief : Apply the Softmax function to an n-dimensional input Tensor rescaling them 
 *          so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
 *          softmax(xi) = exp(xi) / sum of exp(xi)
 *
 * @param [in1] input  : a vector of size [N_]
 * @param [out] output : a vector of size [N_]
 */
void softmax(Tensor *input, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input->shape[0];
  #if DEBUG
  float sum = 0.0;

  for (int n=0; n<N_; ++n) {
    sum += expf(input->buf[n]);
  }
  for (int n=0; n<N_; ++n) {
    output->buf[n] = expf(input->buf[n]) / sum;
  }
  #endif

  float *dmem;
  float *tmp = output->d_buf;
  CUDA_CALL(cudaMalloc(&dmem, N_ * sizeof(float)));
  reduce_sum_cal<true><<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, output->d_buf, N_);
  // (l + NUM_THREADS - 1) / NUM_THREADS << This is num reduced value
  for (int l = (N_ + NUM_THREADS - 1) / NUM_THREADS; l > 1; l = (l + NUM_THREADS - 1) / NUM_THREADS) {
    reduce_sum_cal<false><<<(l + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(output->d_buf, dmem, l);
    tmp = dmem;
    dmem = output->d_buf;
    output->d_buf = tmp;
  }

  float *sum_p;
  CUDA_CALL(cudaMalloc(&sum_p, sizeof(float)));
  CUDA_CALL(cudaMemcpyAsync(sum_p, tmp, sizeof(float), cudaMemcpyDeviceToDevice, stream[s_idx][d]));

  #if DEBUG
  float sum2 = 0.0;
  CUDA_CALL(cudaStreamSynchronize(stream[s_idx][d]));
  CUDA_CALL(cudaMemcpy(&sum2, sum_p, sizeof(float), cudaMemcpyDeviceToHost));
  #endif

  softmax_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, output->d_buf, sum_p, N_);
}

/*
 * bmm
 * @brief : Perform a batch matrix-matrix product of matrices stored in input and weight.
 *          However, bmm performs matrix-vector product in this project.
 *          
 * @param [in1] input  : a vector of size [K_]
 * @param [in2] weight : a matrix of size [K_ x N_]
 * @param [out] output : a vector of size [N_]
 */
void bmm(Tensor *input, Tensor *weight, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int K_ = weight->shape[0];
  int N_ = weight->shape[1];
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    float c = 0.0;
    for (int k=0; k<K_; ++k) {
      c += input->buf[k] * weight->buf[k * N_ + n];
    }
    output->buf[n] = c;
  }
  #endif
  bmm_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, weight->d_buf, output->d_buf, K_, N_);
}

/*
 * relu
 * @brief : Apply the rectified linear unit function element-wise. relu(x) = max(0,x)
 *          
 * @param [in1] input  : a vector of size [N_]
 * @param [out] output : a vector of size [N_]
 */
void relu(Tensor *input, Tensor *output, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  int N_ = input->num_elem();
  
  #if DEBUG
  for (int n=0; n<N_; ++n) {
    float x = input->buf[n];
    if (x < 0.0) output->buf[n] = 0.0;
    else output->buf[n] = x;
  }
  #endif
  relu_cal<<<(N_ + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream[s_idx][d]>>>(input->d_buf, output->d_buf, N_);
}

/*
 * log_top_one
 * @brief : Return the largest element of the log of given input tensor.
 *          
 * @param  [in1] input  : a vector of size [N_]
 * @return [ret] topi   : an index of the largest element
 */
int log_top_one(Tensor *input, int d, int s_idx) {
  CUDA_CALL(cudaSetDevice(d));
  input->to_host(d, s_idx);
  int N_ = input->num_elem();
  int topi = 0;
  float topval = logf(input->buf[0]);
  
  for (int n=1; n<N_; ++n) {
    float x = logf(input->buf[n]);
    if (x >= topval) {
      topi = n;
      topval = x;
    }
  }
  return topi;
}

/*
 * initialize_translator
 * @brief : initialize translator. load the parameter binary file and store parameters into Tensors
 *          
 * @param [in1] parameter_fname  : the name of the binary file where parameters are stored
 */
void initialize_translator(const char *parameter_fname, int N){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    size_t parameter_binary_size = 0;
    float *parameter = (float *) read_binary(parameter_fname, &parameter_binary_size);

    eW_emb_raw = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET0);
    eW_ir_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET1);
    eW_iz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET2);
    eW_in_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET3);
    eW_hr_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET4);
    eW_hz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET5);
    eW_hn_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET6);
    eb_ir_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET7);
    eb_iz_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET8);
    eb_in_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET9);
    eb_hr_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET10);
    eb_hz_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET11);
    eb_hn_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET12);
    dW_emb_raw = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET13);
    dW_ir_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET14);
    dW_iz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET15);
    dW_in_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET16);
    dW_hr_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET17);
    dW_hz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET18);
    dW_hn_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET19);
    db_ir_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET20);
    db_iz_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET21);
    db_in_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET22);
    db_hr_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET23);
    db_hz_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET24);
    db_hn_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET25);
    dW_attn_raw = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE}, parameter + OFFSET26);
    db_attn_raw = new Tensor({MAX_LENGTH}, parameter + OFFSET27);
    dW_attn_comb_raw = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE}, parameter + OFFSET28);
    db_attn_comb_raw = new Tensor({HIDDEN_SIZE}, parameter + OFFSET29);
    dW_out_raw = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET30);
    db_out_raw = new Tensor({OUTPUT_VOCAB_SIZE}, parameter + OFFSET31);
  } else {
      eW_emb_raw = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE});
      eW_ir_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eW_iz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eW_in_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eW_hr_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eW_hz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eW_hn_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      eb_ir_raw = new Tensor({HIDDEN_SIZE});
      eb_iz_raw = new Tensor({HIDDEN_SIZE});
      eb_in_raw = new Tensor({HIDDEN_SIZE});
      eb_hr_raw = new Tensor({HIDDEN_SIZE});
      eb_hz_raw = new Tensor({HIDDEN_SIZE});
      eb_hn_raw = new Tensor({HIDDEN_SIZE});
      dW_emb_raw = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
      dW_ir_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      dW_iz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      dW_in_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      dW_hr_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      dW_hz_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      dW_hn_raw = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
      db_ir_raw = new Tensor({HIDDEN_SIZE});
      db_iz_raw = new Tensor({HIDDEN_SIZE});
      db_in_raw = new Tensor({HIDDEN_SIZE});
      db_hr_raw = new Tensor({HIDDEN_SIZE});
      db_hz_raw = new Tensor({HIDDEN_SIZE});
      db_hn_raw = new Tensor({HIDDEN_SIZE});
      dW_attn_raw = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE});
      db_attn_raw = new Tensor({MAX_LENGTH});
      dW_attn_comb_raw = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE});
      db_attn_comb_raw = new Tensor({HIDDEN_SIZE});
      dW_out_raw = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
      db_out_raw = new Tensor({OUTPUT_VOCAB_SIZE});
  }

  for (int d = 0; d < NUM_GPUS; ++d) {
    CUDA_CALL(cudaSetDevice(d));
    eW_emb[d] = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE});
    eW_ir[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_iz[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_in[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hr[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hz[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eW_hn[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    eb_ir[d] = new Tensor({HIDDEN_SIZE});
    eb_iz[d] = new Tensor({HIDDEN_SIZE});
    eb_in[d] = new Tensor({HIDDEN_SIZE});
    eb_hr[d] = new Tensor({HIDDEN_SIZE});
    eb_hz[d] = new Tensor({HIDDEN_SIZE});
    eb_hn[d] = new Tensor({HIDDEN_SIZE});
    dW_emb[d] = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
    dW_ir[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_iz[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_in[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hr[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hz[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    dW_hn[d] = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE});
    db_ir[d] = new Tensor({HIDDEN_SIZE});
    db_iz[d] = new Tensor({HIDDEN_SIZE});
    db_in[d] = new Tensor({HIDDEN_SIZE});
    db_hr[d] = new Tensor({HIDDEN_SIZE});
    db_hz[d] = new Tensor({HIDDEN_SIZE});
    db_hn[d] = new Tensor({HIDDEN_SIZE});
    dW_attn[d] = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE});
    db_attn[d] = new Tensor({MAX_LENGTH});
    dW_attn_comb[d] = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE});
    db_attn_comb[d] = new Tensor({HIDDEN_SIZE});
    dW_out[d] = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE});
    db_out[d] = new Tensor({OUTPUT_VOCAB_SIZE});
    for (int s_idx = 0; s_idx < NUM_STREAMS; ++s_idx) {
      CUDA_CALL(cudaStreamCreate(&stream[s_idx][d]));

      encoder_hidden[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_outputs[s_idx][d] = new Tensor({MAX_LENGTH, HIDDEN_SIZE});
      encoder_embedded[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_rtmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_rtmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_rt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_ztmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_ztmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_zt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_ntmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_ntmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_ntmp5[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_nt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_htmp1[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_htmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      encoder_htmp3[s_idx][d] = new Tensor({HIDDEN_SIZE});

      encoder_ht[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_input[s_idx][d] = new Tensor({MAX_LENGTH});
      decoder_output[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoded_words[s_idx][d] = new Tensor({MAX_LENGTH});
      decoder_embedded[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_embhid[s_idx][d] = new Tensor({2 * HIDDEN_SIZE});
      decoder_attn[s_idx][d] = new Tensor({MAX_LENGTH});
      decoder_attn_weights[s_idx][d] = new Tensor ({MAX_LENGTH});
      decoder_attn_applied[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_embattn[s_idx][d] = new Tensor({2 * HIDDEN_SIZE});
      decoder_attn_comb[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_relu[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_rtmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_rtmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_rt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ztmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ztmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_zt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ntmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ntmp4[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ntmp5[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_nt[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_htmp1[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_htmp2[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_htmp3[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_ht[s_idx][d] = new Tensor({HIDDEN_SIZE});
      decoder_out[s_idx][d] = new Tensor({OUTPUT_VOCAB_SIZE});
      decoder_logsoftmax[s_idx][d] = new Tensor({OUTPUT_VOCAB_SIZE});
    }
  }
}

/*
 * finalize_translator
 * @brief : free all dynamically allocated variables
 */
void finalize_translator(){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stderr, "\n");
  }
  delete eW_emb_raw;
  delete eW_ir_raw;
  delete eW_iz_raw;
  delete eW_in_raw;
  delete eW_hr_raw;
  delete eW_hz_raw;
  delete eW_hn_raw;
  delete eb_ir_raw;
  delete eb_iz_raw;
  delete eb_in_raw;
  delete eb_hr_raw;
  delete eb_hz_raw;
  delete eb_hn_raw;
  delete dW_emb_raw;
  delete dW_ir_raw;
  delete dW_iz_raw;
  delete dW_in_raw;
  delete dW_hr_raw;
  delete dW_hz_raw;
  delete dW_hn_raw;
  delete db_ir_raw;
  delete db_iz_raw;
  delete db_in_raw;
  delete db_hr_raw;
  delete db_hz_raw;
  delete db_hn_raw;
  delete dW_attn_raw;
  delete db_attn_raw;
  delete dW_attn_comb_raw;
  delete db_attn_comb_raw;
  delete dW_out_raw;
  delete db_out_raw;

  for (int d = 0; d < NUM_GPUS; ++d) {
    CUDA_CALL(cudaSetDevice(d));

    // free parameters
    // delete eW_emb[d];
    // delete eW_ir[d]; 
    // delete eW_iz[d]; 
    // delete eW_in[d]; 
    // delete eW_hr[d]; 
    // delete eW_hz[d]; 
    // delete eW_hn[d]; 
    // delete eb_ir[d]; 
    // delete eb_iz[d]; 
    // delete eb_in[d]; 
    // delete eb_hr[d]; 
    // delete eb_hz[d]; 
    // delete eb_hn[d]; 
    // delete dW_emb[d];
    // delete dW_ir[d]; 
    // delete dW_iz[d]; 
    // delete dW_in[d]; 
    // delete dW_hr[d]; 
    // delete dW_hz[d]; 
    // delete dW_hn[d]; 
    // delete db_ir[d]; 
    // delete db_iz[d]; 
    // delete db_in[d]; 
    // delete db_hr[d]; 
    // delete db_hz[d]; 
    // delete db_hn[d]; 
    // delete dW_attn[d];
    // delete db_attn[d];
    // delete dW_attn_comb[d];
    // delete db_attn_comb[d];
    // delete dW_out[d];
    // delete db_out[d];


    for (int s_idx = 0; s_idx < NUM_STREAMS; ++s_idx) {
      CUDA_CALL(cudaStreamDestroy(stream[s_idx][d]));

      delete encoder_hidden[s_idx][d];
      delete encoder_outputs[s_idx][d];
      delete encoder_embedded[s_idx][d];
      delete encoder_rtmp2[s_idx][d];
      delete encoder_rtmp4[s_idx][d];
      delete encoder_rt[s_idx][d];
      delete encoder_ztmp2[s_idx][d]; 
      delete encoder_ztmp4[s_idx][d]; 
      delete encoder_zt[s_idx][d]; 
      delete encoder_ntmp2[s_idx][d];
      delete encoder_ntmp4[s_idx][d];
      delete encoder_ntmp5[s_idx][d];
      delete encoder_nt[s_idx][d];
      delete encoder_htmp1[s_idx][d];
      delete encoder_htmp2[s_idx][d];
      delete encoder_htmp3[s_idx][d];
      delete encoder_ht[s_idx][d];

      delete decoder_input[s_idx][d];
      delete decoder_output[s_idx][d];
      delete decoded_words[s_idx][d];
      delete decoder_embedded[s_idx][d];
      delete decoder_embhid[s_idx][d];
      delete decoder_attn[s_idx][d];
      delete decoder_attn_weights[s_idx][d];
      delete decoder_attn_applied[s_idx][d];
      delete decoder_embattn[s_idx][d];
      delete decoder_attn_comb[s_idx][d];
      delete decoder_relu[s_idx][d];
      delete decoder_rtmp2[s_idx][d];
      delete decoder_rtmp4[s_idx][d];
      delete decoder_rt[s_idx][d];
      delete decoder_ztmp2[s_idx][d];
      delete decoder_ztmp4[s_idx][d];
      delete decoder_zt[s_idx][d];
      delete decoder_ntmp2[s_idx][d];
      delete decoder_ntmp4[s_idx][d];
      delete decoder_ntmp5[s_idx][d];
      delete decoder_nt[s_idx][d];
      delete decoder_htmp1[s_idx][d];
      delete decoder_htmp2[s_idx][d];
      delete decoder_htmp3[s_idx][d];
      delete decoder_ht[s_idx][d];
      delete decoder_out[s_idx][d];
      delete decoder_logsoftmax[s_idx][d];
    }
  }
}

__global__ void linear_cal(const float *input, const float *weight, const float *bias, float *output, int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    float sum = 0.0;
    for (int i = 0; i < K; i++) {
      sum += weight[m * K + i] * input[i];
    }
    output[m] = sum + bias[m];
  }
}

__global__ void bmm_cal(const float *input, const float *weight, float *output, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += weight[k * N + n] * input[k];
    }
    output[n] = sum;
  }
}

__global__ void elemwise_add_cal(const float *input1, const float *input2, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = input1[m] + input2[m];
  }
}

__global__ void elemwise_add_sigmoid_cal(const float *input1, const float *input2, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = 1.0 / (1.0 + exp(- (input1[m] + input2[m])));
  }
}

__global__ void elemwise_add_tanh_cal(const float *input1, const float *input2, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = tanhf(input1[m] + input2[m]);
  }
}

__global__ void elemwise_mult_cal(const float *input1, const float *input2, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = input1[m] * input2[m];
  }
}

__global__ void elemwise_oneminus_cal(const float *input, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = 1.0 - input[m];
  }
}

__global__ void softmax_cal(const float *input, float *output, float *sum, int M) {
  float threadSum = *sum;
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = expf(input[m]) / threadSum;
  }
}

__global__ void relu_cal(const float *input, float *output, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    output[m] = fmaxf(input[m], 0);
  }
}

// add_sigmoid(linear, linear)
__global__ void linear2_add_sigmoid_cal(const float *input1, const float *input2, const float *weight1, const float *weight2, const float *bias1, const float *bias2, float *output, int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    float sum = 0.0;
    for (int i = 0; i < K; i++) {
      sum += weight1[m * K + i] * input1[i];
    }
    for (int i = 0; i < K; i++) {
      sum += weight2[m * K + i] * input2[i];
    }
    output[m] = 1.0 / (1.0 + exp(- (sum + bias1[m] + bias2[m])));
  }
}

// (linear, linear) * mul
__global__ void n_t_part_cal(const float *input1, const float *input2, const float *weight1, const float *weight2, const float *bias1, const float *bias2, const float *input_mul, float *output, int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    float sum1 = 0.0;
    float sum2 = 0.0;
    for (int i = 0; i < K; i++) {
      sum1 += weight1[m * K + i] * input1[i];
    }
    for (int i = 0; i < K; i++) {
      sum2 += weight2[m * K + i] * input2[i];
    }
    sum1 += bias1[m];
    sum2 += bias2[m];
    sum2 *= input_mul[m];
    output[m] = tanhf(sum1 + sum2);
  }
}

__global__ void h_t_part_cal(const float *input_zt, const float *input_nt, float *hidden, int M) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < M) {
    float zt = input_zt[m];
    hidden[m] = (zt * hidden[m]) + (input_nt[m] * (1.0 - zt)) ;
  }
}

void load_parameters(int d) {
  eW_emb[d]->copy_from(eW_emb_raw);
  eW_ir[d]->copy_from(eW_ir_raw);
  eW_iz[d]->copy_from(eW_iz_raw);
  eW_in[d]->copy_from(eW_in_raw);
  eW_hr[d]->copy_from(eW_hr_raw);
  eW_hz[d]->copy_from(eW_hz_raw);
  eW_hn[d]->copy_from(eW_hn_raw);
  eb_ir[d]->copy_from(eb_ir_raw);
  eb_iz[d]->copy_from(eb_iz_raw);
  eb_in[d]->copy_from(eb_in_raw);
  eb_hr[d]->copy_from(eb_hr_raw);
  eb_hz[d]->copy_from(eb_hz_raw);
  eb_hn[d]->copy_from(eb_hn_raw);
  dW_emb[d]->copy_from(dW_emb_raw);
  dW_ir[d]->copy_from(dW_ir_raw);
  dW_iz[d]->copy_from(dW_iz_raw);
  dW_in[d]->copy_from(dW_in_raw);
  dW_hr[d]->copy_from(dW_hr_raw);
  dW_hz[d]->copy_from(dW_hz_raw);
  dW_hn[d]->copy_from(dW_hn_raw);
  db_ir[d]->copy_from(db_ir_raw);
  db_iz[d]->copy_from(db_iz_raw);
  db_in[d]->copy_from(db_in_raw);
  db_hr[d]->copy_from(db_hr_raw);
  db_hz[d]->copy_from(db_hz_raw);
  db_hn[d]->copy_from(db_hn_raw);
  dW_attn[d]->copy_from(dW_attn_raw);
  db_attn[d]->copy_from(db_attn_raw);
  dW_attn_comb[d]->copy_from(dW_attn_comb_raw);
  db_attn_comb[d]->copy_from(db_attn_comb_raw);
  dW_out[d]->copy_from(dW_out_raw);
  db_out[d]->copy_from(db_out_raw);

  CUDA_CALL(cudaSetDevice(d));
  eW_emb[d]->to_device(d);
  eW_ir[d]->to_device(d);
  eW_iz[d]->to_device(d);
  eW_in[d]->to_device(d);
  eW_hr[d]->to_device(d);
  eW_hz[d]->to_device(d);
  eW_hn[d]->to_device(d);
  eb_ir[d]->to_device(d);
  eb_iz[d]->to_device(d);
  eb_in[d]->to_device(d);
  eb_hr[d]->to_device(d);
  eb_hz[d]->to_device(d);
  eb_hn[d]->to_device(d);
  dW_emb[d]->to_device(d);
  dW_ir[d]->to_device(d);
  dW_iz[d]->to_device(d);
  dW_in[d]->to_device(d);
  dW_hr[d]->to_device(d);
  dW_hz[d]->to_device(d);
  dW_hn[d]->to_device(d);
  db_ir[d]->to_device(d);
  db_iz[d]->to_device(d);
  db_in[d]->to_device(d);
  db_hr[d]->to_device(d);
  db_hz[d]->to_device(d);
  db_hn[d]->to_device(d);
  dW_attn[d]->to_device(d);
  db_attn[d]->to_device(d);
  dW_attn_comb[d]->to_device(d);
  db_attn_comb[d]->to_device(d);
  dW_out[d]->to_device(d);
  db_out[d]->to_device(d);
}