#pragma once
#include "memvanta/gguf.hpp"
#include "memvanta/worker_pool.hpp"
#include <cstddef>

namespace memvanta {
void tensor_read_row_f32(const GgufFile&,const GgufTensor&,std::size_t,float*,std::size_t);
void tensor_matvec(const GgufFile&,const GgufTensor&,const float*,float*,unsigned threads=1,WorkerPool* pool=nullptr);
void tensor_matmul_batch(const GgufFile&,const GgufTensor&,const float* x,float* y,std::size_t batch,unsigned threads=1,WorkerPool* pool=nullptr);
// v0.6 register-blocked Q4/Q8 GEMM path. Activations are quantized once per batch
// into 32-element Q8 blocks and reused across all output rows.
void tensor_matmul_batch_v06(const GgufFile&,const GgufTensor&,const float* x,float* y,std::size_t batch,unsigned threads=1,WorkerPool* pool=nullptr);
void tensor_vector_to_f32(const GgufFile&,const GgufTensor&,float*,std::size_t);
float dot_f32_simd(const float* a,const float* b,std::size_t n);
void axpy_f32_simd(float alpha,const float* x,float* y,std::size_t n);
std::uint16_t fp32_to_fp16(float x);
} // namespace memvanta
