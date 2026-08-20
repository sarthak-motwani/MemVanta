#include "memvanta/gguf_kernels.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(MEMVANTA_USE_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace memvanta {
namespace {
#pragma pack(push,1)
struct GgufBlockQ4_0 { std::uint16_t d; std::uint8_t qs[16]; };
struct GgufBlockQ8_0 { std::uint16_t d; std::int8_t qs[32]; };
#pragma pack(pop)
static_assert(sizeof(GgufBlockQ4_0)==18);
static_assert(sizeof(GgufBlockQ8_0)==34);

using Clock = std::chrono::steady_clock;
std::atomic<bool> g_profile_enabled{false};
std::mutex g_profile_mu;
std::vector<KernelProfileEntry> g_profile_entries;

KernelProfileKind classify_tensor(const std::string& name) {
    if (name.find(".attn_q.") != std::string::npos) return KernelProfileKind::QProj;
    if (name.find(".attn_k.") != std::string::npos) return KernelProfileKind::KProj;
    if (name.find(".attn_v.") != std::string::npos) return KernelProfileKind::VProj;
    if (name.find(".attn_output.") != std::string::npos) return KernelProfileKind::OProj;
    if (name.find(".ffn_gate.") != std::string::npos) return KernelProfileKind::FfnGate;
    if (name.find(".ffn_up.") != std::string::npos) return KernelProfileKind::FfnUp;
    if (name.find(".ffn_down.") != std::string::npos) return KernelProfileKind::FfnDown;
    if (name == "output.weight") return KernelProfileKind::Output;
    return KernelProfileKind::Other;
}

int tensor_layer(const std::string& name) {
    if (name.rfind("blk.",0) != 0) return -1;
    auto p = name.find('.',4);
    if (p == std::string::npos) return -1;
    try { return std::stoi(name.substr(4,p-4)); } catch (...) { return -1; }
}

void profile_add(const GgufTensor& t, std::size_t batch, double ms) {
    if (!g_profile_enabled.load(std::memory_order_relaxed)) return;
    KernelProfileEntry e;
    e.layer = tensor_layer(t.name);
    e.kind = classify_tensor(t.name);
    e.tensor = t.name;
    e.calls = 1;
    e.batch = batch;
    e.ms = ms;
    std::lock_guard<std::mutex> lock(g_profile_mu);
    for (auto& x : g_profile_entries) {
        if (x.layer == e.layer && x.kind == e.kind && x.tensor == e.tensor && x.batch == e.batch) {
            x.calls += 1; x.ms += e.ms; return;
        }
    }
    g_profile_entries.push_back(std::move(e));
}

template<class Fn>
void parallel_rows(std::size_t rows, unsigned threads, WorkerPool* pool, Fn fn) {
    threads=std::max(1u,threads); threads=std::min<unsigned>(threads,static_cast<unsigned>(std::max<std::size_t>(rows,1)));
    if(pool && pool->size()==threads){ pool->parallel_for(rows,fn); return; }
    if(threads==1){fn(0,rows);return;}
#if defined(MEMVANTA_USE_OPENMP)
    const std::size_t step=(rows+threads-1)/threads;
    #pragma omp parallel num_threads(threads)
    {
        const unsigned tid=static_cast<unsigned>(omp_get_thread_num());
        const std::size_t a=tid*step,b=std::min(rows,a+step); if(a<b)fn(a,b);
    }
#else
    std::vector<std::thread> tmp; const std::size_t step=(rows+threads-1)/threads;
    for(unsigned tid=0;tid<threads;++tid){auto a=tid*step,b=std::min(rows,a+step);if(a<b)tmp.emplace_back(fn,a,b);} for(auto&th:tmp)th.join();
#endif
}

inline int dot_i8_16(const std::int8_t* a,const std::int8_t* b){
#if defined(__AVX2__)
    __m128i aa=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i bb=_mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m256i a16=_mm256_cvtepi8_epi16(aa),b16=_mm256_cvtepi8_epi16(bb);
    __m256i p32=_mm256_madd_epi16(a16,b16);
    __m128i lo=_mm256_castsi256_si128(p32),hi=_mm256_extracti128_si256(p32,1);
    __m128i v=_mm_add_epi32(lo,hi);v=_mm_hadd_epi32(v,v);v=_mm_hadd_epi32(v,v);return _mm_cvtsi128_si32(v);
#else
    int s=0;for(int i=0;i<16;++i)s+=int(a[i])*int(b[i]);return s;
#endif
}

struct Q8Vector {
    std::vector<std::int8_t> q;
    std::vector<float> d;
};

Q8Vector quantize_q8(const float* x, std::size_t cols) {
    if (cols % 32) throw std::runtime_error("Q8 activation quantization requires cols multiple of 32");
    Q8Vector a; a.q.resize(cols); a.d.resize(cols/32);
    for (std::size_t bi=0; bi<cols/32; ++bi) {
        const float* z=x+bi*32; float mx=0.0f;
#if defined(__AVX2__)
        __m256 m=_mm256_setzero_ps(); const __m256 sign=_mm256_set1_ps(-0.0f);
        for(int k=0;k<32;k+=8){auto v=_mm256_andnot_ps(sign,_mm256_loadu_ps(z+k));m=_mm256_max_ps(m,v);}alignas(32) float mm[8];_mm256_store_ps(mm,m);for(float v:mm)mx=std::max(mx,v);
#else
        for(int k=0;k<32;++k)mx=std::max(mx,std::abs(z[k]));
#endif
        const float d=mx>0?mx/127.0f:1.0f, inv=1.0f/d; a.d[bi]=d;
        for(int k=0;k<32;++k)a.q[bi*32+k]=static_cast<std::int8_t>(std::clamp(std::lround(z[k]*inv),-127l,127l));
    }
    return a;
}

struct Q8ActBatch {
    std::size_t batch{}, cols{}, blocks{};
    std::vector<std::int8_t> q;
    std::vector<float> d;
};

Q8ActBatch quantize_activations_q8(const float* x,std::size_t batch,std::size_t cols){
    if(cols%32) throw std::runtime_error("Q8 activation quantization requires cols multiple of 32");
    Q8ActBatch a; a.batch=batch;a.cols=cols;a.blocks=cols/32;a.q.resize(batch*cols);a.d.resize(batch*a.blocks);
    for(std::size_t b=0;b<batch;++b){auto v=quantize_q8(x+b*cols,cols);std::memcpy(a.q.data()+b*cols,v.q.data(),cols);std::memcpy(a.d.data()+b*a.blocks,v.d.data(),a.blocks*sizeof(float));}
    return a;
}

inline int dot_q4_q8_block(const GgufBlockQ4_0& w, const std::int8_t* aq) {
    alignas(16) std::int8_t lo[16], hi[16];
#if defined(__AVX2__)
    const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
    const __m128i mask=_mm_set1_epi8(0x0f), bias=_mm_set1_epi8(8);
    const __m128i qlo=_mm_sub_epi8(_mm_and_si128(packed,mask),bias);
    const __m128i qhi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed,4),mask),bias);
    _mm_store_si128(reinterpret_cast<__m128i*>(lo),qlo);
    _mm_store_si128(reinterpret_cast<__m128i*>(hi),qhi);
#else
    for(int i=0;i<16;++i){lo[i]=std::int8_t((w.qs[i]&15)-8);hi[i]=std::int8_t((w.qs[i]>>4)-8);}
#endif
    return dot_i8_16(lo,aq)+dot_i8_16(hi,aq+16);
}

inline float dot_q4_q8(const GgufBlockQ4_0* w,const std::int8_t* aq,const float* ad,std::size_t nb){
    float sum=0.0f;for(std::size_t bi=0;bi<nb;++bi)sum+=fp16_to_fp32(w[bi].d)*ad[bi]*float(dot_q4_q8_block(w[bi],aq+bi*32));return sum;
}
inline float dot_q8_q8(const GgufBlockQ8_0* w,const std::int8_t* aq,const float* ad,std::size_t nb){
    float sum=0.0f;for(std::size_t bi=0;bi<nb;++bi){int s=dot_i8_16(w[bi].qs,aq+bi*32)+dot_i8_16(w[bi].qs+16,aq+bi*32+16);sum+=fp16_to_fp32(w[bi].d)*ad[bi]*float(s);}return sum;
}

inline float dot_q4_0_gguf(const GgufBlockQ4_0* blocks,const float*x,std::size_t n){
    if(n%32) throw std::runtime_error("Q4_0 dot requires block-aligned input");
    auto a=quantize_q8(x,n); return dot_q4_q8(blocks,a.q.data(),a.d.data(),n/32);
}

inline float dot_q8_0_gguf(const GgufBlockQ8_0* blocks,const float*x,std::size_t n){
    const std::size_t nb=n/32;float sum=0;
    for(std::size_t b=0;b<nb;++b){float d=fp16_to_fp32(blocks[b].d);
#if defined(__AVX2__)
        __m256 acc=_mm256_setzero_ps();for(int k=0;k<32;k+=8){auto q8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(blocks[b].qs+k));auto qi=_mm256_cvtepi8_epi32(q8);acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(qi),_mm256_loadu_ps(x+b*32+k),acc);}alignas(32)float tmp[8];_mm256_store_ps(tmp,acc);float s=0;for(float v:tmp)s+=v;sum+=d*s;
#else
        float s=0;for(std::size_t k=0;k<32;++k)s+=blocks[b].qs[k]*x[b*32+k];sum+=d*s;
#endif
    }return sum;
}

inline void q4_row_batch8_q8(const GgufBlockQ4_0* w,const Q8ActBatch& a,std::size_t b0,std::size_t nb,float out[8]){
    std::fill(out,out+8,0.0f);
    for(std::size_t bi=0;bi<nb;++bi){const float wd=fp16_to_fp32(w[bi].d);for(int j=0;j<8;++j){const auto*aq=a.q.data()+(b0+j)*a.cols+bi*32;out[j]+=wd*a.d[(b0+j)*a.blocks+bi]*float(dot_q4_q8_block(w[bi],aq));}}
}
inline void q8_row_batch8_q8(const GgufBlockQ8_0* w,const Q8ActBatch& a,std::size_t b0,std::size_t nb,float out[8]){
    std::fill(out,out+8,0.0f);
    for(std::size_t bi=0;bi<nb;++bi){const float wd=fp16_to_fp32(w[bi].d);for(int j=0;j<8;++j){const auto*aq=a.q.data()+(b0+j)*a.cols+bi*32;int s=dot_i8_16(w[bi].qs,aq)+dot_i8_16(w[bi].qs+16,aq+16);out[j]+=wd*a.d[(b0+j)*a.blocks+bi]*float(s);}}
}
} // namespace

void enable_kernel_profiling(bool enabled){g_profile_enabled.store(enabled,std::memory_order_relaxed);}
void reset_kernel_profile(){std::lock_guard<std::mutex> lock(g_profile_mu);g_profile_entries.clear();}
std::vector<KernelProfileEntry> kernel_profile_snapshot(){std::lock_guard<std::mutex> lock(g_profile_mu);return g_profile_entries;}
const char* kernel_profile_kind_name(KernelProfileKind k){switch(k){case KernelProfileKind::QProj:return "q_proj";case KernelProfileKind::KProj:return "k_proj";case KernelProfileKind::VProj:return "v_proj";case KernelProfileKind::OProj:return "o_proj";case KernelProfileKind::FfnGate:return "ffn_gate";case KernelProfileKind::FfnUp:return "ffn_up";case KernelProfileKind::FfnDown:return "ffn_down";case KernelProfileKind::Output:return "output";case KernelProfileKind::Other:return "other";}return "other";}

void tensor_vector_to_f32(const GgufFile& file,const GgufTensor&t,float*out,std::size_t n){
    if(t.elements()!=n)throw std::runtime_error("tensor vector size mismatch: "+t.name);const std::byte* p=file.tensor_data(t);
    if(t.type==GgmlType::F32){std::memcpy(out,p,n*sizeof(float));return;}if(t.type==GgmlType::F16){auto*q=reinterpret_cast<const std::uint16_t*>(p);for(std::size_t i=0;i<n;++i)out[i]=fp16_to_fp32(q[i]);return;}throw std::runtime_error("unsupported vector tensor type "+ggml_type_name(t.type)+": "+t.name);
}

void tensor_read_row_f32(const GgufFile& file,const GgufTensor&t,std::size_t row,float*out,std::size_t cols){
    if(t.dims.size()<2 || t.ne(0)!=cols || row>=t.ne(1))throw std::runtime_error("tensor row shape mismatch: "+t.name);const std::byte*p=file.tensor_data(t);
    if(t.type==GgmlType::F32){std::memcpy(out,reinterpret_cast<const float*>(p)+row*cols,cols*sizeof(float));return;}
    if(t.type==GgmlType::F16){auto*q=reinterpret_cast<const std::uint16_t*>(p)+row*cols;for(std::size_t i=0;i<cols;++i)out[i]=fp16_to_fp32(q[i]);return;}
    if(t.type==GgmlType::Q4_0){if(cols%32)throw std::runtime_error("Q4_0 row is not block-aligned");auto*b=reinterpret_cast<const GgufBlockQ4_0*>(p)+row*(cols/32);for(std::size_t bi=0;bi<cols/32;++bi){float d=fp16_to_fp32(b[bi].d);for(std::size_t i=0;i<16;++i){out[bi*32+i]=d*(static_cast<int>(b[bi].qs[i]&15)-8);out[bi*32+16+i]=d*(static_cast<int>(b[bi].qs[i]>>4)-8);}}return;}
    if(t.type==GgmlType::Q8_0){if(cols%32)throw std::runtime_error("Q8_0 row is not block-aligned");auto*b=reinterpret_cast<const GgufBlockQ8_0*>(p)+row*(cols/32);for(std::size_t bi=0;bi<cols/32;++bi){float d=fp16_to_fp32(b[bi].d);for(std::size_t i=0;i<32;++i)out[bi*32+i]=d*b[bi].qs[i];}return;}throw std::runtime_error("unsupported embedding row type "+ggml_type_name(t.type)+": "+t.name);
}

float dot_f32_simd(const float* a,const float* b,std::size_t n){
#if defined(__AVX2__)
    __m256 acc=_mm256_setzero_ps();std::size_t i=0;for(;i+8<=n;i+=8)acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);alignas(32) float t[8];_mm256_store_ps(t,acc);float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];for(;i<n;++i)s+=a[i]*b[i];return s;
#else
    float s=0;for(std::size_t i=0;i<n;++i)s+=a[i]*b[i];return s;
#endif
}
void axpy_f32_simd(float alpha,const float*x,float*y,std::size_t n){
#if defined(__AVX2__)
    __m256 av=_mm256_set1_ps(alpha);std::size_t i=0;for(;i+8<=n;i+=8){auto xv=_mm256_loadu_ps(x+i),yv=_mm256_loadu_ps(y+i);_mm256_storeu_ps(y+i,_mm256_fmadd_ps(av,xv,yv));}for(;i<n;++i)y[i]+=alpha*x[i];
#else
    for(std::size_t i=0;i<n;++i)y[i]+=alpha*x[i];
#endif
}
std::uint16_t fp32_to_fp16(float f){std::uint32_t x;std::memcpy(&x,&f,4);std::uint32_t sign=(x>>16)&0x8000u,mant=x&0x7fffffu;int exp=int((x>>23)&0xff)-127+15;if(exp<=0){if(exp<-10)return static_cast<std::uint16_t>(sign);mant=(mant|0x800000u)>>(1-exp);return static_cast<std::uint16_t>(sign+((mant+0x1000u)>>13));}if(exp>=31)return static_cast<std::uint16_t>(sign|0x7c00u);return static_cast<std::uint16_t>(sign|(std::uint32_t(exp)<<10)|((mant+0x1000u)>>13));}

void tensor_matvec(const GgufFile& file,const GgufTensor&t,const float*x,float*y,unsigned threads,WorkerPool* pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("matvec requires rank-2 tensor: "+t.name);const std::size_t cols=static_cast<std::size_t>(t.ne(0)),rows=static_cast<std::size_t>(t.ne(1));const std::byte*p=file.tensor_data(t);
    if(t.type==GgmlType::Q4_0){if(cols%32)throw std::runtime_error("Q4_0 matrix row not block aligned: "+t.name);auto a=quantize_q8(x,cols);auto*A=reinterpret_cast<const GgufBlockQ4_0*>(p);auto bpr=cols/32;parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r)y[r]=dot_q4_q8(A+r*bpr,a.q.data(),a.d.data(),bpr);});}
    else if(t.type==GgmlType::Q8_0){if(cols%32)throw std::runtime_error("Q8_0 matrix row not block aligned: "+t.name);auto*A=reinterpret_cast<const GgufBlockQ8_0*>(p);auto bpr=cols/32;parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r)y[r]=dot_q8_0_gguf(A+r*bpr,x,cols);});}
    else if(t.type==GgmlType::F32){auto*A=reinterpret_cast<const float*>(p);parallel_rows(rows,threads,pool,[&](std::size_t a,std::size_t b){for(std::size_t r=a;r<b;++r)y[r]=dot_f32_simd(A+r*cols,x,cols);});}
    else if(t.type==GgmlType::F16){auto*A=reinterpret_cast<const std::uint16_t*>(p);parallel_rows(rows,threads,pool,[&](std::size_t a,std::size_t b){for(std::size_t r=a;r<b;++r){const auto*w=A+r*cols;double s=0;for(std::size_t c=0;c<cols;++c)s+=static_cast<double>(fp16_to_fp32(w[c]))*x[c];y[r]=static_cast<float>(s);}});}
    else throw std::runtime_error("unsupported matvec tensor type "+ggml_type_name(t.type)+": "+t.name);
    profile_add(t,1,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

void tensor_matmul_batch(const GgufFile& file,const GgufTensor&t,const float*x,float*y,std::size_t batch,unsigned threads,WorkerPool* pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("matmul batch requires rank-2 tensor: "+t.name);const std::size_t cols=static_cast<std::size_t>(t.ne(0)),rows=static_cast<std::size_t>(t.ne(1));const std::byte*p=file.tensor_data(t);const std::size_t jobs=batch*rows;
    auto body=[&](std::size_t a,std::size_t b){for(std::size_t j=a;j<b;++j){std::size_t bi=j/rows,r=j%rows;const float*xv=x+bi*cols;float v=0;if(t.type==GgmlType::Q4_0){auto A=reinterpret_cast<const GgufBlockQ4_0*>(p)+r*(cols/32);v=dot_q4_0_gguf(A,xv,cols);}else if(t.type==GgmlType::Q8_0){auto A=reinterpret_cast<const GgufBlockQ8_0*>(p)+r*(cols/32);v=dot_q8_0_gguf(A,xv,cols);}else if(t.type==GgmlType::F32){v=dot_f32_simd(reinterpret_cast<const float*>(p)+r*cols,xv,cols);}else if(t.type==GgmlType::F16){auto*A=reinterpret_cast<const std::uint16_t*>(p)+r*cols;double s=0;for(std::size_t c=0;c<cols;++c)s+=double(fp16_to_fp32(A[c]))*xv[c];v=float(s);}else throw std::runtime_error("unsupported batch matmul tensor type "+ggml_type_name(t.type)+": "+t.name);y[bi*rows+r]=v;}};parallel_rows(jobs,threads,pool,body);profile_add(t,batch,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

void tensor_matmul_batch_v06(const GgufFile& file,const GgufTensor&t,const float*x,float*y,std::size_t batch,unsigned threads,WorkerPool* pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("blocked GEMM requires rank-2 tensor: "+t.name);const std::size_t cols=static_cast<std::size_t>(t.ne(0)),rows=static_cast<std::size_t>(t.ne(1));
    if(batch<2 || (t.type!=GgmlType::Q4_0 && t.type!=GgmlType::Q8_0) || cols%32){tensor_matmul_batch(file,t,x,y,batch,threads,pool);return;}
    const std::byte*p=file.tensor_data(t);const std::size_t nb=cols/32;auto a=quantize_activations_q8(x,batch,cols);
    constexpr std::size_t ROW_TILE=32, BATCH_TILE=8;const std::size_t tiles=(rows+ROW_TILE-1)/ROW_TILE;
    parallel_rows(tiles,threads,pool,[&](std::size_t ta,std::size_t tz){for(std::size_t ti=ta;ti<tz;++ti){const std::size_t r0=ti*ROW_TILE,r1=std::min(rows,r0+ROW_TILE);for(std::size_t r=r0;r<r1;++r){std::size_t b=0;for(;b+BATCH_TILE<=batch;b+=BATCH_TILE){float o[8];if(t.type==GgmlType::Q4_0)q4_row_batch8_q8(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,a,b,nb,o);else q8_row_batch8_q8(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,a,b,nb,o);for(std::size_t j=0;j<BATCH_TILE;++j)y[(b+j)*rows+r]=o[j];}for(;b<batch;++b){if(t.type==GgmlType::Q4_0)y[b*rows+r]=dot_q4_q8(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,a.q.data()+b*cols,a.d.data()+b*nb,nb);else y[b*rows+r]=dot_q8_q8(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,a.q.data()+b*cols,a.d.data()+b*nb,nb);}}}});
    profile_add(t,batch,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

} // namespace memvanta
