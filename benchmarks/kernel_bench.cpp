#include "memvanta/matmul.hpp"
#include "memvanta/quant.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Stat { double mean=0, sd=0, min=0, max=0; };
static Stat stats(const std::vector<double>& v) {
    Stat s; s.mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();
    double q=0; for(double x:v) q+=(x-s.mean)*(x-s.mean); s.sd=std::sqrt(q/v.size());
    s.min=*std::min_element(v.begin(),v.end()); s.max=*std::max_element(v.begin(),v.end()); return s;
}
int main(int argc,char**argv){
    std::size_t rows=4096, cols=4096, batch=32; unsigned threads=4, reps=7;
    for(int i=1;i<argc;++i){ std::string a=argv[i]; auto val=[&](){return std::string(argv[++i]);};
      if(a=="--rows") rows=std::stoull(val()); else if(a=="--cols") cols=std::stoull(val());
      else if(a=="--batch") batch=std::stoull(val()); else if(a=="--threads") threads=std::stoul(val());
      else if(a=="--reps") reps=std::stoul(val()); }
    if(cols%32) { std::cerr<<"cols must be multiple of 32\n"; return 2; }
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1,1);
    std::vector<float> A(rows*cols), x(cols), y(std::max(rows,batch*rows)), xb(batch*cols);
    for(auto&v:A)v=dist(rng); for(auto&v:x)v=dist(rng); for(auto&v:xb)v=dist(rng);
    auto q8=memvanta::quantize_q8_0(A.data(),A.size());
    auto q4=memvanta::quantize_q4_0(A.data(),A.size());
    auto xq8=memvanta::quantize_q8_0(x.data(),x.size());
    volatile float sink=0;
    auto run=[&](const char*name, std::size_t bytes, unsigned used_threads, std::size_t logical_batch, auto fn){
      fn();
      std::vector<double> ts; ts.reserve(reps);
      for(unsigned r=0;r<reps;++r){ auto t0=Clock::now(); fn(); auto t1=Clock::now(); sink += y[r%rows]; ts.push_back(std::chrono::duration<double>(t1-t0).count()); }
      auto st=stats(ts); const double ops=2.0*rows*cols*logical_batch; const double gflops=ops/st.mean/1e9; const double gib=(double)bytes/st.mean/(1024.0*1024.0*1024.0);
      std::cout<<name<<","<<rows<<","<<cols<<","<<logical_batch<<","<<used_threads<<","<<std::fixed<<std::setprecision(6)<<st.mean<<","<<st.sd<<","<<std::setprecision(3)<<gflops<<","<<gib<<"\n";
    };
    std::cout<<"kernel,rows,cols,batch,threads,mean_s,sd_s,gflop_s,weight_gib_s\n";
    run("f32_gemv", A.size()*sizeof(float), threads, 1, [&]{memvanta::matvec_f32(A.data(),x.data(),y.data(),rows,cols,threads);});
    run("q8_0_fp32_gemv", q8.size()*sizeof(memvanta::BlockQ8_0), threads, 1, [&]{memvanta::matvec_q8_0(q8.data(),x.data(),y.data(),rows,cols,threads);});
    run("q4_0_fp32_gemv_t1", q4.size()*sizeof(memvanta::BlockQ4_0), 1, 1, [&]{memvanta::matvec_q4_0(q4.data(),x.data(),y.data(),rows,cols,1);});
    run("q4_0_fp32_gemv_tN", q4.size()*sizeof(memvanta::BlockQ4_0), threads, 1, [&]{memvanta::matvec_q4_0(q4.data(),x.data(),y.data(),rows,cols,threads);});
    const unsigned q4q8_threads=memvanta::effective_kernel_threads(rows,cols,threads);
    run("q4_0_q8_0_gemv", q4.size()*sizeof(memvanta::BlockQ4_0), q4q8_threads, 1, [&]{memvanta::matvec_q4_q8(q4.data(),xq8.data(),y.data(),rows,cols,threads);});
    const unsigned batch_threads=memvanta::effective_kernel_threads(rows,cols*std::max<std::size_t>(batch,1),threads);
    run("q4_0_fp32_batched_gemm", q4.size()*sizeof(memvanta::BlockQ4_0), batch_threads, batch, [&]{memvanta::matmul_q4_0_batch(q4.data(),xb.data(),y.data(),rows,cols,batch,threads);});
    if(sink==1234567.0f) std::cerr<<sink;
}
