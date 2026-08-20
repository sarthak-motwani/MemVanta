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
    std::size_t rows=4096, cols=4096; unsigned threads=5, reps=7;
    for(int i=1;i<argc;++i){ std::string a=argv[i]; auto val=[&](){return std::string(argv[++i]);};
      if(a=="--rows") rows=std::stoull(val()); else if(a=="--cols") cols=std::stoull(val());
      else if(a=="--threads") threads=std::stoul(val()); else if(a=="--reps") reps=std::stoul(val()); }
    if(cols%32) { std::cerr<<"cols must be multiple of 32\n"; return 2; }
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1,1);
    std::vector<float> A(rows*cols), x(cols), y(rows);
    for(auto&v:A)v=dist(rng); for(auto&v:x)v=dist(rng);
    auto q8=memvanta::quantize_q8_0(A.data(),A.size());
    auto q4=memvanta::quantize_q4_0(A.data(),A.size());
    volatile float sink=0;
    auto run=[&](const char*name, std::size_t bytes, auto fn){
      fn();
      std::vector<double> ts; ts.reserve(reps);
      for(unsigned r=0;r<reps;++r){ auto t0=Clock::now(); fn(); auto t1=Clock::now(); sink += y[r%rows]; ts.push_back(std::chrono::duration<double>(t1-t0).count()); }
      auto st=stats(ts); const double ops=2.0*rows*cols; const double gflops=ops/st.mean/1e9; const double gib=(double)bytes/st.mean/(1024.0*1024.0*1024.0);
      std::cout<<name<<","<<rows<<","<<cols<<","<<threads<<","<<std::fixed<<std::setprecision(6)<<st.mean<<","<<st.sd<<","<<std::setprecision(3)<<gflops<<","<<gib<<"\n";
    };
    std::cout<<"kernel,rows,cols,threads,mean_s,sd_s,gflop_s,weight_gib_s\n";
    run("f32", A.size()*sizeof(float), [&]{memvanta::matvec_f32(A.data(),x.data(),y.data(),rows,cols,threads);});
    run("q8_0", q8.size()*sizeof(memvanta::BlockQ8_0), [&]{memvanta::matvec_q8_0(q8.data(),x.data(),y.data(),rows,cols,threads);});
    run("q4_0", q4.size()*sizeof(memvanta::BlockQ4_0), [&]{memvanta::matvec_q4_0(q4.data(),x.data(),y.data(),rows,cols,threads);});
    if(sink==1234567.0f) std::cerr<<sink;
}
