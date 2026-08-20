#include "memvanta/common.hpp"
#include "memvanta/lru_cache.hpp"
#include "memvanta/quant.hpp"
#include "memvanta/llama_model.hpp"
#include "memvanta/worker_pool.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
int main(){
  assert(memvanta::parse_size("1M")==1048576);
  {
    std::vector<std::byte> src(64); memvanta::TensorCache c(128);
    auto a=c.get_or_load(1,src.data(),64); auto b=c.get_or_load(1,src.data(),64);
    auto st=c.stats(); assert(st.hits==1 && st.misses==1); (void)a;(void)b;
  }
  {
    constexpr std::size_t n=256; std::vector<float>a(n),x(n);
    std::mt19937 g(7); std::uniform_real_distribution<float>d(-1,1);
    float ref=0; for(std::size_t i=0;i<n;++i){a[i]=d(g);x[i]=d(g);ref+=a[i]*x[i];}
    auto q8=memvanta::quantize_q8_0(a.data(),n); auto q4=memvanta::quantize_q4_0(a.data(),n);
    float e8=std::fabs(memvanta::dot_q8_0(q8.data(),x.data(),n)-ref)/(std::fabs(ref)+1e-6f);
    float e4=std::fabs(memvanta::dot_q4_0(q4.data(),x.data(),n)-ref)/(std::fabs(ref)+1e-6f);
    assert(e8 < 0.08f); assert(e4 < 0.35f);
  }

  {
    memvanta::WorkerPool pool(4); std::vector<int> seen(100,0);
    pool.parallel_for(seen.size(),[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)seen[i]=1;});
    for(int v:seen) assert(v==1);
  }
  {
    constexpr std::size_t d=64; std::vector<float> k(d),v(d),q(d),out(d),refv(d); std::mt19937 g(11); std::uniform_real_distribution<float> dist(-1,1);
    for(std::size_t i=0;i<d;++i){k[i]=dist(g);v[i]=dist(g);q[i]=dist(g);refv[i]=0.25f*v[i];}
    float refdot=0;for(std::size_t i=0;i<d;++i)refdot+=k[i]*q[i];
    for(auto typ:{memvanta::KVCacheType::F32,memvanta::KVCacheType::F16,memvanta::KVCacheType::Q8}){
      memvanta::PagedKVCache c(d,16,typ);c.write(0,k.data(),v.data());std::fill(out.begin(),out.end(),0);float got=c.dot_key(0,0,q.data(),d);c.add_value(0,0,0.25f,out.data(),d);
      float de=std::fabs(got-refdot)/(std::fabs(refdot)+1e-5f),ve=0,base=0;for(std::size_t i=0;i<d;++i){ve+=std::fabs(out[i]-refv[i]);base+=std::fabs(refv[i]);}ve/=base+1e-5f;
      if(typ==memvanta::KVCacheType::F32){assert(de<1e-5f);assert(ve<1e-5f);}else if(typ==memvanta::KVCacheType::F16){assert(de<0.01f);assert(ve<0.01f);}else{assert(de<0.08f);assert(ve<0.03f);}
    }
  }
  std::cout<<"ok\n";
}
