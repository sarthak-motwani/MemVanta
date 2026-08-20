#include "memvanta/runtime.hpp"
#include <iostream>
int main(int argc,char**argv){ if(argc<2){std::cerr<<"usage: memvanta_bench FILE\n";return 1;} memvanta::TensorStore store(argv[1],64ull<<20); for(auto cache:{128ull<<20,256ull<<20,512ull<<20}){ memvanta::Runtime rt(store,{cache,1,2,true}); auto s=rt.run_stream(); std::cout<<"cache="<<(cache>>20)<<"MiB throughput="<<s.gib_per_s<<"GiB/s peakRSS="<<s.peak_rss_kb/1024.0<<"MiB hits="<<s.cache.hits<<" misses="<<s.cache.misses<<"\n";} }
