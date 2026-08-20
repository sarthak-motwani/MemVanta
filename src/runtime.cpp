#include "memvanta/runtime.hpp"
#include "memvanta/common.hpp"
#include <algorithm>
#include <fstream>
#include <string>
namespace memvanta {
Runtime::Runtime(const TensorStore&s,RunConfig c):store_(s),cfg_(c),cache_(c.cache_bytes),prefetcher_(s,cache_){}
std::uint64_t Runtime::rss_kb(){ std::ifstream f("/proc/self/status"); std::string k; while(f>>k){ if(k=="VmHWM:"){ std::uint64_t v; std::string u; f>>v>>u; return v;} std::string rest; std::getline(f,rest);} return 0; }
RunStats Runtime::run_stream(){ auto start=std::chrono::steady_clock::now(); std::uint64_t checksum=1469598103934665603ull,total=0; for(std::uint32_t p=0;p<cfg_.passes;++p){ for(std::uint32_t i=0;i<store_.count();++i){ for(std::uint32_t d=1;d<=cfg_.prefetch_depth;++d) if(i+d<store_.count()){ if(cfg_.copy_cache) prefetcher_.request(i+d); else store_.prefetch(i+d); } auto&s=store_.slice(i); const std::byte* ptr=nullptr; if(cfg_.copy_cache){ auto buf=cache_.get_or_load(i,store_.ptr(i),s.bytes); store_.release(i); ptr=buf->data(); } else { store_.prefetch(i); ptr=store_.ptr(i); } const auto* u=reinterpret_cast<const unsigned char*>(ptr); std::uint64_t stride=4096; for(std::uint64_t j=0;j<s.bytes;j+=stride){ checksum^=u[j]; checksum*=1099511628211ull; } if(!cfg_.copy_cache) store_.release(i); total+=s.bytes; } }
 auto end=std::chrono::steady_clock::now(); double sec=std::chrono::duration<double>(end-start).count(); prefetcher_.stop(); return {sec,gib(total)/sec,checksum,cache_.stats(),rss_kb()}; }
}
