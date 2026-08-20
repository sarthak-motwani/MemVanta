#pragma once
#include "memvanta/tensor_store.hpp"
#include "memvanta/lru_cache.hpp"
#include "memvanta/prefetcher.hpp"
#include <cstdint>
namespace memvanta {
struct RunConfig { std::uint64_t cache_bytes; std::uint32_t passes=1; std::uint32_t prefetch_depth=2; bool copy_cache=true; };
struct RunStats { double seconds=0, gib_per_s=0; std::uint64_t checksum=0; CacheStats cache; std::uint64_t peak_rss_kb=0; };
class Runtime {
public:
  Runtime(const TensorStore& store, RunConfig cfg);
  RunStats run_stream();
private:
  static std::uint64_t rss_kb();
  const TensorStore& store_; RunConfig cfg_; TensorCache cache_; Prefetcher prefetcher_;
};
}
