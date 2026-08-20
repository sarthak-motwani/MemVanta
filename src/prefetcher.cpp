#include "memvanta/prefetcher.hpp"
namespace memvanta {
Prefetcher::Prefetcher(const TensorStore&s,TensorCache&c):store_(s),cache_(c),worker_(&Prefetcher::loop,this){}
Prefetcher::~Prefetcher(){ stop(); }
void Prefetcher::request(std::uint32_t id){ if(id>=store_.count() || cache_.contains(id)) return; {std::lock_guard lk(mu_); if(pending_.insert(id).second) q_.push_back(id);} cv_.notify_one(); }
void Prefetcher::stop(){ bool expected=false; if(stop_.compare_exchange_strong(expected,true)){ cv_.notify_all(); if(worker_.joinable()) worker_.join(); } }
void Prefetcher::loop(){ while(true){ std::uint32_t id; {std::unique_lock lk(mu_); cv_.wait(lk,[&]{return stop_||!q_.empty();}); if(stop_&&q_.empty()) return; id=q_.front();q_.pop_front();pending_.erase(id);} store_.prefetch(id); auto&s=store_.slice(id); cache_.insert_prefetched(id,store_.ptr(id),s.bytes); store_.release(id); } }
}
