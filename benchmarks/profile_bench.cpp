#include "memvanta/llama_model.hpp"
#include "memvanta/gguf_kernels.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc,char**argv){
    try{
        std::string model,csv="memvanta-layer-profile.csv";unsigned threads=4;std::size_t ctx=128,prompt_n=32,gen_n=8,batch=16;
        for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
            if(a=="--model")model=val();else if(a=="--threads")threads=std::stoul(val());else if(a=="--ctx")ctx=std::stoull(val());else if(a=="--prompt")prompt_n=std::stoull(val());else if(a=="--gen")gen_n=std::stoull(val());else if(a=="--batch")batch=std::stoull(val());else if(a=="--csv")csv=val();else throw std::runtime_error("unknown arg: "+a);
        }
        if(model.empty())throw std::runtime_error("--model required");
        memvanta::LlamaModel m(model,threads,ctx,128,memvanta::KVCacheType::F16);
        std::string seed="Once upon a time there was a small village with a curious child who loved stories. ";
        std::vector<int> ids;while(ids.size()<prompt_n){auto z=m.tokenizer().encode(seed,ids.empty());ids.insert(ids.end(),z.begin(),z.end());}ids.resize(std::min(prompt_n,m.config().n_ctx>gen_n?m.config().n_ctx-gen_n:prompt_n));
        memvanta::Sampler greedy({0.0f,1,42});
        memvanta::reset_kernel_profile();memvanta::enable_kernel_profiling(true);m.reset();
        auto t0=Clock::now();m.prefill(ids,batch,true);auto t1=Clock::now();
        int cur=ids.empty()?0:ids.back();double sampling_ms=0.0;
        for(std::size_t i=0;i<gen_n;++i){const auto&logits=m.forward(cur,true);auto s0=Clock::now();cur=greedy.sample(logits);auto s1=Clock::now();sampling_ms+=std::chrono::duration<double,std::milli>(s1-s0).count();}
        auto t2=Clock::now();memvanta::enable_kernel_profiling(false);
        auto entries=memvanta::kernel_profile_snapshot();std::ofstream f(csv);f<<"layer,kind,tensor,batch,calls,ms\n";
        std::map<std::pair<int,memvanta::KernelProfileKind>,double> agg;double kernel_ms=0;
        for(const auto&e:entries){f<<e.layer<<','<<memvanta::kernel_profile_kind_name(e.kind)<<','<<e.tensor<<','<<e.batch<<','<<e.calls<<','<<std::fixed<<std::setprecision(6)<<e.ms<<'\n';agg[{e.layer,e.kind}]+=e.ms;kernel_ms+=e.ms;}
        const double prefill_ms=std::chrono::duration<double,std::milli>(t1-t0).count();const double decode_total_ms=std::chrono::duration<double,std::milli>(t2-t1).count();const double model_total_ms=std::chrono::duration<double,std::milli>(t2-t0).count();
        double qkv=0,attn_proj=0,ffn=0,output=0;for(auto&[k,v]:agg){switch(k.second){case memvanta::KernelProfileKind::QProj:case memvanta::KernelProfileKind::KProj:case memvanta::KernelProfileKind::VProj:qkv+=v;break;case memvanta::KernelProfileKind::OProj:attn_proj+=v;break;case memvanta::KernelProfileKind::FfnGate:case memvanta::KernelProfileKind::FfnUp:case memvanta::KernelProfileKind::FfnDown:ffn+=v;break;case memvanta::KernelProfileKind::Output:output+=v;break;default:break;}}
        const double residual=std::max(0.0,model_total_ms-kernel_ms-sampling_ms);
        std::cout<<"# MemVanta trained-model layer profile\n";
        std::cout<<"prefill_ms="<<prefill_ms<<" decode_total_ms="<<decode_total_ms<<" sampling_ms="<<sampling_ms<<"\n";
        std::cout<<"qkv_projection_ms="<<qkv<<" attention_output_projection_ms="<<attn_proj<<" ffn_gemm_ms="<<ffn<<" output_head_ms="<<output<<"\n";
        std::cout<<"non_gemm_core_ms="<<residual<<" (attention softmax/KV + RMSNorm + RoPE + elementwise/residual)\n";
        std::cout<<"layer,qkv_ms,o_proj_ms,ffn_ms\n";
        for(std::size_t li=0;li<m.config().n_layer;++li){double lq=0,lo=0,lf=0;for(auto&[k,v]:agg)if(k.first==static_cast<int>(li)){switch(k.second){case memvanta::KernelProfileKind::QProj:case memvanta::KernelProfileKind::KProj:case memvanta::KernelProfileKind::VProj:lq+=v;break;case memvanta::KernelProfileKind::OProj:lo+=v;break;case memvanta::KernelProfileKind::FfnGate:case memvanta::KernelProfileKind::FfnUp:case memvanta::KernelProfileKind::FfnDown:lf+=v;break;default:break;}}std::cout<<li<<','<<lq<<','<<lo<<','<<lf<<'\n';}
        std::cout<<"profile_csv="<<csv<<"\n";return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
}
