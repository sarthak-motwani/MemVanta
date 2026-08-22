#!/usr/bin/env python3
import json, os, random, re, statistics, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src/gguf_kernels.cpp'
MODEL=ROOT/os.environ.get('MODEL','stories15M-q4_0.gguf')
OUT=ROOT/'results/q4-activation-pack-screen'
REPS=int(os.environ.get('REPS','5'))

def run(cmd,**kw):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True,**kw)

def build(name):
    b=ROOT/f'build-{name}'
    run(['cmake','-S','.','-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake','--build',str(b),'--target','memvanta_profile','memvanta_tests','memvanta_tokenizer_tests','-j',str(os.cpu_count() or 2)])
    run(['ctest','--test-dir',str(b),'--output-on-failure'])
    return b

def patch(text):
    anchor='inline void q8_row_batch4_fp32(const GgufBlockQ8_0*w,const float*x,std::size_t stride,std::size_t nb,float out[4]){'
    helper=r'''inline void q4_row_batch4_packed_fp32(const GgufBlockQ4_0*w,const float*x,std::size_t nb,float out[4]){
#if defined(__AVX2__)
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();const __m128i mask=_mm_set1_epi8(0x0f),bias=_mm_set1_epi8(8);
    for(std::size_t bi=0;bi<nb;++bi){const float d=fp16_to_fp32(w[bi].d);const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w[bi].qs));const __m128i lo=_mm_sub_epi8(_mm_and_si128(packed,mask),bias),hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed,4),mask),bias);for(int h=0;h<2;++h){const __m128i src=h?hi:lo;for(int k=0;k<16;k+=8){__m256 q=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(src,k))),_mm256_set1_ps(d));const std::size_t off=bi*128+h*16+k;a0=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off),a0);a1=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off+32),a1);a2=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off+64),a2);a3=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off+96),a3);}}}
    alignas(32)float t[8];auto hs=[&](const __m256&v){_mm256_store_ps(t,v);return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];};out[0]=hs(a0);out[1]=hs(a1);out[2]=hs(a2);out[3]=hs(a3);
#else
    for(int j=0;j<4;++j){float s=0;for(std::size_t bi=0;bi<nb;++bi){const float d=fp16_to_fp32(w[bi].d);for(int i=0;i<16;++i){s+=d*(int(w[bi].qs[i]&15)-8)*x[bi*128+j*32+i];s+=d*(int(w[bi].qs[i]>>4)-8)*x[bi*128+j*32+16+i];}}out[j]=s;}
#endif
}

'''
    if anchor not in text: raise RuntimeError('helper anchor not found')
    text=text.replace(anchor,helper+anchor,1)
    marker='    } else {\n        const bool ffn=is_ffn_tensor(t.name);'
    start=text.find(marker)
    if start<0: raise RuntimeError('hybrid branch start not found')
    tail='\n    }\n    profile_add(t,batch,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());'
    end=text.find(tail,start)
    if end<0: raise RuntimeError('hybrid branch end not found')
    original=text[start+len('    } else {\n'):end]
    replacement=r'''    } else {
        const bool ffn=is_ffn_tensor(t.name);
        const char* pe=std::getenv("MEMVANTA_Q4_PACK");
        const bool pack_q4=t.type==GgmlType::Q4_0&&pe&&*pe&&((std::strcmp(pe,"all")==0)||(std::strcmp(pe,"ffn")==0&&ffn));
        if(pack_q4){
            const std::size_t groups=(batch+3)/4;
            std::vector<float> packed(groups*nb*128,0.0f);
            for(std::size_t g=0;g<groups;++g){for(std::size_t bi=0;bi<nb;++bi){for(std::size_t j=0;j<4;++j){const std::size_t b=g*4+j;if(b<batch)std::memcpy(packed.data()+g*nb*128+bi*128+j*32,x+b*cols+bi*32,32*sizeof(float));}}}
            parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r){for(std::size_t g=0;g<groups;++g){float o[4];q4_row_batch4_packed_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,packed.data()+g*nb*128,nb,o);for(std::size_t j=0;j<4;++j){const std::size_t b=g*4+j;if(b<batch)y[b*rows+r]=o[j];}}}});
        } else {
'''+original+r'''
        }'''
    return text[:start]+replacement+text[end:]

def bench(base,cand):
    OUT.mkdir(parents=True,exist_ok=True)
    variants=['baseline','pack_ffn','pack_all']
    rng=random.Random(20260822); order=[]
    for _ in range(REPS):
        z=variants[:];rng.shuffle(z);order+=z
    data={v:[] for v in variants}
    mr=re.compile(r'\b(prefill_ms|decode_total_ms)=([0-9.]+)'); rr=re.compile(r'Maximum resident set size \(kbytes\):\s*(\d+)')
    for i,v in enumerate(order,1):
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'close','OMP_PLACES':'cores','MALLOC_ARENA_MAX':'2'})
        exe=base/'memvanta_profile'
        if v!='baseline': exe=cand/'memvanta_profile';env['MEMVANTA_Q4_PACK']='ffn' if v=='pack_ffn' else 'all'
        p=subprocess.run(['/usr/bin/time','-v',str(exe),'--model',str(MODEL),'--threads','4','--ctx','128','--prompt','64','--gen','16','--batch','32'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        print(f'[{i}/{len(order)}] {v} rc={p.returncode}',flush=True)
        if p.returncode: raise RuntimeError(f'{v} rc={p.returncode}\n{p.stderr}')
        m=dict((k,float(x)) for k,x in mr.findall(p.stdout));q=rr.search(p.stderr)
        if not q or 'prefill_ms' not in m or 'decode_total_ms' not in m: raise RuntimeError(f'missing metrics {v}')
        data[v].append((m['prefill_ms'],m['decode_total_ms'],float(q.group(1))))
    def agg(v):
        x=data[v];pm=[a[0] for a in x];dm=[a[1] for a in x];rm=[a[2] for a in x]
        return {'prefill_ms':statistics.mean(pm),'decode_ms':statistics.mean(dm),'rss_kib':statistics.mean(rm),'prefill_cv':100*statistics.stdev(pm)/statistics.mean(pm),'decode_cv':100*statistics.stdev(dm)/statistics.mean(dm)}
    rows={v:agg(v) for v in variants};b=rows['baseline']
    def score(r):return 2/(1/(64000/r['prefill_ms'])+1/(16000/r['decode_ms']))
    bs=score(b);bd=16000/b['decode_ms'];passing=[]
    for v,r in rows.items():
        r['prefill_tps']=64000/r['prefill_ms'];r['decode_tps']=16000/r['decode_ms'];r['gain']=(score(r)/bs-1)*100;r['rss_delta']=(r['rss_kib']/b['rss_kib']-1)*100;r['decode_delta']=(r['decode_tps']/bd-1)*100
        r['pass']=v!='baseline' and r['gain']>=5 and r['rss_delta']<=2 and r['prefill_cv']<=5 and r['decode_cv']<=5 and b['prefill_cv']<=5 and b['decode_cv']<=5 and r['decode_delta']>=-2
        if r['pass']:passing.append((score(r),v))
    winner=max(passing)[1] if passing else None
    (OUT/'summary.json').write_text(json.dumps({'winner':winner,'rows':rows},indent=2)+'\n')
    ranked=sorted(variants,key=lambda v:score(rows[v]),reverse=True)
    lines=['# Q4 activation packing screen','',f'Baseline plus 2 activation-layout candidates; {REPS} randomized reps each; production source restored.','', '| Rank | Variant | Prefill tok/s | Decode tok/s | Gain | RSS delta | Prefill CV | Decode CV | Pass |','|---:|---|---:|---:|---:|---:|---:|---:|:---:|']
    for i,v in enumerate(ranked,1):
        r=rows[v];lines.append(f"| {i} | {v} | {r['prefill_tps']:.2f} | {r['decode_tps']:.2f} | {r['gain']:.2f}% | {r['rss_delta']:.2f}% | {r['prefill_cv']:.2f}% | {r['decode_cv']:.2f}% | {'YES' if r['pass'] else 'no'} |")
    lines+=['',f'Winner: **{winner or "none"}**.','Gate: >=5% combined gain, <=2% RSS increase, <=5% timing CV on baseline/candidate, <=2% decode regression.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def main():
    original=SRC.read_text()
    base=build('q4pack-baseline')
    try:
        SRC.write_text(patch(original));cand=build('q4pack-candidate')
    finally:
        SRC.write_text(original)
    run(['git','diff','--exit-code','--','src/gguf_kernels.cpp'])
    bench(base,cand)

if __name__=='__main__':main()
