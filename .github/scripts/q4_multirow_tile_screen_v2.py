#!/usr/bin/env python3
import json, os, random, re, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src/gguf_kernels.cpp'
MODEL = ROOT / os.environ.get('MODEL', 'stories15M-q4_0.gguf')
RESULT = ROOT / 'results/q4-multirow-tile-screen'
REPS = 5


def run(cmd, **kw):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def build(name):
    b = ROOT / f'build-{name}'
    run(['cmake','-S','.','-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake','--build',str(b),'--target','memvanta_profile','memvanta_tests','memvanta_tokenizer_tests','-j',str(os.cpu_count() or 2)])
    run(['ctest','--test-dir',str(b),'--output-on-failure'])
    return b


def patch(text):
    marker = 'inline void q4_row_batch8_fp32(const GgufBlockQ4_0*w,const float*x,std::size_t stride,std::size_t nb,float out[8]){'
    helper = r'''inline void q4_rows2_batch4_fp32(const GgufBlockQ4_0*w0,const GgufBlockQ4_0*w1,const float*x,std::size_t stride,std::size_t nb,float out[2][4]){
#if defined(__AVX2__)
    __m256 a00=_mm256_setzero_ps(),a01=_mm256_setzero_ps(),a02=_mm256_setzero_ps(),a03=_mm256_setzero_ps();
    __m256 a10=_mm256_setzero_ps(),a11=_mm256_setzero_ps(),a12=_mm256_setzero_ps(),a13=_mm256_setzero_ps();
    const __m128i mask=_mm_set1_epi8(0x0f),bias=_mm_set1_epi8(8);
    for(std::size_t bi=0;bi<nb;++bi){
        const float d0=fp16_to_fp32(w0[bi].d),d1=fp16_to_fp32(w1[bi].d);
        const __m128i p0=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w0[bi].qs));
        const __m128i p1=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w1[bi].qs));
        const __m128i l0=_mm_sub_epi8(_mm_and_si128(p0,mask),bias),h0=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(p0,4),mask),bias);
        const __m128i l1=_mm_sub_epi8(_mm_and_si128(p1,mask),bias),h1=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(p1,4),mask),bias);
        auto half=[&](const __m128i s0,const __m128i s1,std::size_t base){
            const __m256 x00=_mm256_loadu_ps(x+base),x01=_mm256_loadu_ps(x+stride+base),x02=_mm256_loadu_ps(x+2*stride+base),x03=_mm256_loadu_ps(x+3*stride+base);
            const __m256 q00=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s0)),_mm256_set1_ps(d0));
            const __m256 q10=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s1)),_mm256_set1_ps(d1));
            a00=_mm256_fmadd_ps(q00,x00,a00);a01=_mm256_fmadd_ps(q00,x01,a01);a02=_mm256_fmadd_ps(q00,x02,a02);a03=_mm256_fmadd_ps(q00,x03,a03);
            a10=_mm256_fmadd_ps(q10,x00,a10);a11=_mm256_fmadd_ps(q10,x01,a11);a12=_mm256_fmadd_ps(q10,x02,a12);a13=_mm256_fmadd_ps(q10,x03,a13);
            const std::size_t b8=base+8;
            const __m256 x10=_mm256_loadu_ps(x+b8),x11=_mm256_loadu_ps(x+stride+b8),x12=_mm256_loadu_ps(x+2*stride+b8),x13=_mm256_loadu_ps(x+3*stride+b8);
            const __m256 q01=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s0,8))),_mm256_set1_ps(d0));
            const __m256 q11=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s1,8))),_mm256_set1_ps(d1));
            a00=_mm256_fmadd_ps(q01,x10,a00);a01=_mm256_fmadd_ps(q01,x11,a01);a02=_mm256_fmadd_ps(q01,x12,a02);a03=_mm256_fmadd_ps(q01,x13,a03);
            a10=_mm256_fmadd_ps(q11,x10,a10);a11=_mm256_fmadd_ps(q11,x11,a11);a12=_mm256_fmadd_ps(q11,x12,a12);a13=_mm256_fmadd_ps(q11,x13,a13);
        };
        half(l0,l1,bi*32);half(h0,h1,bi*32+16);
    }
    alignas(32) float t[8];auto hs=[&](const __m256&v){_mm256_store_ps(t,v);return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];};
    out[0][0]=hs(a00);out[0][1]=hs(a01);out[0][2]=hs(a02);out[0][3]=hs(a03);out[1][0]=hs(a10);out[1][1]=hs(a11);out[1][2]=hs(a12);out[1][3]=hs(a13);
#else
    q4_row_batch4_fp32(w0,x,stride,nb,out[0]);q4_row_batch4_fp32(w1,x,stride,nb,out[1]);
#endif
}

'''
    if marker not in text:
        raise RuntimeError('helper insertion marker not found')
    text = text.replace(marker, helper + marker, 1)
    old = '''        const bool ffn=is_ffn_tensor(t.name);parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r){std::size_t b=0;if(ffn){for(;b+8<=batch;b+=8){float o[8];if(t.type==GgmlType::Q4_0)q4_row_batch8_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch8_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<8;++j)y[(b+j)*rows+r]=o[j];}}else{for(;b+4<=batch;b+=4){float o[4];if(t.type==GgmlType::Q4_0)q4_row_batch4_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch4_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<4;++j)y[(b+j)*rows+r]=o[j];}}for(;b<batch;++b){if(t.type==GgmlType::Q4_0)y[b*rows+r]=dot_q4_0_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols);else y[b*rows+r]=dot_q8_0_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols);}}});'''
    new = r'''        const bool ffn=is_ffn_tensor(t.name);const char* mr=std::getenv("MEMVANTA_MULTIROW_MODE");const bool use_mr=t.type==GgmlType::Q4_0&&mr&&*mr&&((std::strcmp(mr,"all")==0)||(std::strcmp(mr,"ffn")==0&&ffn));
        if(use_mr){auto*A=reinterpret_cast<const GgufBlockQ4_0*>(p);const std::size_t pairs=(rows+1)/2;parallel_rows(pairs,threads,pool,[&](std::size_t p0,std::size_t p1){for(std::size_t pr=p0;pr<p1;++pr){const std::size_t r=pr*2;if(r+1<rows){std::size_t b=0;for(;b+4<=batch;b+=4){float o[2][4];q4_rows2_batch4_fp32(A+r*nb,A+(r+1)*nb,x+b*cols,cols,nb,o);for(int j=0;j<4;++j){y[(b+j)*rows+r]=o[0][j];y[(b+j)*rows+r+1]=o[1][j];}}for(;b<batch;++b){y[b*rows+r]=dot_q4_0_fp32(A+r*nb,x+b*cols,cols);y[b*rows+r+1]=dot_q4_0_fp32(A+(r+1)*nb,x+b*cols,cols);}}else{std::size_t b=0;if(ffn){for(;b+8<=batch;b+=8){float o[8];q4_row_batch8_fp32(A+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<8;++j)y[(b+j)*rows+r]=o[j];}}else{for(;b+4<=batch;b+=4){float o[4];q4_row_batch4_fp32(A+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<4;++j)y[(b+j)*rows+r]=o[j];}}for(;b<batch;++b)y[b*rows+r]=dot_q4_0_fp32(A+r*nb,x+b*cols,cols);}}});
        }else{''' + old + r'''}'''
    if old not in text:
        raise RuntimeError('v06 replacement target not found')
    return text.replace(old, new, 1)


def bench(build0, build1):
    RESULT.mkdir(parents=True, exist_ok=True)
    variants=['baseline','rows2_ffn','rows2_all']; data={v:[] for v in variants}; order=[]; rng=random.Random(20260822)
    for _ in range(REPS):
        block=variants[:]; rng.shuffle(block); order += block
    mr=re.compile(r'\b(prefill_ms|decode_total_ms)=([0-9.]+)'); rr=re.compile(r'Maximum resident set size \(kbytes\):\s*(\d+)')
    for v in order:
        env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'close','OMP_PLACES':'cores','MALLOC_ARENA_MAX':'2'}); exe=build0/'memvanta_profile'
        if v!='baseline':
            exe=build1/'memvanta_profile'; env['MEMVANTA_MULTIROW_MODE']='ffn' if v=='rows2_ffn' else 'all'
        p=subprocess.run(['/usr/bin/time','-v',str(exe),'--model',str(MODEL),'--threads','4','--ctx','128','--prompt','64','--gen','16','--batch','32'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        if p.returncode:
            raise RuntimeError(f'{v} rc={p.returncode}\n{p.stderr}')
        m=dict((k,float(x)) for k,x in mr.findall(p.stdout)); q=rr.search(p.stderr)
        if 'prefill_ms' not in m or 'decode_total_ms' not in m or q is None:
            raise RuntimeError(f'missing metrics for {v}')
        data[v].append((m['prefill_ms'],m['decode_total_ms'],float(q.group(1))))
    def agg(v):
        x=data[v]; pm=[z[0] for z in x]; dm=[z[1] for z in x]; rm=[z[2] for z in x]
        return {'prefill_ms':statistics.mean(pm),'decode_ms':statistics.mean(dm),'rss_kib':statistics.mean(rm),'prefill_cv':100*statistics.stdev(pm)/statistics.mean(pm),'decode_cv':100*statistics.stdev(dm)/statistics.mean(dm)}
    rows={v:agg(v) for v in variants}; base=rows['baseline']
    def score(r): return 2/(1/(64000/r['prefill_ms'])+1/(16000/r['decode_ms']))
    bs=score(base); bd=16000/base['decode_ms']; passed=[]
    for v,r in rows.items():
        r['prefill_tps']=64000/r['prefill_ms']; r['decode_tps']=16000/r['decode_ms']; r['gain']=(score(r)/bs-1)*100; r['rss_delta']=(r['rss_kib']/base['rss_kib']-1)*100; r['decode_delta']=(r['decode_tps']/bd-1)*100
        r['pass']=v!='baseline' and r['gain']>=5 and r['rss_delta']<=2 and base['prefill_cv']<=5 and base['decode_cv']<=5 and r['prefill_cv']<=5 and r['decode_cv']<=5 and r['decode_delta']>=-2
        if r['pass']: passed.append((score(r),v))
    winner=max(passed)[1] if passed else None; ranked=sorted(variants,key=lambda v:score(rows[v]),reverse=True)
    (RESULT/'summary.json').write_text(json.dumps({'winner':winner,'rows':rows},indent=2)+'\n')
    lines=['# Q4 multi-row tile screen','',f'Baseline plus 2 direct 2-row x 4-token Q4 candidates; {REPS} randomized reps; no activation packing; production source restored.','', '| Rank | Variant | Prefill tok/s | Decode tok/s | Gain | RSS delta | Prefill CV | Decode CV | Pass |','|---:|---|---:|---:|---:|---:|---:|---:|:---:|']
    for i,v in enumerate(ranked,1):
        r=rows[v]; lines.append(f"| {i} | {v} | {r['prefill_tps']:.2f} | {r['decode_tps']:.2f} | {r['gain']:.2f}% | {r['rss_delta']:.2f}% | {r['prefill_cv']:.2f}% | {r['decode_cv']:.2f}% | {'YES' if r['pass'] else 'no'} |")
    lines += ['',f'Winner: **{winner or "none"}**.','Gate: >=5% combined gain, <=2% RSS increase, <=5% timing CV baseline/candidate, <=2% decode regression.']
    (RESULT/'SUMMARY.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


def main():
    original=SRC.read_text(); b0=build('mr-baseline')
    try:
        SRC.write_text(patch(original)); b1=build('mr-candidate')
    finally:
        SRC.write_text(original)
    run(['git','diff','--exit-code','--','src/gguf_kernels.cpp']); bench(b0,b1)

if __name__=='__main__':
    main()
