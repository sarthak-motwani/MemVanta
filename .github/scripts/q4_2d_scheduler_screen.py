#!/usr/bin/env python3
import json, os, random, re, shutil, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src/gguf_kernels.cpp'
MODEL = ROOT / os.environ.get('MODEL', 'stories15M-q4_0.gguf')
RESULT = ROOT / 'results/q4-2d-scheduler-screen-v3'
REPS = int(os.environ.get('REPS', '3'))


def run(cmd, **kw):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def build(name):
    b = ROOT / f'build-{name}'
    run(['cmake','-S','.','-B',str(b),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake','--build',str(b),'--target','memvanta_profile','memvanta_tests','memvanta_tokenizer_tests','-j',str(os.cpu_count() or 2)])
    run(['ctest','--test-dir',str(b),'--output-on-failure'])
    return b


def patch_scheduler(text):
    needle = "        const bool ffn=is_ffn_tensor(t.name);parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r){std::size_t b=0;if(ffn){for(;b+8<=batch;b+=8){float o[8];if(t.type==GgmlType::Q4_0)q4_row_batch8_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch8_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<8;++j)y[(b+j)*rows+r]=o[j];}}else{for(;b+4<=batch;b+=4){float o[4];if(t.type==GgmlType::Q4_0)q4_row_batch4_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch4_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<4;++j)y[(b+j)*rows+r]=o[j];}}for(;b<batch;++b){if(t.type==GgmlType::Q4_0)y[b*rows+r]=dot_q4_0_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols);else y[b*rows+r]=dot_q8_0_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols);}}});"
    if needle not in text:
        raise RuntimeError('scheduler replacement target not found')
    repl = r'''        const char* ge=std::getenv("MEMVANTA_SCHED_GROUP");
        if(ge&&*ge){
            const std::size_t group=std::strtoul(ge,nullptr,10);
            const char* te=std::getenv("MEMVANTA_SCHED_ROW_TILE");
            const std::size_t tile=std::max<std::size_t>(1,te?std::strtoul(te,nullptr,10):1);
            const char* oe=std::getenv("MEMVANTA_SCHED_ORDER");
            const bool bm=oe&&std::strcmp(oe,"batch")==0;
            if((group!=4&&group!=8)||batch%group)throw std::runtime_error("invalid 2D scheduler group");
            const std::size_t rt=(rows+tile-1)/tile,bg=batch/group,jobs=rt*bg;
            parallel_rows(jobs,threads,pool,[&](std::size_t a,std::size_t z){for(std::size_t job=a;job<z;++job){const std::size_t ri=bm?job%rt:job/bg,bi=bm?job/rt:job%bg;const std::size_t r0=ri*tile,r1=std::min(rows,r0+tile),b0=bi*group;for(std::size_t r=r0;r<r1;++r){if(group==8){float o[8];if(t.type==GgmlType::Q4_0)q4_row_batch8_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b0*cols,cols,nb,o);else q8_row_batch8_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b0*cols,cols,nb,o);for(int k=0;k<8;++k)y[(b0+k)*rows+r]=o[k];}else{float o[4];if(t.type==GgmlType::Q4_0)q4_row_batch4_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b0*cols,cols,nb,o);else q8_row_batch4_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b0*cols,cols,nb,o);for(int k=0;k<4;++k)y[(b0+k)*rows+r]=o[k];}}}});
        }else{
''' + needle + r'''
        }'''
    return text.replace(needle, repl, 1)


def benchmark(build_baseline, build_sched):
    RESULT.mkdir(parents=True, exist_ok=True)
    variants = ['baseline'] + [f'g{g}_r{r}_{o}' for g in (4,8) for r in (1,2,4,8,16) for o in ('row','batch')]
    rng = random.Random(20260822)
    order=[]
    for _ in range(REPS):
        block=variants[:]; rng.shuffle(block); order += block
    data={v:[] for v in variants}
    mr=re.compile(r'\b(prefill_ms|decode_total_ms)=([0-9.]+)')
    rr=re.compile(r'Maximum resident set size \(kbytes\):\s*(\d+)')
    for i,v in enumerate(order,1):
        env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'close','OMP_PLACES':'cores','MALLOC_ARENA_MAX':'2'})
        exe=build_baseline/'memvanta_profile'
        if v!='baseline':
            exe=build_sched/'memvanta_profile'; a=v.split('_'); env['MEMVANTA_SCHED_GROUP']=a[0][1:]; env['MEMVANTA_SCHED_ROW_TILE']=a[1][1:]; env['MEMVANTA_SCHED_ORDER']=a[2]
        p=subprocess.run(['/usr/bin/time','-v',str(exe),'--model',str(MODEL),'--threads','4','--ctx','128','--prompt','64','--gen','16','--batch','32'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        print(f'[{i}/{len(order)}] {v} rc={p.returncode}', flush=True)
        if p.returncode:
            raise RuntimeError(f'{v} failed rc={p.returncode}\n{p.stderr}')
        m=dict((k,float(x)) for k,x in mr.findall(p.stdout)); q=rr.search(p.stderr)
        if 'prefill_ms' not in m or 'decode_total_ms' not in m or q is None:
            raise RuntimeError(f'missing metrics for {v}')
        data[v].append((m['prefill_ms'],m['decode_total_ms'],float(q.group(1))))
    def agg(v):
        x=data[v]; pm=[z[0] for z in x]; dm=[z[1] for z in x]; rm=[z[2] for z in x]
        return {'prefill_ms':statistics.mean(pm),'decode_ms':statistics.mean(dm),'rss_kib':statistics.mean(rm),'prefill_cv':100*statistics.stdev(pm)/statistics.mean(pm),'decode_cv':100*statistics.stdev(dm)/statistics.mean(dm)}
    rows={v:agg(v) for v in variants}; base=rows['baseline']
    def score(r): return 2/(1/(64000/r['prefill_ms'])+1/(16000/r['decode_ms']))
    bs=score(base); bdec=16000/base['decode_ms']; passed=[]
    for v,r in rows.items():
        r['prefill_tps']=64000/r['prefill_ms']; r['decode_tps']=16000/r['decode_ms']; r['gain']=(score(r)/bs-1)*100; r['rss_delta']=(r['rss_kib']/base['rss_kib']-1)*100; r['decode_delta']=(r['decode_tps']/bdec-1)*100
        r['pass']=v!='baseline' and r['gain']>=8 and r['rss_delta']<=2 and r['prefill_cv']<=5 and r['decode_cv']<=5 and base['prefill_cv']<=5 and base['decode_cv']<=5 and r['decode_delta']>=-2
        if r['pass']: passed.append((score(r),v))
    winner=max(passed)[1] if passed else None
    ranked=sorted(variants,key=lambda v:score(rows[v]),reverse=True)
    (RESULT/'summary.json').write_text(json.dumps({'winner':winner,'rows':rows},indent=2)+'\n')
    lines=['# Q4 2D scheduler screen v3','',f'20 schedules plus baseline; {REPS} randomized reps each; production source restored.','', '| Rank | Variant | Prefill tok/s | Decode tok/s | Gain | RSS delta | Prefill CV | Decode CV | Pass |','|---:|---|---:|---:|---:|---:|---:|---:|:---:|']
    for i,v in enumerate(ranked,1):
        r=rows[v]; lines.append(f"| {i} | {v} | {r['prefill_tps']:.2f} | {r['decode_tps']:.2f} | {r['gain']:.2f}% | {r['rss_delta']:.2f}% | {r['prefill_cv']:.2f}% | {r['decode_cv']:.2f}% | {'YES' if r['pass'] else 'no'} |")
    lines += ['',f'Winner: **{winner or "none"}**.','Gate: >=8% combined gain, <=2% RSS increase, <=5% timing CV on baseline/candidate, <=2% decode regression.']
    (RESULT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    original=SRC.read_text()
    backup=ROOT/'.q4_scheduler_backup.cpp'
    backup.write_text(original)
    try:
        build_baseline=build('sched-baseline')
        SRC.write_text(patch_scheduler(original))
        build_sched=build('sched-candidate')
    finally:
        SRC.write_text(original)
        if backup.exists(): backup.unlink()
    run(['git','diff','--exit-code','--','src/gguf_kernels.cpp'])
    benchmark(build_baseline, build_sched)

if __name__ == '__main__':
    main()
