#!/usr/bin/env python3
import csv, json, pathlib, re, statistics, sys
out = pathlib.Path(sys.argv[1])

def rss_kib(path):
    if not path.exists(): return None
    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', path.read_text(errors='replace'))
    return int(m.group(1)) if m else None

def parse_environment(path):
    env={}
    if not path.exists(): return env
    for line in path.read_text(errors='replace').splitlines():
        for part in line.split():
            if '=' in part:
                k,v=part.split('=',1)
                env[k]=v
    return env

def memvanta_csv(path):
    rows=list(csv.DictReader(path.open()))
    def vals(candidates):
        for c in candidates:
            if rows and c in rows[0]:
                return [float(r[c]) for r in rows]
        return []
    pp=vals(['pp_tps','prompt_tps','pp_tokens_per_sec'])
    tg=vals(['tg_tps','gen_tps','generation_tps','tg_tokens_per_sec'])
    return pp,tg,rows

def llama_json(path):
    if not path.exists(): return []
    data=json.loads(path.read_text())
    if isinstance(data, dict): data=[data]
    return data

def describe(values, prefix, summary):
    if not values: return
    summary[f'{prefix}_mean']=statistics.mean(values)
    summary[f'{prefix}_sd']=statistics.stdev(values) if len(values)>1 else 0
    summary[f'{prefix}_median']=statistics.median(values)
    summary[f'{prefix}_min']=min(values)
    summary[f'{prefix}_max']=max(values)

cfg=parse_environment(out/'environment.txt')
prompt=int(cfg.get('prompt',0) or 0)
gen=int(cfg.get('gen',0) or 0)
threads=int(cfg.get('threads',0) or 0)
reps=int(cfg.get('reps',0) or 0)
warmup=int(cfg.get('warmup',0) or 0)
ctx=int(cfg.get('ctx',0) or 0)
batch=int(cfg.get('batch',0) or 0)
kv=cfg.get('kv','unknown')

summary={
  'benchmark':'TinyStories stories15M Q4_0 native-context A/B',
  'model':'stories15M-q4_0.gguf',
  'prompt_tokens':prompt,
  'generation_tokens':gen,
  'context':ctx,
  'threads':threads,
  'repetitions':reps,
  'warmup':warmup,
  'batch':batch,
  'memvanta_kv':kv,
  'disclosure':f'same GGUF, CPU-only, same threads; pp{prompt}/tg{gen}, context {ctx}, MemVanta KV={kv}'
}

if (out/'memvanta.csv').exists():
    pp,tg,rows=memvanta_csv(out/'memvanta.csv')
    describe(pp,'memvanta_pp_tps',summary)
    describe(tg,'memvanta_tg_tps',summary)
summary['memvanta_peak_rss_kib']=rss_kib(out/'memvanta.time.txt')
summary['llama_peak_rss_kib']=rss_kib(out/'llama-bench.time.txt')

lj=llama_json(out/'llama-bench.json')
if lj:
    prompt_vals=[]; gen_vals=[]
    for r in lj:
        n_prompt=int(r.get('n_prompt',0) or 0); n_gen=int(r.get('n_gen',0) or 0)
        v=r.get('avg_ts') or r.get('tokens_per_second') or r.get('t/s')
        if v is None: continue
        if n_prompt and not n_gen: prompt_vals.append(float(v))
        if n_gen and not n_prompt: gen_vals.append(float(v))
    describe(prompt_vals,'llama_pp_tps',summary)
    describe(gen_vals,'llama_tg_tps',summary)

for k in ('pp','tg'):
    a=summary.get(f'memvanta_{k}_tps_mean'); b=summary.get(f'llama_{k}_tps_mean')
    if a and b:
        summary[f'memvanta_vs_llama_{k}_ratio']=a/b
        summary[f'llama_vs_memvanta_{k}_speedup']=b/a

mr=summary.get('memvanta_peak_rss_kib'); lr=summary.get('llama_peak_rss_kib')
if mr and lr:
    summary['memvanta_vs_llama_rss_ratio']=mr/lr
    summary['memvanta_rss_reduction_pct']=(1-mr/lr)*100

(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
lines=['# MemVanta vs llama.cpp — real TinyStories Q4_0 A/B','',summary['disclosure'],'']
for k,v in summary.items():
    if k not in ('benchmark','disclosure'): lines.append(f'- **{k}**: {v}')
(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
