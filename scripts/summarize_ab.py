#!/usr/bin/env python3
import csv, json, pathlib, re, statistics, sys
out = pathlib.Path(sys.argv[1])

def rss_kib(path):
    if not path.exists(): return None
    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', path.read_text(errors='replace'))
    return int(m.group(1)) if m else None

def memvanta_csv(path):
    rows=list(csv.DictReader(path.open()))
    # tolerate current schemas by looking for likely names
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

summary={'benchmark':'TinyStories stories15M Q4_0 native-context A/B','disclosure':'same GGUF, CPU-only, same threads; model native context is 128 so pp64/tg64 is used instead of pp512/tg128'}
if (out/'memvanta.csv').exists():
    pp,tg,rows=memvanta_csv(out/'memvanta.csv')
    if pp: summary['memvanta_pp_tps_mean']=statistics.mean(pp); summary['memvanta_pp_tps_sd']=statistics.stdev(pp) if len(pp)>1 else 0
    if tg: summary['memvanta_tg_tps_mean']=statistics.mean(tg); summary['memvanta_tg_tps_sd']=statistics.stdev(tg) if len(tg)>1 else 0
summary['memvanta_peak_rss_kib']=rss_kib(out/'memvanta.time.txt')
summary['llama_peak_rss_kib']=rss_kib(out/'llama-bench.time.txt')
lj=llama_json(out/'llama-bench.json')
if lj:
    # llama-bench JSON emits separate prompt/generation tests; capture avg_ts when present
    prompt=[]; gen=[]
    for r in lj:
        n_prompt=int(r.get('n_prompt',0) or 0); n_gen=int(r.get('n_gen',0) or 0)
        v=r.get('avg_ts') or r.get('tokens_per_second') or r.get('t/s')
        if v is None: continue
        if n_prompt and not n_gen: prompt.append(float(v))
        if n_gen and not n_prompt: gen.append(float(v))
    if prompt: summary['llama_pp_tps_mean']=statistics.mean(prompt)
    if gen: summary['llama_tg_tps_mean']=statistics.mean(gen)
for k in ('pp','tg'):
    a=summary.get(f'memvanta_{k}_tps_mean'); b=summary.get(f'llama_{k}_tps_mean')
    if a and b: summary[f'memvanta_vs_llama_{k}_ratio']=a/b
(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
lines=['# MemVanta vs llama.cpp — real TinyStories Q4_0 A/B','',summary['disclosure'],'']
for k,v in summary.items():
    if k not in ('benchmark','disclosure'): lines.append(f'- **{k}**: {v}')
(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
