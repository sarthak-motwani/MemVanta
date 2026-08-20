#!/usr/bin/env python3
import argparse, math, random, struct
from pathlib import Path
U32=4; F32=6; STRING=8; ARRAY=9

def s(x: str):
    b=x.encode(); return struct.pack('<Q',len(b))+b

def kv_str(k,v): return s(k)+struct.pack('<I',STRING)+s(v)
def kv_u32(k,v): return s(k)+struct.pack('<I',U32)+struct.pack('<I',v)
def kv_f32(k,v): return s(k)+struct.pack('<I',F32)+struct.pack('<f',v)
def kv_strarr(k,a): return s(k)+struct.pack('<I',ARRAY)+struct.pack('<I',STRING)+struct.pack('<Q',len(a))+b''.join(s(x) for x in a)

def q4_row(vals):
    assert len(vals)%32==0
    out=bytearray()
    for j in range(0,len(vals),32):
        x=vals[j:j+32]; ma=max(abs(v) for v in x); d=ma/7 if ma else 0.0
        qs=[]
        for v in x:
            q=max(-8,min(7,round(v/d) if d else 0)); qs.append(q+8)
        out += struct.pack('<e',d)
        out += bytes((qs[i] | (qs[i+16]<<4)) for i in range(16))
    return bytes(out)

def f32(vals): return struct.pack('<%df'%len(vals),*vals)
def mat(rows,cols,rng,scale=0.08):
    return [[rng.uniform(-scale,scale) for _ in range(cols)] for _ in range(rows)]
def qmat(rows,cols,rng,scale=.08): return b''.join(q4_row(r) for r in mat(rows,cols,rng,scale))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');a=ap.parse_args();rng=random.Random(123)
    H,FF,L,NH,NKV,V,CTX=32,64,2,4,2,256,128
    meta=[kv_str('general.architecture','llama'),kv_str('general.name','MemVanta Tiny Real GGUF'),kv_u32('general.alignment',32),kv_u32('llama.context_length',CTX),kv_u32('llama.embedding_length',H),kv_u32('llama.feed_forward_length',FF),kv_u32('llama.block_count',L),kv_u32('llama.attention.head_count',NH),kv_u32('llama.attention.head_count_kv',NKV),kv_f32('llama.attention.layer_norm_rms_epsilon',1e-5),kv_f32('llama.rope.freq_base',10000.0),kv_strarr('tokenizer.ggml.tokens',[chr(i) if 32<=i<127 else f'<{i}>' for i in range(V)]),kv_u32('tokenizer.ggml.bos_token_id',1),kv_u32('tokenizer.ggml.eos_token_id',2)]
    tensors=[]
    def add(name,dims,typ,data): tensors.append((name,dims,typ,data))
    add('token_embd.weight',[H,V],2,qmat(V,H,rng,.08))
    for i in range(L):
        add(f'blk.{i}.attn_norm.weight',[H],0,f32([1+rng.uniform(-.02,.02) for _ in range(H)]))
        add(f'blk.{i}.attn_q.weight',[H,H],2,qmat(H,H,rng,.06))
        add(f'blk.{i}.attn_k.weight',[H,H*NKV//NH],2,qmat(H*NKV//NH,H,rng,.06))
        add(f'blk.{i}.attn_v.weight',[H,H*NKV//NH],2,qmat(H*NKV//NH,H,rng,.06))
        add(f'blk.{i}.attn_output.weight',[H,H],2,qmat(H,H,rng,.06))
        add(f'blk.{i}.ffn_norm.weight',[H],0,f32([1+rng.uniform(-.02,.02) for _ in range(H)]))
        add(f'blk.{i}.ffn_gate.weight',[H,FF],2,qmat(FF,H,rng,.05))
        add(f'blk.{i}.ffn_up.weight',[H,FF],2,qmat(FF,H,rng,.05))
        add(f'blk.{i}.ffn_down.weight',[FF,H],2,qmat(H,FF,rng,.05))
    add('output_norm.weight',[H],0,f32([1.0]*H))
    # tied output: no output.weight
    hdr=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))+b''.join(meta)
    offsets=[];off=0
    for _,_,_,data in tensors:
        off=(off+31)//32*32;offsets.append(off);off+=len(data)
    ti=bytearray()
    for (name,dims,typ,data),off in zip(tensors,offsets):
        ti+=s(name)+struct.pack('<I',len(dims))+struct.pack('<%dQ'%len(dims),*dims)+struct.pack('<IQ',typ,off)
    prefix=hdr+ti; data_start=(len(prefix)+31)//32*32
    out=bytearray(prefix)+bytearray(data_start-len(prefix));cur=0
    for (_,_,_,data),off in zip(tensors,offsets):
        target=data_start+off
        if len(out)<target: out+=bytearray(target-len(out))
        out+=data
    Path(a.out).write_bytes(out)
    print(a.out,len(out),'bytes','tensors',len(tensors),'data_start',data_start)
if __name__=='__main__':main()
