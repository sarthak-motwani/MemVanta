#!/usr/bin/env python3
"""Create a deterministic GGUF v3 with SmolLM2-135M tensor shapes/types.
This is a systems fixture only: it does NOT contain the trained SmolLM2 weights.
"""
import argparse, random, struct
from pathlib import Path
U32=4; F32=6; STRING=8; ARRAY=9
GGML_F32=0; GGML_Q4_0=2; GGML_Q8_0=8

def s(x):
    b=x.encode('utf-8'); return struct.pack('<Q',len(b))+b
def kv_str(k,v): return s(k)+struct.pack('<I',STRING)+s(v)
def kv_u32(k,v): return s(k)+struct.pack('<I',U32)+struct.pack('<I',v)
def kv_f32(k,v): return s(k)+struct.pack('<I',F32)+struct.pack('<f',v)
def kv_strarr(k,a): return s(k)+struct.pack('<I',ARRAY)+struct.pack('<I',STRING)+struct.pack('<Q',len(a))+b''.join(s(x) for x in a)
def f32_ones(n): return struct.pack('<%df'%n,*([1.0]*n))

def q4_bytes(rows, cols, rng):
    assert cols%32==0
    block=struct.pack('<e',0.02)
    out=bytearray(rows*(cols//32)*18); p=0
    for _ in range(rows*(cols//32)):
        out[p:p+2]=block; p+=2
        out[p:p+16]=bytes(rng.randrange(256) for _ in range(16)); p+=16
    return bytes(out)

def q8_bytes(rows, cols, rng):
    assert cols%32==0
    scale=struct.pack('<e',0.002)
    out=bytearray(rows*(cols//32)*34); p=0
    for _ in range(rows*(cols//32)):
        out[p:p+2]=scale; p+=2
        out[p:p+32]=bytes((rng.randrange(-40,41)&255) for _ in range(32)); p+=32
    return bytes(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('out'); args=ap.parse_args(); rng=random.Random(20260820)
    H,FF,L,NH,NKV,V,CTX=576,1536,30,9,3,49152,8192
    meta=[
      kv_str('general.architecture','llama'),kv_str('general.name','MemVanta SmolLM2-135M shape fixture'),kv_u32('general.alignment',32),kv_u32('general.file_type',2),
      kv_u32('llama.context_length',CTX),kv_u32('llama.embedding_length',H),kv_u32('llama.feed_forward_length',FF),kv_u32('llama.block_count',L),kv_u32('llama.attention.head_count',NH),kv_u32('llama.attention.head_count_kv',NKV),
      kv_f32('llama.attention.layer_norm_rms_epsilon',1e-5),kv_f32('llama.rope.freq_base',100000.0),
      kv_str('tokenizer.ggml.model','gpt2'),kv_str('tokenizer.ggml.pre','smollm'),
      kv_strarr('tokenizer.ggml.tokens',[f'<tok{i}>' for i in range(V)]),kv_u32('tokenizer.ggml.bos_token_id',1),kv_u32('tokenizer.ggml.eos_token_id',2)]
    tensors=[]
    def add(name,dims,typ,data): tensors.append((name,dims,typ,data))
    add('token_embd.weight',[H,V],GGML_Q8_0,q8_bytes(V,H,rng))
    KVD=H*NKV//NH
    for i in range(L):
      add(f'blk.{i}.attn_norm.weight',[H],GGML_F32,f32_ones(H))
      add(f'blk.{i}.attn_q.weight',[H,H],GGML_Q4_0,q4_bytes(H,H,rng))
      add(f'blk.{i}.attn_k.weight',[H,KVD],GGML_Q4_0,q4_bytes(KVD,H,rng))
      add(f'blk.{i}.attn_v.weight',[H,KVD],GGML_Q4_0,q4_bytes(KVD,H,rng))
      add(f'blk.{i}.attn_output.weight',[H,H],GGML_Q4_0,q4_bytes(H,H,rng))
      add(f'blk.{i}.ffn_norm.weight',[H],GGML_F32,f32_ones(H))
      add(f'blk.{i}.ffn_gate.weight',[H,FF],GGML_Q4_0,q4_bytes(FF,H,rng))
      add(f'blk.{i}.ffn_up.weight',[H,FF],GGML_Q4_0,q4_bytes(FF,H,rng))
      add(f'blk.{i}.ffn_down.weight',[FF,H],GGML_Q4_0,q4_bytes(H,FF,rng))
    add('output_norm.weight',[H],GGML_F32,f32_ones(H))
    hdr=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))+b''.join(meta)
    offsets=[]; off=0
    for _,_,_,data in tensors:
      off=(off+31)//32*32; offsets.append(off); off+=len(data)
    ti=bytearray()
    for (name,dims,typ,_),o in zip(tensors,offsets):
      ti += s(name)+struct.pack('<I',len(dims))+struct.pack('<%dQ'%len(dims),*dims)+struct.pack('<IQ',typ,o)
    prefix=hdr+ti; data_start=(len(prefix)+31)//32*32
    out=bytearray(prefix)+bytearray(data_start-len(prefix))
    for (_,_,_,data),o in zip(tensors,offsets):
      target=data_start+o
      if len(out)<target: out += bytearray(target-len(out))
      out += data
    Path(args.out).write_bytes(out)
    print(f'{args.out}: {len(out)/1048576:.2f} MiB, tensors={len(tensors)}, dims H={H} FF={FF} L={L} heads={NH}/{NKV} vocab={V}')
if __name__=='__main__': main()
