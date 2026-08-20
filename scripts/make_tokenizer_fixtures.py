#!/usr/bin/env python3
import argparse, struct, tempfile
from pathlib import Path
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

U32=4; F32=6; BOOL=7; STRING=8; ARRAY=9; I32=5

def gs(x: str):
    b=x.encode('utf-8'); return struct.pack('<Q',len(b))+b

def kv_str(k,v): return gs(k)+struct.pack('<I',STRING)+gs(v)
def kv_u32(k,v): return gs(k)+struct.pack('<I',U32)+struct.pack('<I',v)
def kv_bool(k,v): return gs(k)+struct.pack('<I',BOOL)+struct.pack('<B',1 if v else 0)
def kv_strarr(k,a): return gs(k)+struct.pack('<I',ARRAY)+struct.pack('<I',STRING)+struct.pack('<Q',len(a))+b''.join(gs(x) for x in a)
def kv_farr(k,a): return gs(k)+struct.pack('<I',ARRAY)+struct.pack('<I',F32)+struct.pack('<Q',len(a))+struct.pack('<%df'%len(a),*a)
def kv_iarr(k,a): return gs(k)+struct.pack('<I',ARRAY)+struct.pack('<I',I32)+struct.pack('<Q',len(a))+struct.pack('<%di'%len(a),*a)
def write_gguf(path, meta):
    blob=b'GGUF'+struct.pack('<IQQ',3,0,len(meta))+b''.join(meta)
    Path(path).write_bytes(blob)

def byte_unicode():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b);cs.append(256+n);n+=1
    return {b:chr(c) for b,c in zip(bs,cs)}

def make_gpt2(out):
    m=byte_unicode(); special=['<|endoftext|>','<|im_start|>','<|im_end|>']
    toks=special+[m[i] for i in range(256)]
    types=[3,3,3]+[1]*256
    # A few deterministic merges to exercise ranking and pre-token boundaries.
    merges=['h e','he l','hel l','hell o',f'{m[32]} w',f'{m[32]}w o',f'{m[32]}wo r',f'{m[32]}wor l',f'{m[32]}worl d']
    for mg in merges:
        a,b=mg.split(' ',1); z=a+b
        if z not in toks: toks.append(z);types.append(1)
    meta=[kv_str('tokenizer.ggml.model','gpt2'),kv_str('tokenizer.ggml.pre','smollm'),kv_bool('tokenizer.ggml.add_space_prefix',False),kv_strarr('tokenizer.ggml.tokens',toks),kv_iarr('tokenizer.ggml.token_type',types),kv_strarr('tokenizer.ggml.merges',merges),kv_u32('tokenizer.ggml.bos_token_id',1),kv_u32('tokenizer.ggml.eos_token_id',2),kv_u32('tokenizer.ggml.unknown_token_id',0)]
    write_gguf(out,meta)

def make_sp(out, model_prefix):
    corpus=Path(model_prefix).with_suffix('.txt')
    corpus.write_text('''Hello world\nThe quick brown fox jumps over the lazy dog.\nData science and machine learning in C++ are useful.\nनमस्ते दुनिया\nभारत में कृत्रिम बुद्धिमत्ता\nこんにちは 世界\n你好 世界\nemoji 😀 🚀 ❤️\nCafé naïve résumé coöperate\nif (x < 10) { return x * x; }\nspaces   tabs\tand newlines\n''',encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=model_prefix,vocab_size=512,model_type='unigram',byte_fallback=True,character_coverage=1.0,normalization_rule_name='identity',add_dummy_prefix=True,remove_extra_whitespaces=False,bos_id=1,eos_id=2,unk_id=0,pad_id=-1,hard_vocab_limit=False)
    proto=pb.ModelProto();proto.ParseFromString(Path(model_prefix+'.model').read_bytes())
    toks=[];scores=[];types=[]
    typemap={pb.ModelProto.SentencePiece.NORMAL:1,pb.ModelProto.SentencePiece.UNKNOWN:2,pb.ModelProto.SentencePiece.CONTROL:3,pb.ModelProto.SentencePiece.USER_DEFINED:4,pb.ModelProto.SentencePiece.UNUSED:5,pb.ModelProto.SentencePiece.BYTE:6}
    for p in proto.pieces:
        toks.append(p.piece);scores.append(p.score);types.append(typemap[p.type])
    meta=[kv_str('tokenizer.ggml.model','llama'),kv_str('tokenizer.ggml.pre','default'),kv_bool('tokenizer.ggml.add_space_prefix',True),kv_strarr('tokenizer.ggml.tokens',toks),kv_farr('tokenizer.ggml.scores',scores),kv_iarr('tokenizer.ggml.token_type',types),kv_u32('tokenizer.ggml.bos_token_id',1),kv_u32('tokenizer.ggml.eos_token_id',2),kv_u32('tokenizer.ggml.unknown_token_id',0)]
    write_gguf(out,meta)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('outdir');a=ap.parse_args();d=Path(a.outdir);d.mkdir(parents=True,exist_ok=True)
    make_gpt2(d/'gpt2_smollm_tokenizer.gguf')
    make_sp(d/'llama_sentencepiece_tokenizer.gguf',str(d/'sp_fixture'))
    print(d)
