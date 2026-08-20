#!/usr/bin/env python3
import argparse, math, random, struct
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--dim',type=int,default=64);p.add_argument('--hidden',type=int,default=172);p.add_argument('--layers',type=int,default=5);p.add_argument('--heads',type=int,default=8);p.add_argument('--kv-heads',type=int,default=4);p.add_argument('--vocab',type=int,default=512);p.add_argument('--ctx',type=int,default=512);a=p.parse_args()
assert a.dim%a.heads==0
rng=random.Random(1234)
def vals(n,scale):
    # deterministic, low-magnitude fixture for numerical stability
    for _ in range(n): yield (rng.random()*2-1)*scale
with open(a.out,'wb') as f:
    f.write(struct.pack('7i',a.dim,a.hidden,a.layers,a.heads,a.kv_heads,a.vocab,a.ctx))
    head=a.dim//a.heads;kv=a.kv_heads*head
    groups=[
      (a.vocab*a.dim,.02),(a.layers*a.dim,1.0),(a.layers*a.dim*a.dim,.02),
      (a.layers*a.dim*kv,.02),(a.layers*a.dim*kv,.02),(a.layers*a.dim*a.dim,.02),
      (a.layers*a.dim,1.0),(a.layers*a.dim*a.hidden,.02),(a.layers*a.hidden*a.dim,.02),
      (a.layers*a.dim*a.hidden,.02),(a.dim,1.0),(a.ctx*head//2,0.0),(a.ctx*head//2,0.0)]
    for n,scale in groups:
        buf=[]
        for x in vals(n,scale):
            buf.append(x)
            if len(buf)==16384:
                f.write(struct.pack('<%df'%len(buf),*buf));buf=[]
        if buf:f.write(struct.pack('<%df'%len(buf),*buf))
print(a.out)
