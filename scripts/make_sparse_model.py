#!/usr/bin/env python3
import os, argparse
p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--size',type=int,default=8,help='GiB');a=p.parse_args()
with open(a.path,'wb') as f: f.truncate(a.size*1024**3)
print(a.path, os.path.getsize(a.path))
