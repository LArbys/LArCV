import os,sys
import sys
import numpy as np
import pandas as pd
import itertools

sig = pd.read_pickle("signal.pkl")
bkg = pd.read_pickle("background.pkl")

d=int(sys.argv[1])
print "@d=",d
xlo=-60
xhi=0
dx=0.5
bins=np.arange(xlo,xhi+dx,dx)
highest = -1.0
ccc     = np.array([])
ppos    = 0.0

d = int(d)
for comb in itertools.combinations(np.arange(0,24),d):
    cc = np.array(comb)
    
    hs=np.histogram(sig.values[:,cc].sum(axis=1),bins=bins)
    hb=np.histogram(bkg.values[:,cc].sum(axis=1),bins=bins)
    
    hssum=float(hs[0].sum())
    hbsum=float(hb[0].sum())
    
    pos = np.where((np.add.accumulate(hb[0]) / hbsum)>0.995)[0][0]
    res = (1.0 - np.add.accumulate(hs[0]) / hssum)[pos]
    
    if res > highest:
        highest = res
        ccc     = cc
        ppos    = pos

print highest,list(ccc),ppos
with open("out/%03d.txt" % d,"w+") as f:
    f.write("{},{},{}".format(highest,str(list(ccc)),ppos))
    f.write("\n")
