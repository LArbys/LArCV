import os, sys

if len(sys.argv) != 5:
    print
    print "PKL    = str(sys.argv[1])"
    print "RUN    = str(sys.argv[2])"
    print "SUBRUN = str(sys.argv[3])"
    print "EVENT  = str(sys.argv[4])"
    print
    sys.exit(1)

PKL    = str(sys.argv[1])
RUN    = str(sys.argv[2])
SUBRUN = str(sys.argv[3])
EVENT  = str(sys.argv[4])

import pandas as pd

df = pd.read_pickle(PKL);

row = df.query("run==@RUN&subrun==@SUBRUN&event==@EVENT").iloc[0]
ENTRY = int(row['entry'])
fname = str(row['fname'])

_RUN    = int(fname.split("-")[0][3:])
_SUBRUN = int(fname.split("-")[-1][6:])

runmod100    = _RUN%100
rundiv100    = _RUN/100
subrunmod100 = _SUBRUN%100
subrundiv100 = _SUBRUN/100

jobtag = 10000*_RUN + _SUBRUN
out = []
out.append(str(ENTRY))
out.append(str(jobtag))
out.append("/%03d/%02d/%03d/%02d/"%(rundiv100,runmod100,subrundiv100,subrunmod100))
print out
