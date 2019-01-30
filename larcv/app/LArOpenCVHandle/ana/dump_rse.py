import os,sys,gc
import root_numpy as rn
import pandas as pd

if len(sys.argv) != 3:
    print
    print "OPFILE = str(sys.argv[1])"
    print "OUTDIR = str(sys.argv[2])"
    print
    sys.exit(1)
    
OPFILE = str(sys.argv[1])
OUTDIR = str(sys.argv[2])
res = pd.DataFrame(rn.root2array(OPFILE,treename="larlite_id_tree"))

rename = {"_run_id" : "run",
          "_subrun_id" : "subrun",
          "_event_id"  : "event"}

res.rename(columns=rename,inplace=True)
res['entry'] = res.index
fname = "-".join(os.path.basename(OPFILE).split("opreco"))[2:-5]
res['fname'] = fname
res.to_pickle(os.path.join(OUTDIR,"%s.pkl" % fname))
sys.exit(0)
