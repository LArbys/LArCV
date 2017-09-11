import os,sys
import ROOT
import numpy as np
import root_numpy as rn
import pandas as pd

FILE=str(sys.argv[1])
NAME=str(os.path.basename(FILE))
df = pd.DataFrame(rn.root2array(FILE,treename="analysistree/pottree"))[['run','subrun','pot']]
df['pot_fname'] = np.array([ROOT.std.string(NAME) for _ in xrange(df.index.size)])
rec = df.to_records()
FOUT="pot_%s.root"%NAME
tf = ROOT.TFile.Open(FOUT,"RECREATE")
print "OPEN %s"%FOUT
tf.cd()
tree = rn.array2tree(rec)
tree.Write()
print "WRITE %s"%tree.GetName()
tf.Close()
print "CLOSE %s"%FOUT
                   




