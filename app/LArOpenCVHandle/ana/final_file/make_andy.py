import os,sys,gc

if len(sys.argv) != 4:
    print
    print "ANDY_ROOT   = str(sys.argv[1])"
    print "NUM         = str(sys.args[2])"
    print "OUTDIR      = str(sys.argv[3])" 
    print 
    sys.exit(1)

ANDY_ROOT   = str(sys.argv[1])
NUM         = str(sys.argv[2])
OUTDIR      = str(sys.argv[3]) 

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

import ROOT
from lib.andy_handler import AndyHandler

FOUT = os.path.join(OUTDIR,"andy_out_%s.root" % NUM)
tf = ROOT.TFile.Open(FOUT,"RECREATE")
tf.cd()
print "OPEN %s"%FOUT

andy_tree   = ROOT.TTree("mc_tree","")

print "andy tree  @",andy_tree

andy_handler = AndyHandler(andy_tree)
andy_handler.reshape(ANDY_ROOT)

prod = "EventAndyTree"
ac = ROOT.TChain(prod)
ac.AddFile(ANDY_ROOT)
nentries = int(ac.GetEntries())
ac.GetEntry(0)

for entry in xrange(nentries):
    ac.GetEntry(entry)

    run    = int(ac.run)
    subrun = int(ac.subrun)
    event  = int(ac.event)

    print "@entry=%d (r,s,e)=(%d,%d,%d)" % (entry,run,subrun,event)

    andy_fill = andy_handler.fill(run,subrun,event,True)

    if andy_fill == False: raise Exception

andy_tree.Write()

tf.Close()
sys.exit(0)
