import os,sys,gc

if len(sys.argv) != 6:
    print 
    print "NUE_LL_PKL  = str(sys.argv[1])"
    print "NUMU_LL_PKL = str(sys.argv[2])"
    print "ANDY_ROOT   = str(sys.argv[3])"
    print "VTX_OUT     = str(sys.argv[4])"
    print "IS_MC       = bool(int(sys.argv[5]))"
    print "OUTDIR      = str(sys.argv[6])" 
    print 
    sys.exit(1)

NUE_LL_PKL  = str(sys.argv[1])
NUMU_LL_PKL = str(sys.argv[2])
ANDY_ROOT   = str(sys.argv[3])
VTX_OUT     = str(sys.argv[4])
IS_MC       = bool(int(sys.argv[5]))
OUTDIR      = str(sys.argv[6]) 
NUM         = int(os.path.basename(NUE_LL_PKL).split(".")[0].split("_")[-1])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

import ROOT
from lib.nue_handler import NueHandler
from lib.numu_handler import NumuHandler
from lib.andy_handler import andy_handler

FOUT = os.path.join(OUTDIR,"dllee_analysis_%d.root" % NUM)
tf = ROOT.TFile.Open(FOUT,"RECREATE")
tf.cd()
print "OPEN %s"%FOUT

nue_tree  = ROOT.TTree("nue_ana_tree","")
numu_tree = ROOT.TTree("numu_ana_tree","")
andy_tree = ROOT.TTree("mc_tree","")

print "nue  tree @",nue_tree
print "numu tree @",numu_tree
print "andy tree @",andy_tree

nue_handler  = NueHandler(nue_tree)
numu_handler = NumuHandler(numu_tree)
andy_handler = AndyHandler(andy_tree)

nue_handler.reshape(NUE_LL_PKL)
numu_handler.reshape(NUMU_LL_PKL)
andy_handler.reshape(ANDY_ROOT)

prod = "pgraph_test"
vtx_chain = ROOT.TChain(prod+"_tree")
vtx_chain.AddFile(VTX_OUT)
nentries = int(vtx_chain.GetEntries())
vtx_chain.GetEntry(0)
exec('br = vtx_chain.%s_branch' % prod)

for entry in xrange(nentries):
    vtx_chain.GetEntry(entry)
    run    = int(br.run())
    subrun = int(br.subrun())
    event  = int(br.event())

    print "@entry=%d (r,s,e)=(%d,%d,%d)" % (entry,run,subrun,event)

    nue_fill  = nue_handler.fill(run,subrun,event,IS_MC)
    numu_fill = numu_handler.fill(run,subrun,event,IS_MC)
    andy_fill = andy_handler.fill(run,subrun,event,IS_MC)
    
    if nue_fill  == False: raise Exception
    if numu_fill == False: raise Exception
    if andy_fill == False: raise Exception

nue_tree.Write()
numu_tree.Write()
andy_tree.Write()
tf.Close()
