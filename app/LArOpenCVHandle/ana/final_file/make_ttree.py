import os,sys,gc

if len(sys.argv) != 10:
    print 
    print "VTX_OUT     = str(sys.argv[1])"
    print "VTX_PKL     = str(sys.argv[2])"
    print "TRK_ANA     = str(sys.argv[3])"
    print "FINAL_VARS  = str(sys.argv[4])"
    print "MC_INFORM   = str(sys.argv[5])"
    print "NUE_CUT_DF  = str(sys.argv[6])"
    print "NUE_LL_DF   = str(sys.argv[7])"
    print "NUM         = str(sys.argv[8])"
    print "OUTDIR      = str(sys.argv[9])" 
    print 
    sys.exit(1)

VTX_OUT    = str(sys.argv[1])
VTX_PKL    = str(sys.argv[2])
TRK_ANA    = str(sys.argv[3])
FINAL_VARS = str(sys.argv[4])
MC_INFORM  = str(sys.argv[5])
NUE_CUT_DF = str(sys.argv[6])
NUE_LL_DF  = str(sys.argv[7])
NUM        = str(sys.argv[8])
OUTDIR     = str(sys.argv[9])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

import ROOT

from lib.numu_handler import NumuHandler
from lib.nue_cut_handler import NueCutHandler
from lib.nue_ll_handler import NueLLHandler
from lib.segment_handler import SegmentHandler

FOUT = os.path.join(OUTDIR,"dllee_analysis_%s.root" % NUM)
tf = ROOT.TFile.Open(FOUT,"RECREATE")
tf.cd()
print "OPEN %s"%FOUT

numu_ana_tree    = ROOT.TTree("numu_ana_tree","")
nue_cut_ana_tree = ROOT.TTree("nue_cut_ana_tree","")
nue_ll_ana_tree  = ROOT.TTree("nue_ll_ana_tree","")
segment_tree     = ROOT.TTree("segment_tree","")

print "numu_tree @",numu_ana_tree
print "nue_cut_tree @",nue_cut_ana_tree
print "nue_ll_tree @",nue_ll_ana_tree 
print "segment tree @",segment_tree

numu_handler    = NumuHandler(numu_ana_tree)
nue_cut_handler = NueCutHandler(nue_cut_ana_tree)
nue_ll_handler  = NueLLHandler(nue_ll_ana_tree)
segment_handler = SegmentHandler(segment_tree)

IS_MC = True
if os.path.exists(MC_INFORM) == False:
    print "--> is data"
    MC_INFORM = None
    IS_MC = False

numu_handler.reshape(VTX_PKL,TRK_ANA,FINAL_VARS,MC_INFORM)
nue_cut_handler.reshape(NUE_CUT_DF,MC_INFORM)
nue_ll_handler.reshape(NUE_LL_DF,MC_INFORM)
segment_handler.reshape(MC_INFORM)

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
    
    numu_fill    = numu_handler.fill(run,subrun,event,IS_MC)
    nue_cut_fill = nue_cut_handler.fill(run,subrun,event,IS_MC)
    nue_ll_fill  = nue_ll_handler.fill(run,subrun,event,IS_MC)
    seg_fill     = segment_handler.fill(run,subrun,event,IS_MC)

    if numu_fill == False: raise Exception
    if nue_cut_fill == False: raise Exception
    if nue_ll_fill == False: raise Exception
    if seg_fill == False: raise Exception

tf.cd()
numu_ana_tree.Write()
nue_cut_ana_tree.Write()
nue_ll_ana_tree.Write()
segment_tree.Write()

tf.Close()
