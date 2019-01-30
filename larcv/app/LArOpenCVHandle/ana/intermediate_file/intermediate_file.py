import os,sys,gc

import pandas as pd
import ROOT

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

def main(argv):
    
    PKLFILE = str(argv[1])
    ISMC    = int(argv[2])
    OUTDIR  = str(argv[3])
    
    df = pd.read_pickle(PKLFILE)

    NUM = int(PKLFILE.split(".")[0].split("_")[-1])

    OUTFILE = "intermediate_file_%d.root" % NUM
    OUTFILE = os.path.join(OUTDIR,OUTFILE)

    tf = ROOT.TFile.Open(OUTFILE,"RECREATE")
    tf.cd()
    
    tree = ROOT.TTree("vertex_tree","")
    
    from lib.rootdata import ROOTData
    rd = ROOTData(df,ISMC)
    
    rd.init_tree(tree)
    
    for rowid in xrange(int(df.shape[0])):
        print "@rowid=%d" % rowid
        rd.fill(df.iloc[rowid],tree)
        print "... filled"
        rd.reset()
    
    tree.Write()
    tf.Close()

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print
        print "PKLFILE = str(sys.argv[1])"
        print "ISMC    = int(sys.argv[2])"
        print "OUTDIR  = str(sys.argv[3])"
        print
        sys.exit(1)
    
    main(sys.argv)
    
    sys.exit(0)
