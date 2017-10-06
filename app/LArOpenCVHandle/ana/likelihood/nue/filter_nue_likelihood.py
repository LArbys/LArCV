#https://open.spotify.com/track/7DcWaEjsO3PTAODV2fKeSn
import os, sys

if len(sys.argv) != 5:
    print 
    print "ANAFILE = str(sys.argv[1])"
    print "PGRFILE = str(sys.argv[2])"
    print "LLCUT   = float(sys.argv[3])"
    print "OUTDIR  = str(sys.argv[4])" 
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd
BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

print "-->initialize"
ANAFILE = str(sys.argv[1])
PGRFILE = str(sys.argv[2])
LLCUT   = float(sys.argv[3])
OUTDIR  = str(sys.argv[4])
OUTFILE = os.path.basename(PGRFILE).split(".")[0].split("_")
OUTFILE = "_".join(OUTFILE[:-1]) + "_nue_LL_filter_" + OUTFILE[-1]

print "-->apply LL"
from util.fill_df import *
pdf_fin = os.path.join(BASE_PATH,"ll_bin","nue_pdfs.root")
LL_df = initialize_df(ANAFILE)
LL_df = nue_assumption(LL_df)
LL_df = fill_parameters(LL_df)
LL_df = apply_ll(LL_df,pdf_fin)

LL_df = LL_df.query("LL>@LLCUT")

print "-->load larcv"
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(os.path.join(BASE_PATH,"filter_likelihood.cfg"))
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(PGRFILE))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE + ".root")))
proc.initialize()

vertex_filter_id = proc.process_id("VertexFilter")
vertex_filter    = proc.process_ptr(vertex_filter_id)

id_v  = ROOT.std.vector("bool")()
par_v = ROOT.std.vector(ROOT.std.pair("int","int"))()

for name, row in LL_df.iterrows():
    entry = int(row['entry'])
    assert entry < proc.io().get_n_entries()

    nvtx = int(row['num_vertex'])
    pgid = int(row['cvtxid'])

    id_v.clear()
    id_v.resize(nvtx,False)

    par_v.clear()
    par_v.resize(nvtx,ROOT.std.pair("int","int")(-1,-1))

    print "insert pgid=",pgid

    id_v[pgid] = True
    
    shower_id = row['shrid']
    track_id  = row['trkid']

    exec("par%d_id=0" % shower_id)
    exec("par%d_id=1" % track_id)

    print "setting..."
    print "\tpar0_id=",par0_id
    print "\tpar1_id=",par1_id

    par_v[pgid] = ROOT.std.pair("int","int")(par0_id,par1_id)

    vertex_filter.SetIndexVector(id_v);
    vertex_filter.SetParticleType(par_v);

    print "process entry=",entry
    proc.process_entry(entry)
    print "...next"

proc.finalize()

sys.exit(0)
