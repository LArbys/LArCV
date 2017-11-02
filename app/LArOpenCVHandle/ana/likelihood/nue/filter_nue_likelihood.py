#https://open.spotify.com/track/7DcWaEjsO3PTAODV2fKeSn
import os, sys, gc

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
num = int(os.path.basename(PGRFILE).split(".")[0].split("_")[-1])

LLCUT   = str(sys.argv[3])
if LLCUT == "None":
    LLCUT = None
else:
    LLCUT = float(LLCUT)

OUTDIR  = str(sys.argv[4])
OUTFILE = os.path.basename(PGRFILE).split(".")[0].split("_")
OUTFILE = "_".join(OUTFILE[:-1]) + "_nue_LL_filter_" + OUTFILE[-1]

print "-->apply LL"
from util.fill_df import *
pdf_fin = os.path.join(BASE_PATH,"ll_bin","nue_pdfs.root")

#
# init all 
# 
all_df = initialize_df(ANAFILE)

#
# apply nue assumption
#
nue_df = nue_assumption(all_df)

#
# reap all
#
all_df.to_pickle(os.path.join(OUTDIR,"ana_all_df_%d.pkl" % num))
del all_df
gc.collect()

#
# fill nue
#
nue_df = fill_parameters(nue_df)

#
# fill LL
#
LL_df = apply_ll(nue_df,pdf_fin)

#
# reap nue
#
nue_df.to_pickle(os.path.join(OUTDIR,"ana_nue_df_%d.pkl" % num))
del nue_df
gc.collect()

#
# store LL
#
LL_sel_df = maximize_ll(LL_df)

if LLCUT != None:
    LL_sel_df = LL_sel_df.query("LL>@LLCUT")

LL_df.to_pickle(os.path.join(OUTDIR,"ana_LL_df_%d.pkl" % num))
del LL_df
gc.collect()

LL_sel_df.to_pickle(os.path.join(OUTDIR,"ana_LL_sel_df_%d.pkl" % num))

print "-->load larcv"
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(os.path.join(BASE_PATH,"filter_likelihood.cfg"))
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(PGRFILE))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE + ".root")))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana")) + ".root"))
proc.initialize()

vertex_filter_id = proc.process_id("VertexFilter")
vertex_filter    = proc.process_ptr(vertex_filter_id)

id_v  = ROOT.std.vector("bool")()
par_v = ROOT.std.vector(ROOT.std.pair("int","int"))()

LL_sel_df.reset_index(inplace=True)
LL_sel_df.set_index('entry',inplace=True)

for entry in xrange(proc.io().get_n_entries()):
    print "@entry=",entry

    #
    # do nothing
    #
    if entry not in LL_sel_df.index:
        proc.process_entry(entry)
        continue
        
    #
    # do something
    #
    row  = LL_sel_df.loc[entry]

    nvtx_v = row['num_vertex']
    pgid_v = row['cvtxid']

    shower_id_v = row['shrid']
    track_id_v  = row['trkid']

    if int(nvtx_v.size)==1:
        assert pgid_v.size == 1
        assert shower_id_v.size == 1
        assert track_id_v.size == 1

        nvtx_v      = np.array([nvtx_v])
        pgid_v      = np.array([pgid_v])
        shower_id_v = np.array([shower_id_v])
        track_id_v  = np.array([track_id_v])
    else:
        nvtx_v      = nvtx_v.values
        pgid_v      = pgid_v.values
        shower_id_v = shower_id_v.values
        track_id_v  = track_id_v.values
        
    id_v.clear()
    id_v.resize(int(nvtx_v[0]),False)

    par_v.clear()
    par_v.resize(int(nvtx_v[0]),ROOT.std.pair("int","int")(-1,-1))

    for vtx_id in xrange(int(nvtx_v.size)):

        nvtx = int(nvtx_v[vtx_id])
        assert nvtx == int(nvtx_v[0])

        pgid      = int(pgid_v[vtx_id])
        shower_id = int(shower_id_v[vtx_id])
        track_id  = int(track_id_v[vtx_id])

        print "insert pgid=",pgid

        id_v[pgid] = True
                
        SS="par%d_id=0" % shower_id
        exec(SS)
        SS="par%d_id=1" % track_id
        exec(SS)

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
