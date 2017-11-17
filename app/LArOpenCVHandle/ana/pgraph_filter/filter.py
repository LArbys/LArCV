import os, sys, gc

if len(sys.argv) != 5:
    print 
    print "SSFILE  = str(sys.argv[1])"
    print "PGRFILE = str(sys.argv[2])"
    print "PKLFILE = str(sys.argv[3])"
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
SSFILE  = str(sys.argv[1])
PGRFILE = str(sys.argv[2])
PKLFILE = str(sys.argv[3])
num = int(os.path.basename(PGRFILE).split(".")[0].split("_")[-1])

OUTDIR  = str(sys.argv[4])
OUTFILE = os.path.basename(PGRFILE).split(".")[0].split("_")
OUTFILE = "_".join(OUTFILE[:-1]) + "_filter_" + OUTFILE[-1]

filter_df = pd.read_pickle(PKLFILE)
print "Load adrien...",PKLFILE

print "-->load larcv"
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(os.path.join(BASE_PATH,"filter.cfg"))
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(SSFILE))
flist_v.push_back(ROOT.std.string(PGRFILE))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE + ".root")))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana")) + ".root"))
proc.initialize()
proc_iom = proc.io()
my_iom = larcv.IOManager()
my_iom.add_in_file(SSFILE)
my_iom.initialize()

vertex_filter_id = proc.process_id("VertexFilter")
vertex_filter    = proc.process_ptr(vertex_filter_id)

id_v  = ROOT.std.vector("bool")()
par_v = ROOT.std.vector(ROOT.std.pair("int","int"))()

filter_df.reset_index(inplace=True)

for entry in xrange(proc_iom.get_n_entries()):

    my_iom.read_entry(entry)

    print "@entry=",entry,proc_iom.current_entry(),my_iom.current_entry()

    ev_img = my_iom.get_data(larcv.kProductImage2D,"wire")
    run    = int(ev_img.run())
    subrun = int(ev_img.subrun())
    event  = int(ev_img.event())

    print "@(rse)=(",run,",",subrun,",",event,")"
    row = filter_df.query("run==@run&subrun==@subrun&event==@event")
    if row.index.size == 0: 
        print "...nothing to see here"
        continue

    print row

    nvtx_v = row['num_vertex']
    pgid_v = row['vtxid']

    if int(nvtx_v.size)==1:
        assert int(pgid_v.size) == 1
        nvtx_v      = np.array(nvtx_v.values)
        pgid_v      = np.array(pgid_v.values)
    else:
        nvtx_v      = nvtx_v.values
        pgid_v      = pgid_v.values
        
    id_v.clear()
    id_v.resize(int(nvtx_v[0]),False)

    for vtx_id in xrange(int(nvtx_v.size)):

        nvtx = int(nvtx_v[vtx_id])
        assert nvtx == int(nvtx_v[0])

        pgid      = int(pgid_v[vtx_id])

        id_v[pgid] = True

    vertex_filter.SetIndexVector(id_v);

    print "process entry=",entry
    proc.process_entry(entry)
    print "...next"

proc.finalize()
my_iom.finalize()
sys.exit(0)
