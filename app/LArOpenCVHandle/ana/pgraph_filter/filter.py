import os, sys, gc

if len(sys.argv) != 5:
    print 
    print "PGRFILE    = str(sys.argv[1])"
    print "FINAL_FILE = str(sys.argv[2])"
    print "TREE       = str(sys.argv[3])"
    print "OUTDIR     = str(sys.argv[4])" 
    print 
    sys.exit(1)

import ROOT
import numpy as np
import root_numpy as rn
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

print 
print "--> initialize"
print
PGRFILE    = str(sys.argv[1])
FINAL_FILE = str(sys.argv[2])
TREE       = str(sys.argv[3])

num = int(os.path.basename(PGRFILE).split(".")[0].split("_")[-1])

OUTDIR  = str(sys.argv[4])
OUTFILE = os.path.basename(PGRFILE).split(".")[0].split("_")
OUTFILE = "_".join(OUTFILE[:-1]) + "_filter_"+ TREE + "_"  + OUTFILE[-1]

print
print "--> load final file"
print 
branches = ["run","subrun","event","vertex_id","num_vertex"]
final_df = pd.DataFrame(rn.root2array(FINAL_FILE,
                                      treename=TREE,
                                      branches=branches))

print
print "--> load larcv"
print
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(os.path.join(BASE_PATH,"filter.cfg"))
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(PGRFILE))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE + ".root")))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana")) + ".root"))
proc.initialize()
proc_iom = proc.io()
my_iom = larcv.IOManager()
my_iom.add_in_file(PGRFILE)
my_iom.initialize()

vertex_filter_id = proc.process_id("VertexFilter")
vertex_filter    = proc.process_ptr(vertex_filter_id)

print "Got vertex filter @id=%s @ptr=%s" % (vertex_filter_id,vertex_filter)

id_v  = ROOT.std.vector("bool")()
par_v = ROOT.std.vector(ROOT.std.pair("int","int"))()

for entry in xrange(proc_iom.get_n_entries()):

    my_iom.read_entry(entry)

    print "@entry=%s @proc_iom=%s @my_ion=%s"%(entry,proc_iom.current_entry(),my_iom.current_entry())

    ev_pgraph = my_iom.get_data(larcv.kProductPGraph,"test")
    run    = int(ev_pgraph.run())
    subrun = int(ev_pgraph.subrun())
    event  = int(ev_pgraph.event())

    print "@(rse)=(",run,",",subrun,",",event,")"
    row = final_df.query("run==@run&subrun==@subrun&event==@event")

    if row.index.size == 0: 
        print "... empty row! die."
        sys.exit(1)

    num_vertex = int(row['num_vertex'])
    pgraph_id  = int(row['vertex_id'])

    # no vertex, skip
    if num_vertex == 0: 
        print "... no vertex next!"
        proc.process_entry(entry)
        continue

    # was vertex but none with valid LL_dist
    if pgraph_id < 0: 
        print "... vertex there but invalid LL next!"
        proc.process_entry(entry)
        continue
    
    if pgraph_id >= num_vertex:
        print "... invalid pgraph identified! die."
        sys.exit(1)

    id_v.clear()
    id_v.resize(num_vertex,False)
    par_v.resize(num_vertex,ROOT.std.pair("int","int")(0,0))

    id_v[pgraph_id] = True

    print "set pgraph_id=%d to True" % pgraph_id 

    vertex_filter.SetIndexVector(id_v);

    print "process entry=",entry
    proc.process_entry(entry)
    print "...next!"

proc.finalize()
my_iom.finalize()
sys.exit(0)
