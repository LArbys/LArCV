#!/usr/bin/env python
import ROOT, sys
from ROOT import std
from larcv import larcv

if len(sys.argv) < 2:

   print 'Usage: python',sys.argv[0],'OUTFILE_NAME CONFIG_FILE [LARCV_FILE1 LARCV_FILE2 ...]'
   sys.exit(1)

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(sys.argv[1])

if len(sys.argv) > 1:
   
   flist=ROOT.std.vector('std::string')()
   for x in xrange(len(sys.argv)-3):
      flist.push_back(sys.argv[x+3])

   proc.override_input_file(flist)


proc.override_output_file("./out/out_"+sys.argv[2]+".root")
#proc.override_ana_file("./out/ana_"+sys.argv[2]+".root")

proc.initialize()

#proc.batch_process(0,20)
proc.batch_process()

proc.finalize()
