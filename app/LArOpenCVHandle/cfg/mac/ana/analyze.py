import ROOT, sys
from ROOT import std
from larcv import larcv
if len(sys.argv) < 2:
   print 'Usage: python',sys.argv[0],'CONFIG_FILE [LARCV_FILE1 LARCV_FILE2 ...]'
   sys.exit(1)
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(sys.argv[1])
if len(sys.argv) > 1:
   flist=ROOT.std.vector('std::string')()
   for x in xrange(len(sys.argv)-3):
      flist.push_back(sys.argv[x+3])
   proc.override_input_file(flist)
ana_id = proc.process_id("LArbysImageAna")
extract_id = proc.process_id("LArbysImageExtract")
larbysimg_ana = proc.process_ptr(ana_id)
larbysimg_ana.SetInputLArbysFile(sys.argv[2])
larbysimg_extract = proc.process_ptr(extract_id)
proc.initialize()

for entry in xrange(100):
   if not proc.process_entry(entry): continue

   for ix,vertex in enumerate(larbysimg_ana.Verticies()):
      print ix,") ",vertex

proc.finalize()

