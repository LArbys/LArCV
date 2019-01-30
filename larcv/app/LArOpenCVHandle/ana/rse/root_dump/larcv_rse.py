import ROOT, sys
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(ROOT.std.string("lcv.cfg"))
proc.override_input_file(ROOT.std.vector("std::string")(1,sys.argv[2]))
proc.override_ana_file("lc_ana/lc_ana_%d.root" % int(sys.argv[1]))
proc.initialize()
proc.batch_process()
proc.finalize()
