import ROOT
#ROOT.gSystem.Load("libLArCV")
ROOT.gSystem.Load("libLArCVData")
from ROOT import larcv
c=larcv.IOManager("larcv::Image2D")
o=c(c.kBOTH)
o.reset()
o.set_verbosity(0)
o.add_in_file("aho.root")
o.set_out_file("baka.root")
o.initialize()

o.read_entry(0)
o.get_data("aho").resize(0)
o.get_data("boke")
o.save_entry()

o.read_entry(1)
#o.get_data("aho").resize(1)
o.get_data("boke")
o.save_entry()

o.read_entry(2)
#o.get_data("aho").resize(2)
o.save_entry()

o.read_entry(3)
#o.get_data("aho").resize(3)
o.save_entry()

o.finalize()
