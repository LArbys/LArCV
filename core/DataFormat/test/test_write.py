import ROOT,sys
#ROOT.gSystem.Load("libLArCV")
ROOT.gSystem.Load("libLArCVDataFormat")
from ROOT import larcv
larcv.logger.force_level(0)
#c=larcv.IOManager("larcv::Image2D")

o=larcv.IOManager(larcv.IOManager.kWRITE)
#sys.exit(1)
#o=c(c.kWRITE)
o.reset()
o.set_verbosity(0)
o.set_out_file("aho.root")
o.initialize()

ptr=o.get_data(larcv.kProductImage2D,"aho")
print ptr.Image2DArray().size()
img=larcv.Image2D()
ptr.Append(img)
print ptr.Image2DArray().size()
o.get_data(larcv.kProductROI,"boke")
o.set_id(1,1,0)
o.save_entry()

#o.get_data("aho").resize(1)
o.get_data(larcv.kProductImage2D,"boke")
o.set_id(1,1,1)
o.save_entry()

#o.get_data("aho").resize(2)
o.set_id(1,1,2)
o.save_entry()

#o.get_data("aho").resize(3)
o.set_id(1,1,3)
o.save_entry()

o.finalize()
