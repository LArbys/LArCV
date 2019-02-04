from __future__ import print_function
import os,sys
from larcv import larcv
import ROOT as rt

inputfile  = sys.argv[1]
outputfile = sys.argv[2]
treename   = sys.argv[3]

entry = 0
if len(sys.argv)==5:
    entry = int(sys.argv[4])

# read in backward-tick larcv1, write out forward tick
io = larcv.IOManager( larcv.IOManager.kBOTH, "io", larcv.IOManager.kTickBackward )
io.add_in_file( inputfile )
if os.path.exists( outputfile ):
    print("outputfile, %s, already exists. will not overwrite"%(outputfile))
    sys.exit(0)
    
io.set_out_file( outputfile )
io.initialize()
io.set_verbosity(0)

io.read_entry(entry)
ev_img = io.get_data( larcv.kProductImage2D, treename )
if ev_img.Image2DArray().size()==0:
    print("entry %d of %s does not have any images in tree=%s"%(entry,inputfile,treename))
    sys.exit(0)
    
img = ev_img.Image2DArray().front()
print("image meta={}".format(img.meta().dump()))
larcv.load_rootutil()
h = larcv.as_th2d( img, "test" )

c = rt.TCanvas("c","c",800,600)
c.Draw()
h.Draw("colz")
c.Update()

io.save_entry()
io.finalize()

raw_input()
