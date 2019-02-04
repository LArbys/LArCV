import os,sys
from larcv import larcv
import ROOT as rt

"""
load in file, display image from entry
"""

inputfile = sys.argv[1]
imagetree = sys.argv[2]
if len(sys.argv)==4:
    entry = int(sys.argv[3])
else:
    entry = 0

# read in backward-tick larcv1, write out forward tick
#io = larcv.IOManager( larcv.IOManager.kBOTH, "io", larcv.IOManager.kTickBackward )
#io.add_in_file( "testdata/supera-Run000001-SubRun006867.root" )
#io.set_out_file( "test.root" )

# read in forward-tick larcv1
io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( inputfile )

io.initialize()
io.set_verbosity(0)

io.read_entry(entry)
ev_img = io.get_data( larcv.kProductImage2D, imagetree )

if ev_img.Image2DArray().size()>0:
    img = ev_img.Image2DArray().front()
else:
    print "No images in tree=%s and entry=%d"%(imagetree,entry)

rt.gStyle.SetOptStat(0)
larcv.load_rootutil()    
h = larcv.as_th2d( img, "test" )

c = rt.TCanvas("c","c",800,600)
c.Draw()
h.Draw("colz")
c.Update()

# clean up
io.finalize()
raw_input()
