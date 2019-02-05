from __future__ import print_function
import os,sys
import ROOT as rt
import json,cStringIO
from larcv import larcv
larcv.json.load_jsonutils()

inputfile = sys.argv[1]
imagetree = sys.argv[2]
if len(sys.argv)==4:
    entry = int(sys.argv[3])
else:
    entry = 0

# read in forward-tick larcv1
io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( inputfile )

io.initialize()
io.set_verbosity(0)

io.read_entry(entry)
ev_img = io.get_data( larcv.kProductImage2D, imagetree )

if ev_img.Image2DArray().size()>0:
    img = ev_img.Image2DArray().front()
    print("image loaded. meta=",img.meta().dump())
else:
    print("No images in tree=%s and entry=%d"%(imagetree,entry))

nexpected_elems = 1008*3456
print("num expected elements: {}".format(nexpected_elems))
strimg = larcv.json.as_json_str( img )
jimg = json.load( cStringIO.StringIO(strimg) )
print(len(jimg["data"]))

bson = larcv.json.as_bson( img )
print(bson," size=",len(bson)/1.0e6,"MB")

roundtrip_img  = larcv.json.image2d_from_json_str( strimg )
roundtrip_bson = larcv.json.image2d_from_bson( bson )

rt.gStyle.SetOptStat(0)
larcv.load_rootutil()
horig = larcv.as_th2d( img, "orig" )
hrt   = larcv.as_th2d( roundtrip_img, "roundstrip" )
hdiff = horig.Clone("hdiff")
hdiff.Add( hrt, -1.0 )
hbson = larcv.as_th2d( roundtrip_bson, "bson" )

c = rt.TCanvas("c","c",1600,400)
c.Divide(4,1)
c.Draw()
c.cd(1)
horig.Draw("colz")
c.cd(2)
hrt.Draw("colz")
c.cd(3)
hdiff.Draw("colz")
c.cd(4)
hbson.Draw("colz")
c.Update()

raw_input()

io.finalize()
