import os,sys
from larcv import larcv

inputfile     = sys.argv[1]
imageproducer = sys.argv[2]
entry         = int(sys.argv[3])
if len(sys.argv)>=5 and sys.argv[4]=="tickbackward":
    tickbackward = True
else:
    tickbackward = False

if not tickbackward:
    io = larcv.IOManager( larcv.IOManager.kREAD )
else:
    io = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )

io.add_in_file( inputfile )
io.initialize()

# start test
io.read_entry(entry)
ev_data = io.get_data( larcv.kProductImage2D, imageproducer )

for iimg in xrange(ev_data.Image2DArray().size()):

    img2d = ev_data.Image2DArray().at(iimg)
    
    print "------------------------------------------"
    print "image2d -> pixel array conversion"
    print "  meta: ",img2d.meta().dump()
    pixlist = larcv.as_pixelarray( img2d, 10.0, larcv.msg.kDEBUG )
    print "  pixarray shape: ",pixlist.shape
    print "  pixarray: ",pixlist
