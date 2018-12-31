from __future__ import print_function
import sys,os
import ROOT as rt
from larlite import larlite
from larcv import larcv

# Test function opens larcv and larlite file


larcv_input   = sys.argv[1]
larlite_input = sys.argv[2]

iolarcv = larcv.IOManager( larcv.IOManager.kREAD )
iolarcv.add_in_file( larcv_input )
iolarcv.initialize()

out = larcv.IOManager( larcv.IOManager.kWRITE )
out.set_out_file( "out_test.root" )
out.initialize()

cfg  = larcv.CreatePSetFromFile( "test_dlcosmicutil.fcl", "test" )
pset = cfg.get_pset("TestDLCosmicUtil")
print ("--- PSET ---------------")
print (pset.dump())
print ("------------------------")

algo = larcv.DLCosmicTagUtil()
algo.Configure(pset)
algo.set_verbosity(0)

nentries = iolarcv.get_n_entries()

logger = larcv.logger.get_shared()

for ientry in xrange(nentries):

    msg = "Entry {}\n".format(ientry)
    logger.send(2,"Event Loop").write( msg, len(msg) )
    
    iolarcv.read_entry(ientry)
    algo.go_to_entry(ientry)

    

    # whole adc larcv image_v
    event_img_v  = iolarcv.get_data( larcv.kProductImage2D, "wire" )
    img_v = event_img_v.Image2DArray()

    masked = algo.makeCosmicMaskedImage( img_v )

    # output images
    evout_wire   = out.get_data( larcv.kProductImage2D, "wire" )
    evout_masked = out.get_data( larcv.kProductImage2D, "intimemask" )

    for iimg in xrange( img_v.size() ):
        evout_wire.Append(   img_v.at(iimg) )
        evout_masked.Append( masked.at(iimg) )

    out.set_id( iolarcv.event_id().run(), iolarcv.event_id().subrun(), iolarcv.event_id().event() )
    out.save_entry()


out.finalize()
iolarcv.finalize()
print ("FIN")
raw_input()
