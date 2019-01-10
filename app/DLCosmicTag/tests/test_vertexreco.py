from __future__ import print_function
import sys,os
import ROOT as rt
from larlite import larlite
from larcv import larcv
larcv.PSet # why this is needed I still don't understand

# This script is used to run and test the DLCosmicVertexReco
larcv_input   = sys.argv[1]

larcv_input_v = rt.std.vector("string")()
larcv_input_v.push_back( larcv_input )

cfg = "prod_dlcosmictag_vertexreco.cfg"

pset = larcv.CreatePSetFromFile(cfg);
proccfg = pset.get( "ProcessDriver" )
print(proccfg.dump())

#proc = larcv.ProcessDriver("ProcessDriverCosmicTagVertexReco")
proc = larcv.ProcessDriver("ProcessDriver")
#proc.configure( proccfg )
proc.configure( cfg )
proc.override_input_file( larcv_input_v )
proc.initialize()


nentries = proc.io.get_n_entries()

logger = larcv.logger.get_shared()

msg = "Number of events: {}".format(nentries)
logger.send(2,"Setup").write( msg, len(msg) )

sys.exit(-1)

#for ientry in xrange(nentries):
for ientry in xrange(1,2):

    msg = "Entry {}\n".format(ientry)
    logger.send(2,"Event Loop").write( msg, len(msg) )
    
    iolarcv.read_entry(ientry)
    algo.goto_entry(ientry)

    
    # whole adc larcv image_v
    event_img_v  = iolarcv.get_data( larcv.kProductImage2D, "wire" )
    img_v = event_img_v.Image2DArray()

    if algo.getNumClusters()>0:
        crops = algo.makeClusterCrops( 0, img_v )
        wholeshower_v = algo.getWholeViewDLOutputImage( 0 )
    else:
        crops = None

    # output images for first cluster
    evout_wire   = out.get_data( larcv.kProductImage2D, "wire" )
    evout_masked = out.get_data( larcv.kProductImage2D, "intimemask" )
    evout_shower = out.get_data( larcv.kProductImage2D, "shower" )
    evout_track  = out.get_data( larcv.kProductImage2D, "track" )
    evout_infill = out.get_data( larcv.kProductImage2D, "infill" )

    evout_shower_whole = out.get_data( larcv.kProductImage2D, "wholeshower" )

    if crops is not None:
        for iimg in xrange( img_v.size() ):
            evout_wire.Append(   img_v.at(iimg) )
            evout_masked.Append( crops.clustermask_v.at(iimg) )
            evout_shower.Append( crops.ssnet_shower_v.at(iimg) )
            evout_track.Append(  crops.ssnet_track_v.at(iimg) )
            evout_infill.Append(  crops.infill_v.at(iimg) )            
            evout_shower_whole.Append( wholeshower_v.at(iimg) )

    out.set_id( iolarcv.event_id().run(), iolarcv.event_id().subrun(), iolarcv.event_id().event() )
    out.save_entry()


out.finalize()
iolarcv.finalize()
print ("FIN")
raw_input()
