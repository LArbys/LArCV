from __future__ import print_function
import sys,os
import ROOT as rt
from larlite import larlite
from larcv import larcv

# Test function opens larcv and larlite file


larcv_input   = sys.argv[1]

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
#for ientry in xrange(1,2):

    msg = "Entry {}\n".format(ientry)
    logger.send(2,"Event Loop").write( msg, len(msg) )
    
    iolarcv.read_entry(ientry)
    algo.goto_entry(ientry)

    
    # whole adc larcv image_v
    event_img_v  = iolarcv.get_data( larcv.kProductImage2D, "wire" )
    img_v = event_img_v.Image2DArray()

    if algo.numClusters()>0:
        crops = algo.makeClusterCrops( 0, img_v )
        wholeshower_v = algo.getWholeViewDLOutputImage( 0 )
        wholetrack_v  = algo.getWholeViewDLOutputImage( 1 )
        wholeendpt_v  = algo.getWholeViewDLOutputImage( 2 )        
    else:
        crops = None

    # output images for first cluster
    evout_wire   = out.get_data( larcv.kProductImage2D, "wire" )
    evout_masked = out.get_data( larcv.kProductImage2D, "intimemask" )
    evout_shower = out.get_data( larcv.kProductImage2D, "shower" )
    evout_track  = out.get_data( larcv.kProductImage2D, "track" )
    evout_endpt  = out.get_data( larcv.kProductImage2D, "endpt" )    
    evout_infill = out.get_data( larcv.kProductImage2D, "infill" )

    evout_shower_whole = out.get_data( larcv.kProductImage2D, "wholeshower" )
    evout_track_whole  = out.get_data( larcv.kProductImage2D, "wholetrack" )
    evout_endpt_whole  = out.get_data( larcv.kProductImage2D, "wholeendpt" )        

    if crops is not None:
        for iimg in xrange( img_v.size() ):

            # crops
            evout_masked.Append( crops.clustermask_v.at(iimg) )
            evout_shower.Append( crops.ssnet_shower_v.at(iimg) )
            evout_track.Append(  crops.ssnet_track_v.at(iimg) )
            evout_endpt.Append(  crops.ssnet_endpt_v.at(iimg) )            
            evout_infill.Append(  crops.infill_v.at(iimg) )

            # whole-images
            evout_wire.Append(   img_v.at(iimg) )            
            evout_shower_whole.Append( wholeshower_v.at(iimg) )
            evout_track_whole.Append( wholetrack_v.at(iimg) )
            evout_endpt_whole.Append( wholeendpt_v.at(iimg) )            

    out.set_id( iolarcv.event_id().run(), iolarcv.event_id().subrun(), iolarcv.event_id().event() )
    out.save_entry()


out.finalize()
iolarcv.finalize()
print ("FIN")
raw_input()
