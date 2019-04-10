import os,sys

from larlite import larlite
from larcv import larcv
import numpy as np



if __name__ == "__main__":
    # Start up IO Manager
    iomanager_in = larcv.IOManager(larcv.IOManager.kREAD)
    iomanager_in.add_in_file( "/media/disk1/jmills/mcc9jan_extbnb_data/larcv_wholeview_fffea264-968c-470e-9141-811b5f3d6dcd.root")
    iomanager_in.set_verbosity(larcv.msg.kDEBUG)
    iomanager_in.initialize()

    iomanager = larcv.IOManager(larcv.IOManager.kWRITE)
    iomanager.set_out_file("test.root")
    iomanager.set_verbosity(larcv.msg.kDEBUG)
    iomanager.initialize()
    treename = 'cluster_test'
    # print(type(larcv.kProductClusterMask))
    # print(type(treename))

    # Make an eventclustermask object
    ev_clustermasks = iomanager.\
                    get_data(larcv.kProductClusterMask,
                             treename)
    # print(type(ev_clustermasks))
    # Make clustermasks to stuff into Ev Clustermask
    masks_vv = ev_clustermasks.as_vector()
    print(type(masks_vv))
    planes =3
    nmasks = 10
    masks_vv.resize(planes)
    # get data

    ev_wholeview = iomanager_in.get_data(larcv.kProductImage2D,
                                          "wire")
    wholeview_v = ev_wholeview.Image2DArray()
    print(type(wholeview_v.at(0)))
    meta = wholeview_v.at(0).meta()
    print(len(masks_vv))
    print(type(masks_vv.at(0)))
    print(type(meta))
    meta.reset_origin(0,0)

    for plane in range(planes):
        print("Length of plane: ", plane , "  start  ", len(masks_vv.at(plane)))

        for mask in range(nmasks):
            box = np.array([0,5,10,15,4], np.float32)
            # sparse_mask = np.random.randint(10, size=(25, 2))
            # sparse_mask = np.as_type(np.float32)
            sparse_mask = np.random.randint(10, size=(5, 2)).astype(np.float32)
            m = larcv.as_clustermask(sparse_mask, box ,meta)
            masks_vv.at(plane).push_back(m)
        print("Length of plane: ", plane , "  final  ", len(ev_clustermasks.as_vector().at(plane)))
        # for x in range(len(masks_vv.at(plane).at(7)._box)):
        #     print(masks_vv.at(plane).at(7)._box[x])
        # for x in range(len(masks_vv.at(plane).at(7)._mask)):
        #     print(masks_vv.at(plane).at(7)._mask[x])

    # iomanager.set_id( self._inlarcv.event_id().run(),
    #                        self._inlarcv.event_id().subrun(),
    #                        self._inlarcv.event_id().event())

    iomanager.set_id( 15, 84, 1)
    iomanager.save_entry()
    iomanager_in.finalize()
    iomanager.finalize()


#end
