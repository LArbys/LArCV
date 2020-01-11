from __future__ import print_function

import os,sys
from math import fabs
import argparse

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv

def parse_args():
    # parse input arguments
    parser = argparse.ArgumentParser(description='ShowerMask')
    parser.add_argument('--reco2d')
    parser.add_argument('--ssnet')
    parser.add_argument('--vtx')
    parser.add_argument('--supera')

    return parser.parse_args();


def main():

    args = parse_args()
    print(args)

    larcvinputfiles = [args.supera,args.vtx]
    larcvforwardinputfiles = [args.ssnet]
    larliteinputfiles = [args.reco2d]


    io = larcv.IOManager(larcv.IOManager.kREAD, "IO" )
    for inputfile in larcvinputfiles:
        io.add_in_file( inputfile )
    io.initialize()

    ioforward = larcv.IOManager(larcv.IOManager.kREAD, "IO")
    for inputfile in larcvforwardinputfiles:
        ioforward.add_in_file( inputfile )
    ioforward.reverse_image2d_tree("uburn_plane0")
    ioforward.reverse_image2d_tree("uburn_plane1")
    ioforward.reverse_image2d_tree("uburn_plane2")
    ioforward.initialize()

    showermask = larcv.showermask.ShowerMask()

    # larlite io manager
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    for l in larliteinputfiles:
        ioll.add_in_filename( l )
        print("opened: ", l)
    ioll.open()

    outputdata = larlite.storage_manager( larlite.storage_manager.kWRITE );
    outputdata.set_out_filename( "testoutput.root" );
    outputdata.open();

    # ----make config file----
    showermask_cfg = """
    InputWireProducer: \"wiremc\"
    InputSSNetUProducer: \"uburn_plane0\"
    InputSSNetVProducer: \"uburn_plane1\"
    InputSSNetYProducer: \"uburn_plane2\"
    InputHitsProducer: \"gaushit\"
    InputVtxProducer: \"test\"
    """
    lfcfg = open("showermask.cfg",'w')
    print(showermask_cfg, file=lfcfg)
    lfcfg.close()
    lfpset = larcv.CreatePSetFromFile( "showermask.cfg", "ShowerMask" )

    # ----Algos-----
    showermask.configure(lfpset)
    print ("finished config")
    showermask.initialize()

    nentries = io.get_n_entries()
    print("Num Entries: ",nentries)

    for ientry in xrange(nentries):

        io.read_entry(ientry)
        ioll.go_to(ientry)
        ioforward.read_entry(ientry)
        print("ON ENTRY: ",ientry)
        showermask.process(io,ioll,ioforward,outputdata)
        outputdata.next_event()
        print(" ")
    #
    # ientry = 12
    # io.read_entry(ientry)
    # ioll.go_to(ientry)
    # ioforward.read_entry(ientry)
    # print("ON ENTRY: ",ientry)
    # showermask.process(io,ioll,ioforward)
    # print(" ")

    showermask.finalize()
    io.finalize()
    ioll.close()
    ioforward.finalize()
    outputdata.close()

    print("FIN")

if __name__ == '__main__':
    main()
