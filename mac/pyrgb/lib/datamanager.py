from __future__ import print_function
import os,sys
import numpy as np
import time, re

from .. import ROOT
from .. import larcv

from .iomanager import IOManager
from .imagefactory import ImageFactory

# data manger helps get the producers from the ROOT file
# as well as manage factory creation of images as the user
# hits replot, next/prev event

class DataManager(object):

    __handled_larlite_types = [] # non right now
    
    def __init__(self,argv):
        """
        look into file contents, determine if larlite/larcv and register products
        """
        input_list = []
        for arg in argv:
            if "--" not in arg:
                if os.path.exists(arg):
                    input_list.append(arg)
                else:
                    print("Could not load input file: ",arg)


        # support for old larcv1 files
        tick_forward=True
        if "--tickbackward" in argv:
            tick_forward=False

        # prepare dictionary holding product names (keys) and producer names of that type (values)
        self.keys = {}
        for i in range(larcv.kProductUnknown):
            product = str(larcv.ProductName(i))
            self.keys[product] = []
        
        self.logger = larcv.logger("pyrgb::DataManager")
        self.larcv_list = []
        self.larlite_list = []

        for infile in input_list:
            islarcv   = False
            islarlite = False
            try:
                self.logger.send(larcv.msg.kINFO,"datamanager").write( "keys inside {}\n".format(infile),
                                                                       len("keys inside {}\n".format(infile) ) )
                rfile = ROOT.TFile(infile)
                keys = rfile.GetListOfKeys()
                for ikey in range(keys.GetEntries()):
                    keyname = str(keys.At(ikey).GetName())
                    prodname = keyname.split("_")[0]
                    if prodname == "partroi":
                        typename = "roi" # hack need for (maybe) bug in larcv
                    else:
                        typename = prodname
                    lcv_prodid = larcv.GetProductTypeID(typename)
                    fwtype = '[unrecognized]'
                    if lcv_prodid!=larcv.kProductUnknown:
                        # recognized larcv product
                        fwtype = "[larcv]"
                        islarcv = True
                        self.keys[prodname].append(keyname[len(prodname)+1:-len("_tree")])
                    elif prodname in DataManager.__handled_larlite_types:
                        fwtype = "[larlite]"
                        islarlite = True

                    fwinfo = "  {} {} {}\n".format( ikey, keyname, fwtype)
                    self.logger.send(larcv.msg.kINFO,"datamanager").write( fwinfo, len(fwinfo) )

            except:
                msg = "keys inside {}\n".format(infile)
                errmsg = "  err={}".format(sys.exc_info()[0])
                self.logger.send(larcv.msg.kWARNING,"datamanager").write( msg, len(msg) )
                self.logger.send(larcv.msg.kWARNING,"datamanager").write( errmsg, len(errmsg) )
            if islarcv:
                self.larcv_list.append( infile )
            if islarlite:
                self.larlite_list.append( infile )

        # open the input IOManager
        self.iom = IOManager(self.larcv_list,tick_forward=tick_forward)
        self.iom.set_verbosity(1)

        self.IF = ImageFactory()
        
        # run subrun and event start at zero
        self.run    = -1
        self.subrun = -1
        self.event  = -1
        

    def get_nchannels(self,ii,imgprod) :
        # Sorry Vic I hacked this
        # --> it's ok
        self.iom.read_entry(ii)
        imdata  = self.iom.get_data(larcv.kProductImage2D,imgprod)
        return imdata.Image2DArray().size()

    def get_event_image(self,ii,imgprod,roiprod,planes, refresh=True) :

        #Load data in TChain
        self.iom.read_entry(ii)

        # there may be no ROI
        hasroi = False
        roidata = None
        if roiprod is not None:
            roidata = self.iom.iom.get_data(larcv.kProductROI,roiprod)
            roidata = roidata.ROIArray()
            hasroi = True

        # get the EventImage2D
        imdata  = self.iom.get_data(larcv.kProductImage2D,imgprod) # goes to disk

        self.run    = imdata.run()
        self.subrun = imdata.subrun()
        self.event  = imdata.event()

        # get the std::vector<larcv::Image2D>
        imdata  = imdata.Image2DArray()
        if imdata.size() == 0 : return (None, False)

        # hand it off to the factory, the producer name should query the
        # the correct subclass of PlotImage
        image = self.IF.get(imdata,roidata,planes,imgprod) # returns PlotImgae


        # return it to rgbviewer
        return ( image, hasroi )         
    
    
    
    # -----------------------------------------------------------------------------
    # Erez, July-21, 2016 - get an image using R/S/E navigation
    # -----------------------------------------------------------------------------
    def get_all_images(self,imgprod,event_base_and_images,rse_map) :
        
        for entry in range(self.iom.get_n_entries()):
            read_entry = self.iom.read_entry(entry)
            event_base = self.iom.get_data(larcv.kProductImage2D,imgprod)
            event_base_and_images[entry] = event_base
            rse = ( int(event_base.run()),int(event_base.subrun()),int(event_base.event()) )
            #print(rse
            #rse_map[entry] = [event_base.run(),event_base.subrun(),event_base.event()]
            rse_map[ rse ] = entry
#            print(rse_map[entry])
        print("collected %d images...\nready for RSE navigation"%len(event_base_and_images))

        return
    



    # -----------------------------------------------------------------------------
    # Erez, July-21, 2016 - get an image using R/S/E navigation
    # -----------------------------------------------------------------------------
    def get_rse_image(self,event_base_and_images,rse_map,wanted_rse,imgprod,roiprod,planes, refresh=True) :
        if wanted_rse in rse_map:
            return self.get_event_image(rse_map[wanted_rse],imgprod,roiprod,planes,refresh)
        else:
            print("i couldn't find this R/S/E...")
            return None, False
 
        ii = -1
        for i in range(len(event_base_and_images)):
            
            if rse_map[i] == wanted_rse:
                ii = i
                break
    
        if (ii==-1):
            print("i couldn't find this R/S/E...")
                        
        return self.get_event_image(ii,imgprod,roiprod,planes,refresh)

    # -----------------------------------------------------------------------------










