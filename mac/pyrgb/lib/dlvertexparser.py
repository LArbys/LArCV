import os,sys

import ROOT as rt
from ROOT import std
from larcv import larcv
from larlite import larutil
import numpy as np
import pyqtgraph

class DLVertexParser:
    """
    purpose of this class is to take an iomanager object 
    which container the output
    of the DL LEE vertexer and returns numpy arrays
    """

    def init(self):
        pass

    def getVertexPlotItems(self, vtx_idx,iom,
                           tickforward=True,
                           producername="test"):
        """
        inputs
        ------
        vtx_idx: index of the vertex to plot
        iom: IOManager class
        producername: stem name of the vertex tree objects

        you can find the IOManager instance in the DataManager class.
        In the main display class, the iomanager can be retrived via: self.dm.iom

        we expect the following trees in the ROOT file (managed by IOManager).
        assuming stem name 'test'

        KEY: TTree	pgraph_test_tree;1	test tree
        KEY: TTree	pixel2d_test_ctor_tree;1	test_ctor tree
        KEY: TTree	pixel2d_test_img_tree;1	test_img tree
        KEY: TTree	pixel2d_test_super_ctor_tree;1	test_super_ctor tree
        KEY: TTree	pixel2d_test_super_img_tree;1	test_super_img tree
        KEY: TTree	pixel2d_dlshr_tree;1	dlshr tree

        """
        ev_pgraph    = iom.get_data( larcv.kProductPGraph,  producername )
        ev_super_ctr = iom.get_data( larcv.kProductPixel2D, producername+"_ctor" )
        pgraph_v  = ev_pgraph.PGraphArray()
        #print "Num pgraphs for the event: ",pgraph_v.size()
        if vtx_idx<0 or vtx_idx>=pgraph_v.size():
            return
        
        pgraph = pgraph_v.at(vtx_idx)
        #print "pgraph[{}]: num particle rois={} num clusters={}".format(vtx_idx,
        #                                                                pgraph.ParticleArray().size(),
        #                                                                pgraph.ClusterIndexArray().size())

        plot_dict = {0:[],
                     1:[],
                     2:[],
                     "vertex":{0:None,1:None,2:None}}
        pencolors = [(255,0,0),
                     (0,255,0),
                     (0,0,255)]

        brush_colors = [(100,100,200),
                        (200,100,100)]

        vertex_brush = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
        #print vertex_brush
        
        for iroi in xrange(pgraph.ParticleArray().size()):
            roi   = pgraph.ParticleArray().at(iroi)
            clustindex = pgraph.ClusterIndexArray().at(iroi)            
            #print "  part[{}] vertex=({},{},{}) cluster-index={}".format(iroi,roi.X(),roi.Y(),roi.Z(),clustindex)

            vertex = std.vector("double")(3,0)
            vertex[0] = roi.X()
            vertex[1] = roi.Y()
            vertex[2] = roi.Z()

            vertex_tick = 3200 + roi.X()/larutil.LArProperties.GetME().DriftVelocity()/0.5
            vertex_row  = (vertex_tick-2400)/6

            n_super_ctrs = ev_super_ctr.Pixel2DClusterArray().size()
            for iplane in xrange(n_super_ctrs):
                if iplane>=3:
                    continue

                wire = larutil.Geometry.GetME().NearestWire( vertex, iplane )

                if plot_dict['vertex'][iplane] is None:
                    vertex_np = np.zeros( (1,2) )
                    vertex_np[0,0] = wire
                    vertex_np[0,1] = vertex_row
                    if iplane in [0,1,2]:
                        pencolor = pencolors[iplane]
                    plot_dict["vertex"][iplane] = pyqtgraph.ScatterPlotItem( pos=vertex_np, pxMode=True, size=10,
                                                                             pen=pencolor, symbol='o', brush=vertex_brush )
                                
                pencolor = pencolors[iplane]
                if True:
                    cluster_v    = ev_super_ctr.Pixel2DClusterArray(iplane)
                    meta_v       = ev_super_ctr.ClusterMetaArray(iplane)
                    for icluster in xrange(cluster_v.size()):
                        cluster = cluster_v.at(icluster)
                        meta    = meta_v.at(icluster)

                        if not tickforward:
                            meta_tick = meta.min_y()-meta.height()
                        
                        pixdata_np  = np.zeros( (cluster.size()+1,2) )
                        #print "Number of pixels in cluster[{}]: {}".format(icluster,cluster.size())
                        #print " meta: ",meta.dump()
                        #print " meta_tick: ",meta_tick
                        for ipix in xrange(cluster.size()):
                            pix2d = cluster.at(ipix)
                            pixdata_np[ipix,0] = meta.min_x()+pix2d.X()
                            if tickforward:
                                pixdata_np[ipix,1] = meta.min_y()/6 + pix2d.Y()
                            else:
                                pixdata_np[ipix,1] = (meta_tick-2400)/6 + meta.height()/6 - pix2d.Y() # needed for tick-backward era objects
                            if ipix==0:
                                pixdata_np[cluster.size(),0] = pixdata_np[ipix,0]
                                pixdata_np[cluster.size(),1] = pixdata_np[ipix,1]
                            #print " pix[{}] ({},{})".format(ipix,pix2d.X(),pix2d.Y())

                        if roi.Shape()==2:
                            pen = pyqtgraph.mkPen( width=2, color=pencolor )
                        elif roi.Shape()==1:
                            pen = pyqtgraph.mkPen( width=2, color=pencolor, style=QtCore.Qt.DashLine )
                        else:
                            pen = pyqtgraph.mkPen( width=2, color=(255,255,255) )
                        pix_plot = pyqtgraph.PlotCurveItem( x=pixdata_np[:,0], y=pixdata_np[:,1], pen=pen )
                                                            
                        #if icluster<2:
                        #    pix_plot.setBrush( color=brush_colors[icluster] )
                        #else:
                        #    pix_plot.setBrush( color=(125,125,12) )
                        plot_dict[iplane].append(pix_plot)
                    
                else:
                    continue
        #print "return DL vertexer plot items"
        return plot_dict
        


if __name__ == "__main__":

    larcvfile = sys.argv[1]
    iom = larcv.IOManager(larcv.IOManager.kREAD,"input",larcv.IOManager.kTickBackward)
    iom.add_in_file( larcvfile )
    iom.initialize()
    iom.read_entry(3)
    
    vertex = DLVertexParser()
    vertex.getVertexPlotItems(0,iom)
    
