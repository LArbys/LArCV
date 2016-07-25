import os,sys
import numpy as np
from .. import pg
from .. import QtGui

class LabelingTool:
    """ Class holding labeling functions.
    This class is a state machine that modifies a ImageItem stored in a PlotItem.
    
    states:  
    uninitiated: not yet worked with an image.
    drawing: swaps ImageItem from PlotItem. The grabbed image is flattened, 
             and the user is allowed to label on top of it.
    display: swaps ImageItem with an image it creates from the labeling that has been done.
    inactive: means that the original imageitem has been returned once drawing or displaying over
    """
    
    # state enum
    kUNINIT = 0
    kDRAWING = 1
    kDISPLAY = 2
    kINACTIVE = 3
    
    def __init__(self,labels,color_codes):
        """
        labels: a list of strings. index of labels used in code.
        color_codes: dictionary mapping label names to color
        """
        self.state = LabelingTool.kUNINIT
        self.labels = labels
        self.colors = color_codes
        if "background" not in self.labels:
            raise ValueError("Labels in labeling tool must have 'background' label!")
            

    def goIntoLabelingMode(self, label, plotitem, imageitem, src_images ):
        """
        switches state from inactive to drawing
        plotitem: the plot item it is suppose to manipulate
        imageitem: the image item in the plot it is suppose to manipulate
        src_images: plot image object (see lib/image_types/plotimage.py)

        for the drawing mode, we can only use black-to-white images as far as I can tell.
        so we convert to grey-scale, marking previously labeled pixels (same as current) with white.
        confusing, but what can we do for now?

        """
        if self.state==LabelingTool.kUNINIT:
            self.initialize(src_images)

        if self.state not in [LabelingTool.kINACTIVE,LabelTool.kDISPLAY]:
            # can transition into drawing mode only from the above states
            print "Moving to drawing state must be from inactive or display"
            return
            
        if self.shape()!=src_images.plot_mat.shape[:2]:
            raise ValueError("Image shape passed into LabelingTool is not the same shape as before")

        if label not in self.labels:
            raise ValueError("Label not recognized")


        # first keep pointer to data from the image item
        self.orig_img = src_images
        
        # for pixels we've already labeled as 'label', we put them into the image
        # we use a trick. the reduce color pixels by 10
        # this way labeled values can be distinguished later
        working = np.copy( self.orig_img.plot_mat )
        fmax = np.max( working )*1.1
        self.current_idx_label = self.labels.index( label )
        if label!="background":
            working[ self.labelmat==self.current_idx_label ] = fmax
        else:
            working[ self.labelmat!=self.current_idx_label ] = fmax
        working /= fmax # normalize

        # now set the image
        imageitem.setImage( image=working, levels=[[0.0,1.0],[0.0,1.0],[0.0,1.0]] )

        # and turn on the drawing state:
        #kern = np.array( [[[ self.colors[label][0], self.colors[label][1], self.colors[label][2] ]]] )
        if label!="background":
            kern = np.array( [[[ 1,1,1 ]]] )
            imageitem.setDrawKernel(kern, mask=kern, center=(0,0), mode='add')
        else:
            kern = np.array( [[[ 0,0,0]]] )
            imageitem.setDrawKernel(kern, mask=kern, center=(0,0), mode='set')

        self.current_img = imageitem.image
        self.state = LabelingTool.kDRAWING

    def initialize(self, src_images ):
        """ we setup the numpy array to store the labels now that we know image shape we are dealing with."""
        labelmatshape = src_images.plot_mat.shape[:2]
        self.labelmat = np.ones( labelmatshape, dtype=np.int )*self.labels.index("background")
        self.state = LabelingTool.kINACTIVE

    def shape( self ):
        if self.state==LabelingTool.kUNINIT:
            return None
        return self.labelmat.shape

    def drawLabeling(self,plotitem,imageitem,src_images):
        """ put labeling colors onto current image """
        if self.state==LabelingTool.kDRAWING:
            # transitioning from drawing state, save labeling
            self.saveLabeling()

        # set state to display
        self.state=LabelingTool.kDISPLAY
        self.orig_img = src_images
        self.imageitem = imageitem
        working = np.copy( self.orig_img.plot_mat )
        fmax = np.max( working )*1.1
        working /= fmax # normalize

        for idx,label in enumerate(self.labels):
            if label=="background":
                continue
            color = np.array( self.colors[label] )/255.0
            working[ self.labelmat==idx ] = color

        self.imageitem.setImage( image=working, levels=[[0.0,1.0],[0.0,1.0],[0.0,1.0]] )
        self.imageitem.setDrawKernel(None) # turn off draw mode?

    def saveLabeling(self):
        """ take current image, find labeled pixels, store them in self.labelmat """
        if self.state!=LabelingTool.kDRAWING:
            print "Label Tool not in Drawing state"
            return
        # use current img
        print "Saving labels"
        marked = (self.current_img >= 1.0).all(axis=2)
        print marked.shape
        print np.count_nonzero( marked )
        self.labelmat[ marked ] = self.current_idx_label
        return
    
    def undo(self):
        """ restore current image, losing latest labeling """
        working = np.copy( self.orig_img.plot_mat )
        fmax = np.max( working )*1.1
        if self.labels[self.current_idx_label]!="background":
            working[ self.labelmat==self.current_idx_label ] = fmax
        else:
            working[ self.labelmat!=self.current_idx_label ] = fmax
        working /= fmax # normalize

        # now set the image
        self.imageitem.setImage( image=working, levels=[[0.0,1.0],[0.0,1.0],[0.0,1.0]] )
        
    def switchLabel(self, newlabel):
        """ switch within drawing mode, the label used """
        if self.state!=LabelingTool.kDRAWING:
            return

        self.saveLabeling()
        self.current_idx_label = self.labels.index( newlabel )
        if newlabel!="background":
            kern = np.array( [[[ 1,1,1 ]]] )
            imageitem.setDrawKernel(kern, mask=kern, center=(0,0), mode='add')
        else:
            kern = np.array( [[[ 0,0,0]]] )
            imageitem.setDrawKernel(kern, mask=kern, center=(0,0), mode='set')            
        self.undo() # redraws (but now with new label highlighted using current_idx_label)
        

    def deactivateLabelMode(self):
        """ deactivate label mode """
        self.state = LabelingTool.kINACTIVE
        # restore image
        self.imageitem.setImage( image=self.orig_img.plot_mat, autoLevels=True )
        return

