from plotimage import PlotImage
from .. import np


class Ch12Image(PlotImage):
    # notes
    # orig_mat is the ndarray representation of image2d data
    # plot_mat is modified to have overlays
    # when loading, we manip orig_mat to have the orientation we want
    # before going to caffe, __revert_image__ is called to rearrange the image

    def __init__(self, img_v, roi_v, planes):
        super(Ch12Image, self).__init__(img_v, roi_v, planes)
        self.name = "Ch12Image"

    def __create_mat__(self):

        # this 12 ch data (from the lmdb) was made before the current convention
        # thus it stores its data in time order (rather than reverse time order)
        # it also has transposed the matrix
        # to make sure the final orientation that is sent to caffe 
        # is correct while also keeping to this display's conventions
        #  we 1) do not time reverse it, 2) then transpose
        # tmw has checked that this is correct
        
        # working copy
        if not hasattr(self,'work_mat'):
            self.work_mat = np.zeros(list(self.img_v[0].shape)+[len(self.img_v)])

        #compressed images all have the same shape
        self.orig_mat = np.zeros(list(self.img_v[0].shape) + [3])

        for ch in range(0,len(self.img_v)):
            self.work_mat[:,:,ch] = self.img_v[ch]
            if ch in self.planes:
                self.orig_mat[:, :, self.planes.index(ch)] = self.img_v[ch]
        for p,fill_ch in enumerate(self.planes):
            self.idx[fill_ch] = p

        #self.orig_mat = self.orig_mat[:, ::-1, :]
        self.work_mat = np.transpose( self.work_mat, (1,0,2) )
        self.orig_mat = np.transpose( self.orig_mat, (1,0,2) )


    def __swap_mat_channels__(self,imin,imax,newchs):
        print "swap channels to: ",newchs
        # store the current state of the orig_mat into the working matrix
        for p,ch in enumerate(self.planes):
            if ch!=-1:
                self.work_mat[:,:,ch] = self.orig_mat[:,:,p] # don't put a blank in there
        # swap the planes
        self.planes = newchs
        # fill idx, which is needed to put orig_mat back into img_v when we go to the network
        for p,ch in enumerate(self.planes):
            self.idx[ch] = p
        # put work mat values into orig_mat
        for p,ch in enumerate(self.planes):
            if ch!=-1:
                self.orig_mat[:,:,p] = self.work_mat[:,:,ch]
            else:
                self.orig_mat[:,:,p] = np.zeros( (self.orig_mat.shape[0],self.orig_mat.shape[1] ) )
        # make the viewing plot_mat and return
        return self.__set_plot_mat__(imin,imax)

    def __set_plot_mat__(self, imin, imax):

        
        self.plot_mat = self.orig_mat.copy()

        # do contrast thresholding
        self.plot_mat[ self.plot_mat < imin ] = 0
        self.plot_mat[ self.plot_mat > imax ] = imax

        # make sure pixels do not block each other
        self.plot_mat[:,:,0][ self.plot_mat[:,:,1] > 0.0 ] = 0.0
        self.plot_mat[:,:,0][ self.plot_mat[:,:,2] > 0.0 ] = 0.0
        self.plot_mat[:,:,1][ self.plot_mat[:,:,2] > 0.0 ] = 0.0

        return self.plot_mat

    # revert back to how image was in ROOTFILE for caffe...
    def __revert_image__(self): 
        self.orig_mat = np.transpose( self.orig_mat, (1,0,2) )
        self.work_mat = np.transpose( self.work_mat, (1,0,2) )

    def __create_rois__(self):
        
        for ix,roi in enumerate(self.roi_v) :

            nbb = roi.BB().size()
            
            if nbb == 0: #there was no ROI continue...
                continue

            r = {}

            r['type'] = roi.Type()
            r['bbox'] = []

            for iy in xrange(nbb):
                bb = roi.BB()[iy]
                r['bbox'].append(bb)
                
            self.rois.append(r)

