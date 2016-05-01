from plotimage import PlotImage
from .. import np

class DefaultImage(PlotImage):

    def __init__(self,img_v,roi_v,planes) :
        super(DefaultImage,self).__init__(img_v,roi_v,planes)
        self.name = "DefaultImage"
    
    def __create_mat__(self):

        #compressed images all have the same shape
        self.plot_mat = np.zeros(list(self.img_v[0].shape) + [3])

        for ix,img in enumerate(self.img_v):
            
            if ix not in self.planes:
                continue
            
            img = img[:,::-1]
            
            self.plot_mat[:,:,ix] = img
            
        self.orig_mat = self.plot_mat.copy()
        
        self.plot_mat[:,:,0][ self.plot_mat[:,:,1] > 0.0 ] = 0.0
        self.plot_mat[:,:,0][ self.plot_mat[:,:,2] > 0.0 ] = 0.0
        self.plot_mat[:,:,1][ self.plot_mat[:,:,2] > 0.0 ] = 0.0
        
    def __threshold_mat__(self,imin,imax):

        #Have to profile this copy operation, could be bad
        self.plot_mat_t = self.plot_mat.copy()

        #I don't know how to slice
        self.plot_mat_t[ self.plot_mat_t < imin ] = 0
        self.plot_mat_t[ self.plot_mat_t > imax ] = imax
        

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

