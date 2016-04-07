from plotimage import PlotImage

from .. import np

class CompressedImage(PlotImage):

    def __init__(self,img_v) :
        super(CompressedImage,self).__init__(img_v)
        self.name = "CompressedImage"
    
    def __create_mat__(self):
        #compressed images all have the same shape
        self.plot_mat = np.zeros(list(self.img_v[0].shape) + [3])

        for ix,img in enumerate(self.img_v):

            img = img[:,::-1]
            
            self.plot_mat[:,:,ix] = img


        # don't want shit to overlap on the viewer
        self.plot_mat[:,:,0][ self.plot_mat[:,:,1] > 0.0 ] = 0.0
        self.plot_mat[:,:,0][ self.plot_mat[:,:,2] > 0.0 ] = 0.0
        
        self.plot_mat[:,:,1][ self.plot_mat[:,:,2] > 0.0 ] = 0.0

        
    def __threshold_mat__(self,imin,imax):

        #Have to profile this copy operation, could be bad
        self.plot_mat_t = self.plot_mat.copy()

        #I don't know how to slice
        self.plot_mat_t[ self.plot_mat_t < imin ] = 0
        self.plot_mat_t[ self.plot_mat_t > imax ] = imax
            
        
        
        
        
        
        
