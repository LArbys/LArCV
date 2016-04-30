from plotimage import PlotImage
from larcv import larcv
from .. import np

class VicImage(PlotImage):

    def __init__(self,img_v,roi_v,planes) :
        super(VicImage,self).__init__(img_v,roi_v,planes)
        self.name = "VicImage"

      	ometa = None
    	for img in self.imgs:
    		if not ometa: ometa = larcv.ImageMeta(img.meta())
	    	else: ometa = ometa.inclusive(img.meta())

		tmp_img_v=[]
		for i in xrange(len(self.img_v)):
		    meta=larcv.ImageMeta( ometa.width(), ometa.height(), ometa.rows(), ometa.cols(), ometa.min_x(), ometa.max_y(), i)
		    img=larcv.Image2D(meta)
		    img.paint(0.)
		    img.overlay(self.imgs[i])
		    tmp_img_v.append(img)
                    
		self.imgs = tmp_img_v
		self.img_v  = [ larcv.as_ndarray(img) for img in self.imgs  ]

		for img in self.img_v: print img.shape

        self.__create_mat__()
        self.plot_mat_t = None

        
        self.rois = []
    
    def __create_mat__(self):

        #compressed images all have the same shape
        self.plot_mat = np.zeros(list(self.img_v[0].shape) + [3])

        for ix,img in enumerate(self.img_v):

            if ix not in self.planes:
                continue
            
            img = img[:,::-1]
            
            self.plot_mat[:,:,ix] = img


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
            
            if roi.BB().size() == 0: #there was no ROI continue...
                continue

            r = {}

            r['type'] = roi.Type()
            r['bbox'] = []

            for iy in xrange(3):
                r['bbox'].append( roi.BB(iy) )
                
            self.rois.append(r)

