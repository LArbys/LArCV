from plotimage import PlotImage
from .. import np
from .. import pg
from .. import QtGui

class Ch12Image(PlotImage):

    def __init__(self,img_v,roi_v,planes) :

        self.temp_window = pg.GraphicsWindow()
        self.temp_window.resize(50,25)
        self.layout = QtGui.QGridLayout()
        self.label = QtGui.QLabel("12 Channel")
        self.channel = QtGui.QLineEdit("Slice (0-4)")
        self.temp_window.setLayout(self.layout)

        self.lay_inputs = QtGui.QGridLayout()
        self.layout.addLayout( self.lay_inputs,0,0 )

        self.lay_inputs.addWidget(self.label,0,0)
        self.lay_inputs.addWidget(self.channel,1,0)

        self.name = "Ch12Image"

        super(Ch12Image,self).__init__(img_v,roi_v,planes)

        #image will probably be 12 channels under a single product name



    def __create_mat__(self):

        print "\t>>temporary hack for taritree"
        split = [ [self.img_v[i+j] for i in xrange(0,12,4)] for j in xrange(4) ]

        #compressed images all have the same shape
        self.orig_mat = np.zeros(list(self.img_v[0].shape) + [3])

        try: 
            ch = int( str(self.channel.text()) )
        except ValueError:
            print "\t>> Channel no good defaulting to zero!"
            ch = 0

        for ix,img in enumerate(split[ch]):
            
            if ix not in self.planes: continue
            
            self.orig_mat[:,:,ix] = img
            
        self.plot_mat = self.orig_mat.copy()

        self.plot_mat = self.plot_mat[:,::-1,:]
        
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

