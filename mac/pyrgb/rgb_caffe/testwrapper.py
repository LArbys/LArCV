import os,sys
import numpy as np
import yaml
from ..lib.iomanager import IOManager
from .. import larcv
class TestWrapper(object):

    def __init__(self):
        self.name = "TestWrapper"

        self.config = None
        self.net    = None

        self.loaded  = False
        self.pimg    = None
        self.caffe   = None
        self.iom     = None
        
    def set_config(self,config):
        self.config = config
    
    def load(self):
        sys.path.insert(0,'/Users/vgenty/git/caffe/python')

        import caffe
        self.caffe = caffe
        self.caffe.set_mode_cpu()
        self.reload_config()

    def reload_config(self):
        with open(self.config, 'r') as f:
            self.config = yaml.load(f)

        self.__generate_model__()
        self.__create_net__()

    def __create_net__(self):
        assert self.config is not None        
        self.net = self.caffe.Net( self.config['tmpmodel'],
                                   self.config["pretrainedmodel"],
                                   self.caffe.TEST )
        
        
    def set_image(self,image):
        self.pimg = image

    def prep_image(self):
        assert self.pimg is not None
        
        im = self.pimg.astype(np.float32,copy=True)

        #load the mean_file:
        if self.iom is None:
            self.iom = IOManager([self.config['meanfile']])
            self.iom.read_entry(0)
            means  = self.iom.get_data(larcv.kProductImage2D,self.config['meanproducer'])
            self.mean_v = [ larcv.as_ndarray(img)[:,::-1] for img in means.Image2DArray() ]

        for ix,mean in enumerate(self.mean_v):
            assert mean.shape == im[:,:,ix].shape
            im[:,:,ix] -= mean
        
        im[ im < self.config['imin'] ] = self.config['imin']
        im[ im > self.config['imax'] ] = self.config['imax']
        
        return im
        
    def forward_result(self):

        if self.loaded == False: self.load()
        self.loaded = True

        blob = {'data' : None, 'label' : None}
        
        im = self.prep_image()
        
        blob['data'] = np.zeros((1, im.shape[0], im.shape[1], 3),dtype=np.float32)
        print blob['data'].shape

        blob['data'][0,:,:,:] = im

        channel_swap = (0, 3, 1, 2)
        blob['data'] = blob['data'].transpose(channel_swap)  

        print blob['data'].shape

        blob['label'] = np.zeros((1,),dtype=np.float32)
        
        self.net.blobs['data'].reshape(*(blob['data'].shape))
        self.net.blobs['label'].reshape(*(blob['label'].shape))
        
        forward_kwargs = {'data': blob['data'] ,'label': blob['label']}
        
        blobs_out = self.net.forward(**forward_kwargs)
        
        scores  =  self.net.blobs[ self.config['lastfc'] ].data
        softmax =  self.net.blobs[ self.config['loss']   ].data

        self.scores = scores
        print "Scores:  {}".format(scores)
        print "Softmax: {}".format(softmax)

        
    def __generate_model__(self):
        print self.pimg.shape
        td = ""
        td += "input: \"data\"\n"
        td += "input_shape: { dim: 1 dim: 3 dim: %s dim: %s } \n"%(self.pimg.shape[0],
                                                                   self.pimg.shape[1])
        td += "input: \"label\"\n"
        td += "input_shape: { dim: 1 }"
        
        proto = None
        with open(self.config['modelfile'],'r') as f:
            proto = f.read()
        
        proto = td + proto
        fout = open(self.config['tmpmodel'],'w+')
        fout.write(proto)
        fout.close()

