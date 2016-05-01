import os,sys
import numpy as np
import yaml
sys.path.insert(0,'/Users/vgenty/git/caffe/python')
import caffe

class TestWrapper(object):

    def __init__(self):
        self.name = "TestWrapper"
        self.config = "/Users/vgenty/git/LArCV/mac/config.yml"
        self.modelfile = None
        self.net = None
        self.loaded = False
        self.pimg = None

    def load(self):
        caffe.set_mode_cpu()
        self.reload_config(self.config)
        #caffe.set_device(0)

    def reload_config(self,config):
        with open(config, 'r') as f:
            self.config = yaml.load(f)
        print self.config
        self.__generate_model__()
        self.__create_net__()

    def __create_net__(self):
        assert self.config is not None        
        self.net = caffe.Net( self.config['tmpmodel'],
                              self.config["pretrainedmodel"],
                              caffe.TEST )
        
        
    def set_image(self,image):
        self.pimg = image

        
    def forward_result(self):

        if self.loaded == False:
            self.load()

        im = self.pimg.astype(np.float32,copy=True)
        blob = {'data' : None, 'label' : None}

        print "Not subtracting pixel means!"
        #im_orig -= cfg.PIXEL_MEANS
        
        blob['data'] = np.zeros((1, im.shape[0], im.shape[1], 3),dtype=np.float32)
        print blob['data'].shape
        print im.shape
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

