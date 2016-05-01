import os,sys
import numpy as np
import yaml

caffe_home='/home/vgenty/git/caffe/'
sys.path.insert(0, caffe_home + 'python')
import caffe

class TestWrapper(object):
    def __init__(self):
        self.name = "TestWrapper"

        caffe.set_mode_cpu()
        #caffe.set_device(0)

        self.config = None
        self.net = None
        
        #model_file = '/home/vgenty/caffe/trial/2_class_valid.prototxt'
        #pretrained_file = model


    def reload_config(self,config):
        with open(config, 'r') as f:
            self.config = yaml.load(f)

    def create_net(self):
        assert self.config is not None        
        self.net = caffe.Net( self.config["modelfile"],
                              self.config["pretrainedfile"],
                              caffe.TEST )

    def forward_result(self):
        self.net.forward()
        scores  =  net.blobs[ self.config['lastfc'] ].data
        softmax =  net.blobs[ self.config['loss']   ].data

        print "Scores: {}".format(scores)
        print "Softmax: {}".format(softmax)
