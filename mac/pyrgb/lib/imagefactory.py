from image_types.defaultimage   import DefaultImage
from image_types.fakecolorimage import FakeColorImage
from image_types.ch12image      import Ch12Image

class ImageFactory(object):
    def __init__(self):
        self.name = "ImageFactory"

    def get(self,imdata,roidata,planes,improd,**kwargs):
        print "get: ",improd
        if improd == "fake_color": 
            return FakeColorImage(imdata,roidata,planes)

        if improd in ["tpc_12ch","tpc_12ch_mean"]:   
            print "12 ch data"
            return Ch12Image(imdata,roidata,planes)
        
        return DefaultImage(imdata,roidata,planes)
