from image_types.defaultimage   import DefaultImage
from image_types.fakecolorimage import FakeColorImage
from image_types.ch12image      import Ch12Image

class ImageFactory(object):
    def __init__(self):
        self.name = "ImageFactory"

    def get(self,imdata,roidata,planes,improd,**kwargs):

        if improd == "fake_color": 
        	return FakeColorImage(imdata,roidata,planes)

        if improd == "tpc_12ch":   
        	return Ch12Image(imdata,roidata,planes)
        
        return DefaultImage(imdata,roidata,planes)
