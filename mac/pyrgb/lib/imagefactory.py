from image_types.defaultimage   import DefaultImage
from image_types.fakecolorimage import FakeColorImage

class ImageFactory(object):
    def __init__(self):
        self.name = "AhoFactory"

    def get(self,imdata,roidata,planes,improd):

        if improd == "fake_color": return FakeColorImage(imdata,roidata,planes)


        return DefaultImage(imdata,roidata,planes)
