from cv2selection import CV2Selection
import cv2

class CV2CopyPaste(CV2Selection):
    def __init__(self):
    	super(CV2CopyPaste,self).__init__()
        
    	self.name = "CV2CopyPaste"
        # how to impletement this?
    	# default options
        self.copy = None
        
    def __description__(self):
    	return "No description provided!"

    def __implement__(self,image):
    	if self.paste == False: 
            self.copy = image.copy()
            return image

        #can paste multiple times   
        return self.copy

