from cv2selection import CV2Selection
import cv2

class CV2Blur(CV2Selection):
    def __init__(self):
    	super(CV2Blur,self).__init__()
        
    	self.name = "CV2Blur"

    	# default options
    	self.options['ksize']  = (5,5)
    	self.options['anchor'] = (-1,-1)    	 

    def __description__(self):
    	return "No description provided!"

    def __implement__(self,image):
	return cv2.blur(image,**self.options)
