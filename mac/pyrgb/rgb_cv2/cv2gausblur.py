from cv2selection import CV2Selection
import cv2

class CV2GausBlur(CV2Selection):
    def __init__(self):
    	super(CV2GausBlur,self).__init__()
        
    	self.name = "CV2GausBlur"

    	# default options
    	self.options['ksize']  = (5,5)
    	self.options['sigmaX'] = 1.0
        self.options['sigmaY'] = 1.0

        self.types['ksize']  = tuple
        self.types['sigmaX'] = float
        self.types['sigmaY'] = float
        
    def __description__(self):
    	return "No description provided!"

    def __implement__(self,image):
	return cv2.GaussianBlur(image,**self.options)
