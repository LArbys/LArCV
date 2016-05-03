from cv2selection import CV2Selection
import cv2

class CV2Nothing(CV2Selection):
    def __init__(self):
    	super(CV2Nothing,self).__init__()
        
    	self.name = "CV2Nothing"


    def __description__(self):
    	return "No description provided!"

    def __implement__(self,image):
	return image
