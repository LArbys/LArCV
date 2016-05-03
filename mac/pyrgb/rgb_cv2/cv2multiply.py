from cv2selection import CV2Selection

class CV2Multiply(CV2Selection):
    def __init__(self):
    	super(CV2Multiply,self).__init__()
        
    	self.name = "CV2Multiply"
        self.options['multiply'] = float(1)
        self.types['multiply'] = float

    def __description__(self):
    	return "No description provided!"
    
    def __implement__(self,image):
	return self.options['multiply'] * image
