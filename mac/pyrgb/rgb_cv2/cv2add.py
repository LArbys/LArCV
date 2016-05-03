from cv2selection import CV2Selection

class CV2Add(CV2Selection):
    def __init__(self):
    	super(CV2Add,self).__init__()
        
    	self.name = "CV2Add"
        self.options['add'] = int(0)
        self.types['add'] = int

    def __description__(self):
    	return "No description provided!"
    
    def __implement__(self,image):
	return self.options['add'] + image
