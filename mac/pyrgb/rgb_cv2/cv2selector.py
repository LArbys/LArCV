from cv2blur import CV2Blur
from cv2nothing import CV2Nothing

class CV2Selector(object):
        def __init__(self):
		self.selections = { "nothing" : CV2Nothing(),
                                    "blur"    : CV2Blur() }
                
                self.selection = None

	def select(self,name):
		assert name in self.selections.keys()
		self.selection = self.selections[name]

	def apply(self,image):
		return self.selection.apply(image)
