from cv2blur import CV2Blur

class CV2Selector(object):
        def __init__(self):
		self.selections = { "blur" : CV2Blur() }
                self.selection = None

	def select(self,name):
		assert name in self.selections.keys()
		self.selection = self.selections[name]

	def apply(self,image):
		return self.selection.apply(image)
