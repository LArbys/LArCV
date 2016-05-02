import abc

class CV2Selection(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
    	self.name = "CV2Selection"
    	self.options = {}
        self.types = {}

    def description(self):
    	return self.__description__()
    
    def set_options(self,options):
	for op in options:
	    assert op in self.options.keys()
	    self.options[op] = options[op]

    def apply(self,image):
	return self.__implement__(image)

    def dump(self):
	print "\t name: {}".format(self.name)
	print "\t options: {}".format(self.options)
		
    @abc.abstractmethod
    def __description__(self):
    	""" provide description for tooltip """

    @abc.abstractmethod
    def __implement__(self,image):
	""" implement cv2 function """
