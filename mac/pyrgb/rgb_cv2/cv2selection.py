import abc
from .. import QtGui

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


    def set_type(self,option,ty):
        qte = QtGui.QLineEdit()
        qte.setFixedHeight(5*10)
        qte.setFixedWidth(75)

        text = ""

        if ty is tuple:
            text = "("
            for o in option:
                text += str(o) + ","

            text = text[:-1] + ")"
            qte.setText(text)
            return qte

        if ty is int:
            qte.setText(str(option))
            return qte

        if ty is float:
            qte.setText(str(option))
            return qte
            
        raise Exception("Please implement type: {} in set_type cv2selection".format(type(option)))

    def convert(self,tu):

        if tu[1] is tuple:
            t = tu[0][1:-1].split(",")
            return tuple([int(i) for i in t])

        if tu[1] is int:
            return int(tu[0])

        if tu[1] is float:
            return float(tu[0])

        if tu[1] is list:

            t = tu[0].split(",")

            #shave of front and back
            t = t[1:-1]

            #make it
            o = [float(i) for i in t]

            return o

