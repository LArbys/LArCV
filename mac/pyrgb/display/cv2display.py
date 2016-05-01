from .. import pg
from .. import QtGui, QtCore


class CV2Display(QtGui.QWidget):

    def __init__(self):
        super(CV2Display,self).__init__()
        self.enabled = False
        self.cv2 = None
        self.imi = None
	# this should save users from having to load
	# opencv if they don't want to...
    def enable(self):

    	if self.enabled == True:
    		return

    	import cv2
    	self.cv2 = cv2
        self.name  = "CV2Display"
        self.win = pg.GraphicsWindow()
        self.plt = self.win.addPlot()        
        self.layout  = QtGui.QGridLayout()  
        self.layout.addWidget( self.win, 0, 0, 1, 10 )     
        self.setLayout(self.layout)
       	
        ### Input Widgets
        ### Layouts
        self.lay_inputs = QtGui.QGridLayout()
        self.layout.addLayout( self.lay_inputs, 1, 0 )

        self.enabled = True

    def paint(self,sl): #sl == slice of pimg
        if self.enabled == False:
        	return 
        self.plt.clear()
        self.imi = pg.ImageItem()
        self.plt.addItem(self.imi)
        sl = self.cv2.blur(sl,(5,5))
        self.imi.setImage(sl)
