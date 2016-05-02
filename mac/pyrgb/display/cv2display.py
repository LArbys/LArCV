from .. import pg
from .. import QtGui, QtCore
from ..rgb_cv2.cv2selector import CV2Selector

class CV2Display(QtGui.QWidget):

    def __init__(self):
        super(CV2Display,self).__init__()
        self.enabled = False
        self.cv2 = None
        self.imi = None
        self.overwrite = False
        self.transform = True

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

        self.option = QtGui.QLabel("<b>Function</b>")
        self.lay_inputs.addWidget(self.option,0,0)
        
        self.selector = CV2Selector()

        # fill out the combo box selector
        self.comboBoxSelector = QtGui.QComboBox()
        for selection in self.selector.selections: #its a dict
            self.comboBoxSelector.addItem(selection)

        self.lay_inputs.addWidget(self.comboBoxSelector,0,1)

        self.setSelection()
        
        self.tf = QtGui.QCheckBox("Transform")
        self.tf.setChecked(True)
        self.lay_inputs.addWidget( self.tf, 1, 0 )
        self.tf.stateChanged.connect(self.setTransform)
        self.setTransform()
        
        self.ow = QtGui.QCheckBox("Overwrite")
        self.ow.setChecked(False)
        self.lay_inputs.addWidget( self.ow, 1, 1 )
        self.ow.stateChanged.connect(self.setOverwrite)
        self.setOverwrite()
        
        # self.option = QtGui.QLabel("<b>Options</b>")
        # self.lay_inputs.addWidget(self.option,0,1)

        self.changed = False

        self.menu = {}
        self.setMenu()


        self.loaded = QtGui.QPushButton("Reload!")
        self.lay_inputs.addWidget(self.loaded,2,0)
        self.loaded.clicked.connect(self.reLoad)
            
        self.enabled = True


    def setTransform(self):
        if self.tf.isChecked():
            self.transform = True
        else:
            self.transform = False


    def setOverwrite(self):
        if self.ow.isChecked():
            self.overwrite = True
        else:
            self.overwrite = False
    
    
    def reLoad(self):

        self.changed = True
        print "reloaded... {} {}".format(str(self.comboBoxSelector.currentText()),
                                         self.selected)
        
        if str(self.comboBoxSelector.currentText()) != self.selected:
               self.setSelection()
               self.setMenu()

        
    def setMenu(self):

        c = 2
        if len(self.menu) != 0:
            for item in self.menu:
                #explicitly get rid of these fuckers
                self.lay_inputs.removeWidget(self.menu[item][0])
                self.lay_inputs.removeWidget(self.menu[item][1])
                self.menu[item][0].setParent(None)
                self.menu[item][1].setParent(None)
                

        # now they are gone right...
        self.menu = {}
        
        for op,ty in self.selector.selection.options.iteritems():
            l  = QtGui.QLabel(op)
            t  = self.selector.selection.types[op] 
            o  = self.set_type(ty,t)
            
            self.lay_inputs.addWidget(l,1,c)
            self.lay_inputs.addWidget(o,2,c)
            
            self.menu[op] = (l,o,t)
            
            c+=1

        self.reLoad()
            
    def setSelection(self):
        self.selector.select(str(self.comboBoxSelector.currentText()))
        self.selected = str(self.comboBoxSelector.currentText())
                            
    def paint(self,sl): #sl == slice of pimg
        if self.enabled == False:
            return 
        
        self.plt.clear()
        self.imi = pg.ImageItem()
        self.plt.addItem(self.imi)

        # lets tell selector about our options
        if self.changed == True:
            for item in self.menu:
                it = self.menu[item]
                it = (str(it[1].text()),it[2])
                self.selector.selection.options[item] = self.convert(it)

        self.changed = False
            
        # get the selection, apply transformation
        # for now just one at a time
        sl = self.selector.apply(sl)

        self.imi.setImage(sl)
        return sl

    def set_type(self,option,ty):
        qte = QtGui.QLineEdit()
        qte.setFixedHeight(5*10)
        qte.setFixedWidth(75)

        text = ""

        if ty is tuple:
            text = "("
            for o in option:
                text += str(o) + ","

            text = text[:-1]
            text += ")"
            qte.setText(text)
            return qte

        if ty is int:
            qte.setText(str(option))
            return qte

        if ty is float:
            qte.setText(str(option))
            return qte
            
        raise Exception("Please implement type: {} in set_type cv2display".format(type(option)))

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
