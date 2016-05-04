from .. import pg
from .. import QtGui, QtCore
from ..rgb_cv2.cv2selector import CV2Selector

class CV2Layout(QtGui.QWidget):

    def __init__(self):
        super(CV2Layout,self).__init__()
        self.cv2_inputs  = QtGui.QGridLayout()

        self.name  = "CV2Layout"  

        self.enabled = False
        self.cv2 = None
        self.imi = None
        self.overwrite = False
        self.transform = True
        self.option = QtGui.QLabel("<b>Function</b>")
        self.selector = CV2Selector()

        # fill out the combo box selector
        self.comboBoxSelector = QtGui.QComboBox()
        for selection in self.selector.selections: #its a dict
            self.comboBoxSelector.addItem(selection)
        
        self.setSelection()
        
        self.tf = QtGui.QCheckBox("Transform")
        self.tf.setChecked(True)

        self.tf.stateChanged.connect(self.setTransform)
        self.setTransform()
        
        self.ow = QtGui.QCheckBox("Overwrite")
        self.ow.setChecked(False)
        
        self.ow.stateChanged.connect(self.setOverwrite)

        self.setOverwrite()
        self.changed = False

        self.menu = {}
        self.setMenu()

        self.loaded = QtGui.QPushButton("Reload!")      
        self.loaded.clicked.connect(self.reLoad)

    def grid(self,enable):  

        if self.enabled == False: #don't load cv2 unless necessary
            import cv2
            self.cv2 = cv2

        if enable == True:
            self.enabled = True 
            self.cv2_inputs.addWidget(self.option,0,0)
            self.cv2_inputs.addWidget(self.comboBoxSelector,1,0)
            self.cv2_inputs.addWidget(self.tf, 0, 1 )
            self.cv2_inputs.addWidget(self.ow, 1, 1 )
            self.cv2_inputs.addWidget(self.loaded,2,0)

            return self.cv2_inputs

        for i in reversed(range(self.cv2_inputs.count())):
            self.cv2_inputs.itemAt(i).widget().setParent(None)
                
    #does shit or not to slice
    def setTransform(self):
        if self.tf.isChecked():
            self.transform = True
        else:
            self.transform = False


    #apply over and over?
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

        if len(self.menu) != 0:
            for item in self.menu:
                #explicitly get rid of these fuckers
                self.cv2_inputs.removeWidget(self.menu[item][0])
                self.cv2_inputs.removeWidget(self.menu[item][1])
                self.menu[item][0].setParent(None)
                self.menu[item][1].setParent(None)
                

        # now they are gone right...
        self.menu = {}
        c = 2        
        for op,ty in self.selector.selection.options.iteritems():
            l  = QtGui.QLabel(op)
            t  = self.selector.selection.types[op] 
            o  = self.selector.selection.set_type(ty,t)
            
            self.cv2_inputs.addWidget(l,0,c)
            self.cv2_inputs.addWidget(o,1,c)
            
            self.menu[op] = (l,o,t)
            
            c+=1

        self.reLoad()
            
    def setSelection(self):
        self.selector.select(str(self.comboBoxSelector.currentText()))
        self.selected = str(self.comboBoxSelector.currentText())
                            
    def paint(self,sl): #sl == slice of pimg
        
        # lets tell selector about our options
        if self.changed == True:
            for item in self.menu:
                it = self.menu[item]
                it = (str(it[1].text()),it[2])
                self.selector.selection.options[item] = self.selector.selection.convert(it)

        self.changed = False
            
        # get the selection, apply transformation
        # for now just one at a time
        sl = self.selector.apply(sl)

        return sl

