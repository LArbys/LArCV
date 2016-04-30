from .. import QtGui, QtCore

class CaffeLayout(object):

    def __init__(self):
        self.name  = "CaffeLayout"
        
        self.caffe_inputs = QtGui.QGridLayout()

        self.open_deploy = QtGui.QPushButton("Open")
        self.line_deploy = QtGui.QLineEdit("path/to/deploy")
        self.open_deploy.clicked.connect(self.selectFile)


        self.forward = QtGui.QPushButton("Forward")
        self.forward.clicked.connect(self.network_forward)

    def selectFile(self):
        self.line_deploy.setText(QtGui.QFileDialog.getOpenFileName())
        
    def grid(self,enable):
        if enable == True:
            self.caffe_inputs.addWidget(self.line_deploy,0,0)
            self.caffe_inputs.addWidget(self.open_deploy,0,1)
            self.caffe_inputs.addWidget(self.forward,0,2)
                                        
        else:
            for i in reversed(range(self.caffe_inputs.count())):
                self.caffe_inputs.itemAt(i).widget().setParent(None)
            
        return self.caffe_inputs

    def network_forward(self):
        pass
    
