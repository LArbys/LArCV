from .. import QtGui, QtCore

class CaffeLayout(object):

    def __init__(self,tw):
        self.name  = "CaffeLayout"
        
        self.caffe_inputs = QtGui.QGridLayout()

        self.caffe_label = QtGui.QLabel("<b>Caffe Integration</b>")
        self.open_deploy = QtGui.QPushButton("Open")
        self.load_config = QtGui.QPushButton("Load")
        self.line_deploy = QtGui.QLineEdit("absolute path to configuration YAML")
        self.load_config.clicked.connect(self.loadConfig)
        self.open_deploy.clicked.connect(self.selectFile)

        self.forward = QtGui.QPushButton("Forward")
        self.forward.clicked.connect(self.network_forward)

        self.loaded_config = QtGui.QLabel("")
        self.scores = QtGui.QLabel("")
        
        self.tw = tw

    def selectFile(self):
        self.line_deploy.setText(QtGui.QFileDialog.getOpenFileName())
        
    def grid(self,enable):
        if enable == True:
            self.caffe_inputs.addWidget(self.caffe_label,0,0)
            self.caffe_inputs.addWidget(self.line_deploy,1,0)
            self.caffe_inputs.addWidget(self.open_deploy,1,1)
            self.caffe_inputs.addWidget(self.load_config,1,2)
            self.caffe_inputs.addWidget(self.forward,1,3)
            self.caffe_inputs.addWidget(self.loaded_config,2,0)
            self.caffe_inputs.addWidget(self.scores,3,0)
                
        else:
            for i in reversed(range(self.caffe_inputs.count())):
                self.caffe_inputs.itemAt(i).widget().setParent(None)
                
        return self.caffe_inputs

    def loadConfig(self):
        cfg = str( self.line_deploy.text() )
        self.tw.set_config( cfg )
        self.loaded_config.setText("<b>Loaded:</b> {}".format( cfg ) )
                
    def network_forward(self):
        self.tw.forward_result()
        self.scores.setText("<b>Scores:</b> {}".format(self.tw.scores))
