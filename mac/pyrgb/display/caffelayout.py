from .. import QtGui, QtCore

class CaffeLayout(object):

    def __init__(self,tw):
        self.name  = "CaffeLayout"
        
        self.caffe_inputs = QtGui.QGridLayout()

        self.open_deploy = QtGui.QPushButton("Open")
        self.load_config = QtGui.QPushButton("Load")
        self.line_deploy = QtGui.QLineEdit("path/to/deploy")
        self.load_config.clicked.connect(self.loadConfig)
        self.open_deploy.clicked.connect(self.selectFile)

        self.forward = QtGui.QPushButton("Forward")
        self.forward.clicked.connect(self.network_forward)

        self.scores = QtGui.QLabel("Scores:")
        
        self.tw = tw

    def selectFile(self):
        self.line_deploy.setText(QtGui.QFileDialog.getOpenFileName())
        
    def grid(self,enable):
        if enable == True:
            self.caffe_inputs.addWidget(self.line_deploy,0,0)
            self.caffe_inputs.addWidget(self.open_deploy,0,1)
            self.caffe_inputs.addWidget(self.load_config,0,2)
            self.caffe_inputs.addWidget(self.forward,0,3)
            self.caffe_inputs.addWidget(self.scores,1,0)
                
        else:
            for i in reversed(range(self.caffe_inputs.count())):
                self.caffe_inputs.itemAt(i).widget().setParent(None)
                
        return self.caffe_inputs

    def loadConfig(self):
        self.tw.set_config( str(self.line_deploy.text()) )
                
    def network_forward(self):
        self.tw.forward_result()
        
        scores = self.tw.scores[0]
        self.scores.setText("Scores: {}".format(scores))
