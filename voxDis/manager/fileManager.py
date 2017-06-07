import ROOT
from larcv import larcv
from pyqtgraph.Qt import QtCore

class fileManager(QtCore.QObject):
    redrawRequested = QtCore.pyqtSignal()

    """docstring for fileManager"""
    def __init__(self):
        super(fileManager, self).__init__()
        self._files = []

        # Use IOManager to read the files, why reinvent the wheel?
        self._mgr = None

    def set_input_files(self, files):
        if type(files) is str:
            self._files = []
            self._files.append(files)
        else:
            self._files = files
        self.open_files()


    def open_files(self):
        if len(self._files) == 0:
            return

        if self._mgr is None:
            self._mgr = larcv.IOManager()
        else:
            self._mgr.reset()

        for _file in self._files:
            self._mgr.add_in_file(_file)

        self._mgr.initialize()

    def current_run(self):
        if self._mgr is not None:
            return self._mgr.event_id().run()
        else:
            return 0

    def current_subrun(self):
        if self._mgr is not None:
            return self._mgr.event_id().subrun()
        else:
            return 0

    def current_event(self):
        if self._mgr is not None:
            return self._mgr.event_id().subrun()
        else:
            return 0

    def current_entry(self):
        if self._mgr is not None:
            return self._mgr.current_entry()
        else:
            return 0    

    def get_n_entries(self):
        if self._mgr is not None:
            return self._mgr.get_n_entries()
        else:
            return 0    

    def next(self):
        if self.current_entry() + 1 < self.get_n_entries():
            self.goToEntry(self.current_entry() + 1)
        else:
            print("Can't go to the next entry, currently on the last entry.")

    def previous(self):
        if self.current_entry() > 0:
            self.goToEntry(self.current_entry() - 1)
        else:
            print("Can't go to the previous entry, currently on the first entry.")


    def goToEntry(self, entry):

        self._mgr.read_entry(entry)
        self.redrawRequested.emit()


    def get_image_2D(self, plane):
        if self._mgr is not None:
            _producer = self._mgr.producer_list(larcv.kProductImage2D).front()
            image = self._mgr.get_data(larcv.kProductImage2D, _producer)
            return image.at(plane)
        else:
            return None

    def get_image_3D(self):
        if self._mgr is not None:
            _producer = self._mgr.producer_list(larcv.kProductVoxel3D).front()
            image = self._mgr.get_data(larcv.kProductVoxel3D, _producer)
            return image
        # Get the data that is type "voxel3D"
        else:
            return None
