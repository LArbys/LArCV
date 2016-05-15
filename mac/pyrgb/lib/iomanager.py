from .. import larcv

# thin iomanager wrapper

class IOManager(object):

    def __init__(self,infiles) :
        assert type(infiles) == list

        self.iom = larcv.IOManager()

        #self.iom.set_verbosity(0)
        
        for f in infiles :
            self.iom.add_in_file(f)

        self.iom.initialize()

    def read_entry(self,entry):
        self.iom.read_entry(entry)

    def get_data(self,product,producer):
        return self.iom.get_data(product,producer)

    def set_verbosity(self,verb):
        assert type(verb) is int
        self.iom.set_verbosity(verb)
