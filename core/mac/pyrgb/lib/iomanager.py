from .. import larcv

# thin iomanager wrap
class IOManager(object):

    def __init__(self,infiles) :
        assert type(infiles) == list

        self.iom = larcv.IOManager()

        self.iom.set_verbosity(0)
        
        for f in infiles :
            self.iom.add_in_file(f)

        self.iom.initialize()


    # you should just have access to it
    # def iom(self):
    #     return self.iom

    def search_for_next_image(self):
        return None
