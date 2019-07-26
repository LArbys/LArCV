from .. import larcv

# thin iomanager wrapper

class IOManager(object):

    def __init__(self,infiles=[],outfile=None,IOMode=0,tick_forward=True) :
        assert type(infiles) == list

        tf = 0
        if not tick_forward:
            tf = 1
            #print "Assuming tick-backward order to Image2D data"
        self.tick_forward = tick_forward

        self.iom = larcv.IOManager(IOMode,"pyrgb::IOManager",tf)
        self.iom.logger().send( larcv.msg.kINFO, "IOManager").write( "assuming tick-backward order for image2d\n",
                                                                     len("assuming tick-backward order for image2d\n") )

        for f in infiles :
            self.iom.add_in_file(f)
                
        if outfile is not None:
            self.iom.set_verbosity(0)
            self.iom.set_out_file(outfile)
        else:
            self.iom.set_verbosity(2)
            
        self.iom.initialize()

    def read_entry(self,entry):
        self.iom.read_entry(entry)

    def get_data(self,product,producer):
        return self.iom.get_data(product,producer)

    def set_verbosity(self,verb):
        assert type(verb) is int
        self.iom.set_verbosity(verb)

    def set_id(self,r,s,e):
        self.iom.set_id(r,s,e)

    def save_entry(self):
        self.iom.save_entry()

    def finalize(self):
        self.iom.finalize()
        
    def get_n_entries(self):
        return self.iom.get_n_entries()
