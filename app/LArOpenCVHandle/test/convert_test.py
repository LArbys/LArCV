import ROOT
from larlite import larlite as fmwk1
from larcv import larcv as fmwk2
from ROOT import handshake

io1=fmwk1.storage_manager(fmwk1.storage_manager.kBOTH)
io1.add_in_file(sys.argv[1])
io1.open()

io2=fmwk2.IOManager(fmwk2.IOManager.kREAD)
io2.add_in_file(sys.argv[2])
io2.initialize()

hs=handshake.HandShaker()

#io1.next_event()
#io1.go_to()

#io2.read_entry()

io1.close()
io2.finalize()


