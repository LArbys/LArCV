import sys

if len(sys.argv) < 2:
    msg  = '\n'
    msg += "Usage 1: %s $INPUT_ROOT_FILE(s)\n" % sys.argv[0]
    msg += '\n'
    sys.stderr.write(msg)
    sys.exit(1)
import ROOT
from ROOT import larlite as fmwk
#from larlite import larlite as fmwk

# Create ana_processor instance
my_proc = fmwk.ana_processor()

# Specify IO mode
my_proc.set_io_mode(fmwk.storage_manager.kWRITE)

my_proc.set_output_file("out.root")

# Attach an analysis unit ... here we use a base class which does nothing.
# Replace with your analysis unit if you wish.
unit=fmwk.ChStatusFinder()
unit._out_producer="data"
unit._in_producer="data"
# Set input root file
for x in xrange(len(sys.argv)-1):
    unit._io.add_in_file(sys.argv[x+1])
my_proc.add_process(unit)


print
print  "Finished configuring ana_processor. Start event loop!"
print

# Let's run it.
my_proc.run(0,5)

# done!
print
print "Finished running ana_processor event loop!"
print

sys.exit(0)
