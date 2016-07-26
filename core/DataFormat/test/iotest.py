from larcv import larcv
from colored_msg import colored_msg as cmsg
import os, sys
larcv.logger.force_level(0)

#
# Constants
#
MSG_LEVEL=larcv.msg.kERROR
OUT_FNAME="iotest.root"
NUM_EVENT=1000

ERROR_FILE_EXIST      = 1
ERROR_ENTRY_MISSING   = 2
ERROR_PRODUCT_MISSING = 4

if os.path.isfile(OUT_FNAME):
    cmsg.error("Test output file (%s) already exists..." % OUT_FNAME)
    sys.exit(ERROR_FILE_EXIST)
#
# Test Write
#

o=larcv.IOManager(larcv.IOManager.kWRITE)
o.reset()
o.set_verbosity(MSG_LEVEL)
o.set_out_file(OUT_FNAME)
o.initialize()

for idx in xrange(NUM_EVENT):

    for product_type in xrange(larcv.kProductUnknown):

        o.get_data(product_type,"product_type%02d" % product_type)

    o.set_id(0,0,idx)
    o.save_entry()

o.finalize()

#
# Test Read
#

i=larcv.IOManager(larcv.IOManager.kREAD)
i.reset()
i.set_verbosity(MSG_LEVEL)
i.add_in_file(OUT_FNAME)
i.initialize()

product_ctr={}
entry_ctr=0
for idx in xrange(NUM_EVENT):

    if not i.read_entry(idx):
        break
    
    entry_ctr += 1

    for product_type in xrange(larcv.kProductUnknown):

        if product_type not in product_ctr:
            product_ctr[product_type] = 0

        if i.get_data(product_type,"product_type%02d" % product_type):
            product_ctr[product_type] += 1

i.finalize()

error=0
if not entry_ctr == NUM_EVENT:
    cmsg.error("Read-back only found %d/%d events!" % (entry_ctr,NUM_EVENT))
    error += ERROR_ENTRY_MISSING
    
for t,ctr in product_ctr.iteritems():
    if not ctr == NUM_EVENT:
        cmsg.error("Product type %d (name %s) only has %d/%d count!" % (t,larcv.ProductName(t),ctr,NUM_EVENT))
        error += ERROR_PRODUCT_MISSING
        break

if os.path.isfile(OUT_FNAME):
    os.remove(OUT_FNAME)

sys.exit(int(error))
