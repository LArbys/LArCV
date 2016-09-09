from larcv import larcv
import os, sys
larcv.logger.force_level(0)

#
# Constants
#
MSG_LEVEL=larcv.msg.kERROR
if 'debug' in sys.argv:
    MSG_LEVEL = larcv.msg.kDEBUG
if 'info' in sys.argv:
    MSG_LEVEL = larcv.msg.kINFO

OUT_FNAME="merger.root"
NUM_EVENT=1

ERROR_FILE_EXIST      = 1
ERROR_WRITE_INIT      = 2

# if os.path.isfile(OUT_FNAME):
#     print "Test output file (%s) already exists..." % OUT_FNAME
#     sys.exit(ERROR_FILE_EXIST)

from larcv import larcv
o=larcv.IOManager(larcv.IOManager.kWRITE)
o.reset()
o.set_verbosity(MSG_LEVEL)
o.set_out_file(OUT_FNAME)

stream1 = larcv.DataStream("DataStream1")
stream1.set_verbosity(0)
print "READING: ",sys.argv[2]
cfg1 = larcv.CreatePSetFromFile(sys.argv[2],"DataStream1")
stream1.configure(cfg1)

stream2 = larcv.DataStream("DataStream2")
stream2.set_verbosity(0)
print "READING: ",sys.argv[3]
cfg2 = larcv.CreatePSetFromFile(sys.argv[3],"DataStream2")
stream2.configure(cfg2)

p = larcv.ImageMerger()
p.set_verbosity(0)

cfg = larcv.CreatePSetFromFile(sys.argv[1],"ImageMerger")
p.CosmicImageHolder(stream2)
p.NeutrinoImageHolder(stream1)

p.configure(cfg)
p.initialize()

if not o.initialize():
    sys.exit(ERROR_WRITE_INIT)

for idx in xrange(NUM_EVENT):
    
    # sys.stdout.write('On event: {} \r'.format(idx))
    # sys.stdout.flush() 
    
    print 'On event: {}'.format(idx)
    
    img1 = larcv.Image2D(10,10)
    for x in xrange(img1.as_vector().size()):
        if x%2 == 0: img1.set_pixel(x,10)
        else: img1.set_pixel(x,0)
    
    img2 = larcv.Image2D(10,10)
    for x in xrange(img2.as_vector().size()):
        if x%2 == 0: img2.set_pixel(x,0)
        else: img2.set_pixel(x,10)
        
    event_image1_tpc = o.get_data(larcv.kProductImage2D,"stream1_tpc")
    event_image1_tpc.Append(img1)
    event_image1_pmt = o.get_data(larcv.kProductImage2D,"stream1_pmt")
    event_image1_pmt.Append(img1)

    #i have to make a fake ROI because of fantastic ImageMerger assumptions
    event_roi1 = o.get_data(larcv.kProductROI,"stream1")
    event_roi1.Append(larcv.ROI())

    event_image2_tpc = o.get_data(larcv.kProductImage2D,"stream2_tpc")
    event_image2_tpc.Append(img2)
    event_image2_pmt = o.get_data(larcv.kProductImage2D,"stream2_pmt")
    event_image2_pmt.Append(img2)


    stream1.process(o)
    stream2.process(o)
    p.process(o)
    o.set_id(0,0,idx)
    print "SAVING",o.event_id().event_key()
    o.save_entry()

    idx+=1

stream1.finalize()
stream2.finalize()
p.finalize()
o.finalize()
