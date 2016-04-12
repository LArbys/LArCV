from ROOT import larcv
larcv.IOManager
from ROOT import TChain
import sys

ROI_PRODUCER='supera_event'

roi_tree_name='partroi_%s_tree' % ROI_PRODUCER
roi_br_name='partroi_%s_branch' % ROI_PRODUCER
roi_ch = TChain(roi_tree_name)
roi_ch.AddFile(sys.argv[1])

for entry in xrange(roi_ch.GetEntries()):
    roi_ch.GetEntry(entry)
    roi_br=None
    exec('roi_br=roi_ch.%s' % roi_br_name)
    print
    print roi_br.event_key()
    bb_array = roi_br.ROIArray()
    for b in bb_array:
        print b.dump()



