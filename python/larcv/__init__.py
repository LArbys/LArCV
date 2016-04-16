import ROOT,os
if not 'LARCV_BASEDIR' in os.environ:
    print '$LARCV_BASEDIR shell env. var. not found (run configure.sh)'
    raise ImportError
larcv_dir = os.environ['LARCV_BASEDIR']
for l in [x for x in os.listdir(larcv_dir) if x.endswith('.so')]:
    ROOT.gSystem.Load('%s/%s' % (larcv_dir,l))
from ROOT import larcv
if 'LARCV_NUMPY' in os.environ and os.environ['LARCV_NUMPY'] == '1':
    larcv.load_pyutil
