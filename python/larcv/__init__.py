from __future__ import print_function
import ROOT,os

if not 'LARCV_BASEDIR' in os.environ:
    print('$LARCV_BASEDIR shell env. var. not found (run configure.sh)')
    raise ImportError

# must load dependencies first
# LARLITE
if 'LARLITE_BASEDIR' in os.environ:
    from larlite import larlite
#if 'LAROPENCV_BASEDIR' in os.environ:
#    from larocv import larocv
larcv_dir = os.environ['LARCV_LIBDIR']
# We need to load in order
for l in [x for x in os.listdir(larcv_dir) if x.endswith('.so')]:
    #print("loading larcv lib: ",l)
    ROOT.gSystem.Load(l)
#larcv.Vertex
#larcv.CSVData
#k=larcv.logger # this line to load C++ functions
if 'LARCV_NUMPY' in os.environ and os.environ['LARCV_NUMPY'] == '1':
    from ROOT import larcv
    larcv.load_pyutil
if 'LARCV_OPENCV' in os.environ and os.environ['LARCV_OPENCV'] == '1':
    from ROOT import larcv
    #larcv.load_cvutil
#larcv.load_rootutil
#larcv.LoadImageMod
from ROOT import larcv
