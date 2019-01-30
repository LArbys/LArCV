ifndef LARCV_APPDIR
ERROR_MESSAGE := $(error LARCV_APPDIR is not set... run configure.sh!)
endif

#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS := ImageMod Filter PMTWeights HiResDivider Merger Reco3D
ifdef LARLITE_BASEDIR
SUBDIRS += Supera/APILArLite
endif
ifeq ($(LARCV_NUMPY),1)
SUBDIRS += APICaffe
endif

# if we build with dbscan and approx. nearest neighbor tools
ifeq ($(LARCV_ANN),1)
SUBDIRS += ANN dbscan
ifeq ($(shell uname -s),Linux)
  ANN_OS=linux-g++
endif
ifeq ($(shell uname -s),Darwin)
  ANN_OS=macosx-g++
endif
endif

ifdef LAROPENCV_BASEDIR
SUBDIRS += LArOpenCVHandle
endif

#####################################################################################
#
# COMPILATION...
#
#.phony: all configure default-config clean
.phony: all clean

all:
ifeq (LARCV_ANN,1)
	(cd ann_1.1.2 && $(MAKE) $(ANN_OS)) || exit $$?
endif
	@for i in $(SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $$i && $(MAKE) ) || exit $$?; done



#####################################################################################
#
# CLEANs...
#
clean:
	@for i in $(SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $$i && $(MAKE) clean && rm -rf $(LARCV_BUILDDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/lib/*$ii.* ) || exit $$?; done

#####################################################################################
#
# DOCUMENTATION...
#
doxygen:
	@echo 'dOxygenising LArCV app...'
	@mkdir -p $(LARCV_APPDIR)/doc/dOxygenMyProject
	@doxygen $(LARCV_APPDIR)/doc/doxygenMyProject.script
#
#####################################################################################
#EOF
