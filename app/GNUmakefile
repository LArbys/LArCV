
ifndef LARCV_APPDIR
ERROR_MESSAGE := $(error LARCV_APPDIR is not set... run configure.sh!)
endif

#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS := ImageMod Filter PMTWeights HiResDivider Merger APICaffe
ifdef LARLITE_BASEDIR
SUBDIRS += Supera/APILArLite
endif

#####################################################################################
#
# COMPILATION...
#
#.phony: all configure default-config clean
.phony: all clean

all:
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
