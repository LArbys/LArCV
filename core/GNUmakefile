
ifndef LARCV_BASEDIR
ERROR_MESSAGE := $(error LARCV_BASEDIR is not set... run configure.sh!)
endif

#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS := Base DataFormat #Utils
ifeq ($(LARCV_NUMPY),1)
  ifeq ($(LARCV_OPENCV),1)
  SUBDIRS += PyCVUtil
  endif
endif
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
	@for i in $(SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $$i && $(MAKE) clean && rm -rf $(LARCV_BUILDDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/lib ) || exit $$?; done

#####################################################################################
#
# DOCUMENTATION...
#
doxygen:
	@echo 'dOxygenising your code...'
	@mkdir -p $(LARCV_BASEDIR)/doc/dOxygenMyProject
	@doxygen $(LARCV_BASEDIR)/doc/doxygenMyProject.script

doxygen+:
	@echo 'dOxygenising MyProject + local-ROOT...'
	@mkdir -p $(LARCV_BASEDIR)/doc/dOxygenMyProject+
	@doxygen $(LARCV_BASEDIR)/doc/doxygenMyProject+.script
#
#####################################################################################
#EOF
