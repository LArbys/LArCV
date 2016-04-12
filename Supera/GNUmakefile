
ifndef LARCV_SUPERADIR
ERROR_MESSAGE := $(error LARCV_SUPERADIR is not set... run configure.sh!)
endif

ifndef LARLITE_BASEDIR
ERROR_MESSAGE := $(error LARLITE_BASEDIR is not set...)
endif

#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS := APILArLite

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
	@for i in $(SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $$i && $(MAKE) clean && rm -rf $(LARCV_BUILDDIR)/$$i && rm -rf $(LARCV_LIBDIR)/*$$i.* ) || exit $$?; done

#####################################################################################
#
# DOCUMENTATION...
#
doxygen:
	@echo 'dOxygenising Supera code...'
	@mkdir -p $(LARCV_SUPERADIR)/doc/dOxygenMyProject
	@doxygen $(LARCV_SUPERADIR)/doc/doxygenMyProject.script

#
#####################################################################################
#EOF
