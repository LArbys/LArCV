#####################################################################################
#
# A top Makefile for building my project.
# One needs to define $LARCV_BASEDIR environment variable and set it to where this
# makefile exists. 
# One can type "make" and this builds packages that are added in $SUBDIR defined below.
# 
# The original is taken from Glenn A. Smith's example for Double Chooz experiment.
#
#####################################################################################
#
# IMPOSE CONDITION BETWEEN LARCV_BASEDIR & PWD =>
#   do not compile if PWD !=$LARCV_BASEDIR is set elsewhere
#
ifndef LARCV_BASEDIR
 LARCV_BASEDIR := $(shell cd . && pwd -P)
endif
NORMALIZED_LARCV_BASEDIR := $(shell cd ${LARCV_BASEDIR} && pwd -P)
ifneq ($(NORMALIZED_LARCV_BASEDIR), $(shell cd . && pwd -P))
 ifneq ($(wildcard ./Base/*),)
ERROR_MESSAGE := $(error Your source code detected in current dir, but LARCV_BASEDIR is not current dir.  \
   To avoid recompiling the wrong installation,\
   LARCV_BASEDIR must be set to the current directory when making.  \
   Currently it is set to ${LARCV_BASEDIR} [$(NORMALIZED_LARCV_BASEDIR)].  \
   Current directory is $(shell pwd).)
 endif
endif
export LARCV_BASEDIR
#
#####################################################################################
#
# Define directories to be compile upon a global "make"...
#
SUBDIRS := Base DataFormat #Utils
ifeq ($(LARCV_NUPY)$(LARCV_OPENCV),"11")
SUBDIRS += PyCVUtil
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
