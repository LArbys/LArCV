#
# This is an example GNUmakefile for my packages
#
PACKAGE_NAME = ROOTUtil

# specific names for this package
SOURCES = $(wildcard *.cxx)
FMWK_HEADERS = LinkDef.h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
HEADERS_DEST = larcv/core/$(PACKAGE_NAME)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += -I$(LARCV_INCDIR)

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build: # nothing extra for this package
pkg_clean: # nothing extra for this package
