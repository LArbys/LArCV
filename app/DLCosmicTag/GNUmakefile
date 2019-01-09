#
# This is an example GNUmakefile for my packages
#
PACKAGE_NAME = DLCosmicTag

# specific names for this package
SOURCES = $(wildcard *.cxx)
FMWK_HEADERS = LinkDef.h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
#HEADERS += IOManager.inl

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larcv-config --includes)

INCFLAGS += $(shell laropencv-config --includes)
INCFLAGS += $(shell geo2d-config --includes)

ifdef LARLITE_BASEDIR
INCFLAGS += $(shell larlite-config --includes)
INCFLAGS += -I$(LARLITE_BASEDIR)/UserDev/
LDFLAGS  += $(shell geo2d-config --libs)
LDFLAGS  += $(shell larlite-config --libs) -lBasicTool_FhiclLite
else
#SOURCES = $(filter-out ROIMerger.cxx, $(wildcard *.cxx))
#EXCLUDE_HEADERS = LinkDef.h ROIMerger.h
#HEADERS = $(filter-out $(EXCLUDE_HEADERS), $(wildcard *.h))
endif

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

LDFLAGS += $(shell larcv-config --libs)

# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build:
pkg_clean:
