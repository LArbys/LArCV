#
# This is an example GNUmakefile for my packages
#
PACKAGE_NAME = showermask

# specific names for this package
SOURCES = $(filter-out readimg.cxx, $(wildcard *.cxx))
FMWK_HEADERS = LinkDef.h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
#HEADERS += IOManager.inl

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larlite-config --includes)
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += $(shell basictool-config --includes)
INCFLAGS += $(shell recotool-config --includes)
INCFLAGS += -I$(LARLITE_COREDIR)
INCFLAGS += -I$(LAROPENCV_BASEDIR)
INCFLAGS += -I$(GEO2D_BASEDIR)
INCFLAGS += -I$(LARLITE_USERDEVDIR)
INCFLAGS += -I$(LARCV_APPDIR)/UBWireTool
ifeq ($(LARCV_OPENCV),1)
INCFLAGS += -DLARCV_OPENCV
endif

LDFLAGS += -L$(LARLITE_LIBDIR)
LDFLAGS += $(shell larlite-config --libs)
LDFLAGS += -lLArOpenCV_Core
LDFLAGS += -lLArOpenCV_ImageClusterBase
LDFLAGS += -lLArOpenCV_ImageClusterAlgoData
LDFLAGS += -lLArOpenCV_ImageClusterAlgoModule
LDFLAGS += -lBasicTool_FhiclLite

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

LDFLAGS += $(shell larcv-config --libs)
# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build:
#	@cp -f bin/run_merger $(LARCV_BINDIR)/run_merger
pkg_clean:
#	@rm -f $(LARCV_BINDIR)/run_merger
